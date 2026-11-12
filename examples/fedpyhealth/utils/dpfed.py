"""DP-FedAvg: clip each client's update, noise the aggregate.

The mechanism is McMahan et al. (2018), not DP-SGD. The difference matters for
both cost and meaning:

* **Cost.** One Gaussian release per ROUND, not one per gradient step. At 10
  rounds that is 10 compositions instead of ~10,000, which is why this is
  affordable on a cohort where the smallest site has 142 patients. See
  ``dpaccount.py`` for the epsilon table.
* **Meaning.** The unit protected is the CLIENT: swapping out ALL of one
  hospital's data barely changes the released model. This is the STRONGER of
  the two usual guarantees, not the weaker one. Replacing a single patient's
  record is a special case of changing one hospital's data, so hospital-level
  epsilon implies patient-level epsilon at the same value or better -- the
  standard user-level-vs-example-level argument from McMahan et al. (2018).
  State it as hospital-level and note the implication; do NOT claim only
  hospital-level protection, which undersells it.

WHAT IS CLIPPED. The client DELTA (its trained weights minus the weights it was
broadcast), never the weights themselves. Clipping raw weights would shrink the
model toward zero every round, which is a different algorithm and a broken one.

NO SUBSAMPLING AMPLIFICATION. All 8 hospitals participate every round, so
q = 1. The accountant does not claim a discount, and neither should the paper.

WHY THE NOISE IS SCALED BY ``clip / n_clients``. The released quantity is the
MEAN update. One client's delta can move that mean by at most ``clip / n``, so
that is the L2 sensitivity, and the Gaussian must be scaled to it. Scaling to
``clip`` alone over-noises by a factor of n; scaling to ``clip * w_i / sum(w)``
would be correct only for uniform weights. Sample-count weighting makes the
largest site's influence bigger than 1/n, so with ``--weighting sample`` the
sensitivity claim below is NOT valid -- the caller must use uniform weights,
and ``dp_aggregate`` enforces it rather than silently under-noising.
"""
from __future__ import annotations

import math
from typing import Collection, Dict, List, Tuple

import torch


def _delta(state: Dict[str, torch.Tensor],
           ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Client update: what training changed, relative to the broadcast."""
    return {k: (v.to(torch.float64) - ref[k].to(torch.float64))
            for k, v in state.items() if torch.is_floating_point(v)}


def update_norm(delta: Dict[str, torch.Tensor]) -> float:
    """Global L2 norm of one client's update, across all float tensors."""
    return float(torch.sqrt(sum((d * d).sum() for d in delta.values())))


def clip_delta(delta: Dict[str, torch.Tensor], clip: float
               ) -> Tuple[Dict[str, torch.Tensor], float]:
    """Scale the update down to ``clip`` if it is longer. Returns (delta, norm).

    Scale-down only: a short update is left alone rather than inflated, which
    is what bounds sensitivity without adding signal that was never there.
    """
    n = update_norm(delta)
    if n > clip and n > 0:
        f = clip / n
        delta = {k: v * f for k, v in delta.items()}
    return delta, n


def dp_aggregate(
    states: List[Dict[str, torch.Tensor]],
    ref: Dict[str, torch.Tensor],
    clip: float,
    sigma: float,
    weighting: str = "uniform",
    generator: torch.Generator = None,
    keys: Collection[str] = None,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """One DP-FedAvg round: clip every client delta, average, add noise.

    Args:
        states: Each client's post-training state dict.
        ref: The state broadcast to the clients at the start of the round.
        clip: L2 clip norm S applied to each client delta.
        sigma: Noise multiplier. Gaussian std is ``sigma * clip / n_clients``.
            ``0`` means no noise -- the control arm, which still clips so that
            clipping and noise can be attributed separately.
        weighting: Must be "uniform". See the module docstring: sample-count
            weighting breaks the sensitivity bound.
        generator: Optional torch Generator, for reproducible noise.
        keys: The trained parameters. Everything else rides along unchanged.
            Testing for a float dtype is not enough: HALO's causal attention
            mask is a float buffer, read as ``w * b - 1e10 * (1 - b)``, so
            noise on it swamps attention. Every run before this argument
            existed noised that mask.

    Returns:
        ``(new_state, norms)`` -- the aggregated state dict, and each client's
        PRE-clip update norm, which is what you read to choose ``clip``.

    Raises:
        ValueError: If ``weighting`` is not uniform, or ``clip <= 0``.
    """
    if weighting != "uniform":
        raise ValueError(
            "DP-FedAvg requires --weighting uniform. With sample-count weights "
            "one client's influence on the mean exceeds clip/n, so the noise "
            "scale below would understate the sensitivity and the reported "
            "epsilon would be wrong (too small).")
    if clip <= 0:
        raise ValueError(f"--dp-clip must be positive, got {clip}")

    n = len(states)
    deltas, norms = [], []
    for st in states:
        d, raw = clip_delta(_delta(st, ref), clip)
        deltas.append(d)
        norms.append(raw)

    out: Dict[str, torch.Tensor] = {}
    for key, base in ref.items():
        if not torch.is_floating_point(base) or (keys is not None
                                                  and key not in keys):
            out[key] = base.clone()          # buffers ride along unchanged
            continue
        acc = torch.zeros_like(base, dtype=torch.float64)
        for d in deltas:
            acc += d[key]
        acc /= n
        if sigma > 0:
            std = sigma * clip / n
            noise = torch.normal(0.0, std, size=acc.shape, dtype=torch.float64,
                                 device=acc.device, generator=generator)
            acc += noise
        out[key] = (base.to(torch.float64) + acc).to(base.dtype)
    return out, norms


def sgm_aggregate(
    states: List[Dict[str, torch.Tensor]],
    ref: Dict[str, torch.Tensor],
    clip: float,
    sigma: float,
    sketch_b: int,
    keys: Collection[str],
    round_seed: int,
    weighting: str = "uniform",
    device: str = "cpu",
    chunk: int = 512,
    generator: torch.Generator = None,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """One Fed-SGM round (Li, Chen & Banerjee 2025, Algorithm 2).

    Each client clips its update, sketches it with the round's shared Gaussian
    matrix ``R`` (b x d, entries of variance 1/b), and adds ``N(0, sigma^2)`` to
    every sketch coordinate BEFORE sending. The mean sketch is expanded back
    with ``R^T``. Unlike :func:`dp_aggregate`, the noise is added at the client,
    so the server does not need to be trusted.

    ``R`` is never stored. At 6.2M parameters and b = 100,000 it would hold
    6e11 entries, so it is drawn ``chunk`` rows at a time from a seeded
    generator and each block is applied and discarded, which needs only
    ``chunk * d`` floats at once. The result is the same as holding all of it.

    Args:
        states: Each client's post-training state; holds at least ``keys``.
        ref: The complete state broadcast at the start of the round.
        clip: L2 clip norm tau on each client delta.
        sigma: Noise std ``sigma_g`` per sketch coordinate. Absolute, NOT a
            multiplier: :func:`dpaccount.sgm_epsilon` takes the same value.
        sketch_b: Sketch dimension b.
        keys: Parameters to sketch. Only these are noised; see
            :func:`dp_aggregate` for why this cannot be inferred from dtype.
        round_seed: Seeds ``R``, which is public and fresh each round.
        weighting: Must be "uniform".
        device: Where the sketch runs. ``chunk`` x d floats must fit here.
        chunk: Rows of ``R`` held at once.
        generator: Optional generator for the client noise.

    Returns:
        ``(new_state, norms)``: the complete updated state, and each client's
        pre-clip update norm.

    Raises:
        ValueError: If ``weighting`` is not uniform, or a setting is invalid.
    """
    if weighting != "uniform":
        raise ValueError("Fed-SGM requires --weighting uniform: the bound is "
                         "per client, and unequal weights change the mean.")
    if clip <= 0 or sketch_b <= 0:
        raise ValueError(f"need clip > 0 and sketch_b > 0, got {clip}, "
                         f"{sketch_b}")

    names = [k for k in ref if k in keys]
    n = len(states)
    rows, norms = [], []
    for st in states:
        d, raw = clip_delta(_delta({k: st[k] for k in names}, ref), clip)
        norms.append(raw)
        rows.append(torch.cat([d[k].reshape(-1) for k in names]).float())
    updates = torch.stack(rows).to(device)            # (n_clients, d)
    dim = updates.shape[1]

    r_gen = torch.Generator(device=device).manual_seed(int(round_seed))
    mean_update = torch.zeros(dim, device=device)
    for start in range(0, sketch_b, chunk):
        m = min(chunk, sketch_b - start)
        r = torch.randn(m, dim, generator=r_gen, device=device) / math.sqrt(
            sketch_b)
        sketches = r @ updates.T                      # (m, n_clients)
        if sigma > 0:
            sketches = sketches + sigma * torch.randn(
                m, n, device=device, generator=generator)
        mean_update += r.T @ sketches.mean(dim=1)
        del r, sketches

    out = {k: v.clone() for k, v in ref.items()}
    mean_update = mean_update.cpu()
    at = 0
    for k in names:
        size = ref[k].numel()
        step = mean_update[at:at + size].view_as(ref[k])
        out[k] = (ref[k].to(torch.float64) + step.to(torch.float64)
                  ).to(ref[k].dtype)
        at += size
    return out, norms


# ---------------------------------------------------------------------------
# ADAPTIVE CLIPPING -- Andrew et al. (2021), "Differentially Private Learning
# with Adaptive Clipping".
#
# WHY. The probe measured client update norms falling ~30x over 10 rounds
# (6.8 -> 0.24) and differing ~2.5x between the largest and smallest site. A
# fixed C is therefore wrong at one end of training or the other, and wrong for
# one size of hospital or the other. Adaptive clipping removes the choice: aim
# C at a target QUANTILE of the norms and let it follow them down.
#
# HOW. Each client also reports one bit, b_i = 1 if its update was NOT clipped.
# Their mean b_bar estimates the fraction below C. If more than the target
# fraction are below, C is too loose and shrinks; if fewer, it grows:
#
#     C <- C * exp(-eta * (b_bar - gamma))
#
# Geometric, so C stays positive and moves multiplicatively -- which is the
# right scale for a quantity that decays by 30x.
#
# THE BIT IS NOT FREE. b_bar is a second release about the same private data,
# so it must be noised too, and its privacy cost adds. b_bar has sensitivity
# 1/n (one client flips one bit), so noise std sigma_b/n gives it noise
# multiplier sigma_b. RDP adds across the two Gaussians:
#
#     eps_rdp(a) = a/(2 sigma^2) + a/(2 sigma_b^2)
#
# i.e. an EFFECTIVE noise multiplier  1/sigma_eff^2 = 1/sigma^2 + 1/sigma_b^2.
# Use ``effective_sigma`` below when computing epsilon -- quoting the update's
# sigma alone would understate it. Andrew et al. take sigma_b large relative to
# sigma so the overhead is small; sigma_b = 4*sigma costs ~3% in sigma_eff.


def effective_sigma(sigma: float, sigma_b: float) -> float:
    """Noise multiplier to feed the accountant when adaptive clipping is on.

    Both the noisy update and the noisy quantile bit are releases about the
    same data, so their RDP adds. Quoting ``sigma`` alone reports an epsilon
    the mechanism does not achieve.
    """
    if sigma_b is None or sigma_b <= 0:
        return sigma
    return (1.0 / (1.0 / sigma ** 2 + 1.0 / sigma_b ** 2)) ** 0.5


def adapt_clip(clip: float, norms: List[float], target: float = 0.5,
               lr: float = 0.2, sigma_b: float = 0.0,
               rng=None) -> Tuple[float, float]:
    """One geometric update of the clip norm. Returns ``(new_clip, b_bar)``.

    Args:
        clip: Current C.
        norms: This round's PRE-clip client norms.
        target: Quantile to track. 0.5 aims C at the median, so about half the
            clients are clipped each round.
        lr: Step size for the geometric update.
        sigma_b: Noise multiplier on the reported fraction. 0 means an EXACT
            fraction, which is NOT private -- measurement mode only.
        rng: Optional ``random.Random`` for reproducibility.
    """
    n = len(norms)
    b_bar = sum(1.0 for x in norms if x <= clip) / n
    if sigma_b > 0:
        import random
        r = rng or random
        b_bar += r.gauss(0.0, sigma_b / n)
    return clip * math.exp(-lr * (b_bar - target)), b_bar
