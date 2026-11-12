"""RDP accounting for DP-FedAvg: clip the client update, noise the aggregate.

One Gaussian release per ROUND, not per gradient step, so the composition is
over ``rounds`` and not over ``rounds x steps``. That is the whole reason this
mechanism is affordable here.

NO SUBSAMPLING AMPLIFICATION IS CLAIMED. All 8 hospitals participate in every
round, so q = 1 and the usual "sample a few clients" discount does not apply.
Claiming it would understate epsilon by a large factor, which is the standard
way DP numbers end up wrong in papers.

The unit of privacy is the CLIENT: with sensitivity = clip norm S, the
guarantee hides whether any one hospital was in the federation. It does NOT by
itself give a per-patient guarantee -- that needs record-level clipping inside
local training as well.

Gaussian RDP: eps_rdp(alpha) = alpha / (2 * sigma^2) per release. Composition
adds over releases, then converts once.
"""
from __future__ import annotations

import math


def epsilon(sigma: float, rounds: int, delta: float = 1e-5) -> tuple:
    """Return ``(epsilon, best_alpha)`` for ``rounds`` Gaussian releases."""
    if sigma <= 0:
        return float("inf"), float("nan")
    best = (float("inf"), None)
    # Scan orders rather than using the closed-form optimum: the conversion has
    # a log(1/delta)/(alpha-1) term, and the optimum moves with rounds.
    a = 1.01
    while a < 512:
        rdp = rounds * a / (2 * sigma ** 2)
        eps = rdp + math.log(1 / delta) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
        a *= 1.02
    return best


def sgm_epsilon(sigma_g: float, sketch_b: int, rounds: int, clip: float = 1.0,
                delta: float = 1e-5) -> tuple:
    """Return ``(epsilon, best_alpha)`` for ``rounds`` Fed-SGM releases.

    Lemma 3.4 of Li, Chen & Banerjee (2025) gives one release as
    ``(alpha, alpha^2 tau^4 / ((alpha - 1) b sigma_g^4))``-RDP. The paper states
    no composed total, so this adds RDP over rounds and converts once, as
    :func:`epsilon` does, with q = 1. The noise is applied per client and the
    bound is per client, so averaging over clients earns no credit here.

    Raises:
        ValueError: If ``b * sigma_g^2 <= 2 * clip^2``, where the lemma's bound
            is undefined.
    """
    if sigma_g <= 0:
        return float("inf"), float("nan")
    if sketch_b * sigma_g ** 2 <= 2 * clip ** 2:
        raise ValueError(f"need b*sigma_g^2 > 2*clip^2, got b={sketch_b}, "
                         f"sigma_g={sigma_g}, clip={clip}")
    per = rounds * clip ** 4 / (sketch_b * sigma_g ** 4)
    best = (float("inf"), None)
    a = 1.01
    while a < 512:
        eps = (per * a ** 2 + math.log(1 / delta)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
        a *= 1.02
    return best


def sgm_sigma_for(target_eps: float, sketch_b: int, rounds: int,
                  clip: float = 1.0, delta: float = 1e-5) -> float:
    """Smallest ``sigma_g`` reaching ``target_eps``, by bisection."""
    lo, hi = (2.01 * clip ** 2 / sketch_b) ** 0.5, 100.0 * clip
    for _ in range(80):
        mid = (lo + hi) / 2
        if sgm_epsilon(mid, sketch_b, rounds, clip, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    D = 1e-5
    print(f"client-level DP-FedAvg, all 8 sites every round, delta={D:g}")
    print(f"{'sigma':>7} | " + " | ".join(f"R={r:<3}" for r in (10, 25, 50)))
    for s in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        cells = []
        for r in (10, 25, 50):
            e, _ = epsilon(s, r, D)
            cells.append(f"{e:6.2f}")
        print(f"{s:>7} | " + " | ".join(cells))


# ---------------------------------------------------------------------------
# Subsampled Gaussian (DP-SGD). The plain-Gaussian accountant above is for
# DP-FedAvg, where one release per round sees ALL clients (q = 1). DP-SGD
# releases one noisy gradient per STEP over a random minibatch, and that
# subsampling is what makes thousands of releases affordable -- but only if
# q < 1. At q = 1 the amplification vanishes and the two coincide.

def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of the subsampled Gaussian at INTEGER order ``alpha``.

    Mironov et al. (2019), the binomial expansion:

        eps(a) = 1/(a-1) * log( sum_k C(a,k) (1-q)^(a-k) q^k exp(k(k-1)/(2 s^2)) )

    Computed in log space: the exp term reaches e^600 at the orders that matter,
    which overflows a float otherwise.
    """
    if q <= 0:
        return 0.0
    if q >= 1.0:                      # no amplification; plain Gaussian
        return alpha / (2 * sigma ** 2)
    terms = [_log_comb(alpha, k) + (alpha - k) * math.log1p(-q)
             + k * math.log(q) + (k * (k - 1)) / (2 * sigma ** 2)
             for k in range(alpha + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / (alpha - 1)


def epsilon_sgd(sigma: float, q: float, steps: int, delta: float = 1e-5) -> tuple:
    """``(epsilon, best_alpha)`` for ``steps`` subsampled-Gaussian releases."""
    best = (float("inf"), None)
    for a in range(2, 256):
        try:
            e = steps * rdp_subsampled_gaussian(q, sigma, a) \
                + math.log(1 / delta) / (a - 1)
        except (OverflowError, ValueError):
            continue
        if e < best[0]:
            best = (e, a)
    return best
