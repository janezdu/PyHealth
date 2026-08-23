"""Per-patient loss weights that scale with how rare a patient's codes are
*at their own hospital*.

The premise this tests: FedAvg averages eight generators that each saw a
different tail. A code with 0.08% prevalence at hospital 458 contributes ~1.4
patients to an 1,832-patient training set, so the gradient it produces is
indistinguishable from noise and the averaged model learns the head of every
site and the tail of none. Weighting each patient by the rarity of the codes
they carry gives the tail a voice proportional to how rare it is, without
touching the data, the split, or the architecture.

Rarity is judged LOCALLY and per patient. Hospital 429 (142 train patients) has
a completely different notion of rare from 458 (1,832), and a patient is
upweighted by their own site's view even in the pooled ``centralized`` arm --
so all four regimes apply one identical rule and stay comparable.

What "proportional" means here
------------------------------
    raw(p) = 1 + SUM over the local-rare codes p carries of
                 rare_threshold / prevalence_h(c)

so a 0.1%-prevalence code counts ten times a 1% one, and a code sitting exactly
at the rarity threshold counts 1 -- the smallest amount that still registers as
rare. Dividing by the threshold is what puts the scale right: "rare" here means
prevalence <= 0.05, so a bare ``1 / prevalence`` is >= 20 for EVERY rare code by
construction, and any cap below 20 collapses the whole thing into a binary
carries-a-rare-code indicator. Measuring rarity relative to the threshold makes
the sum say how far into the tail a patient reaches, which is the quantity the
experiment is about.

Two corrections make that usable rather than degenerate:

* **Cap, relative to the unweighted baseline.** A patient carrying no rare code
  has raw weight 1, so capping at ``max_ratio`` says directly: the rarest
  patient counts at most ``max_ratio`` times the most ordinary one. That bound
  is the knob worth arguing about, and it is stated rather than emergent.

  Capping at a *quantile* of the raw distribution was tried first and is wrong
  here: eICU's deepest local tail sits near 0.0008, giving raw weights up to
  3,530, and ~85% of patients carry at least one local-rare code -- so the 99th
  percentile is itself ~2,000 and clips almost nothing. Mean-normalising that
  distribution sent no-rare-code patients to weight 0.003 and the median patient
  to 0.414: a 2,550x spread that deletes the common cohort rather than
  upweighting the rare one. The model needs the common structure in order to
  place a rare code in context.
* **Mean-normalisation to 1.** This one is load-bearing, not cosmetic.
  ``BCELoss`` with the default mean reduction divides by the element count, NOT
  by the sum of weights (verified in
  ``tests/core/test_halo_sample_weights.py``), so scaling every weight by c
  scales the loss and every gradient by c. Weights averaging 20 would be a 20x
  learning-rate increase wearing a disguise, and the run would differ from the
  unweighted baseline for two reasons at once with no way to separate them.
  Holding the mean at 1 leaves step size where it was, so the only thing that
  changed is which patients the gradient listens to.

Prevalence is measured on the TRAIN fold only. Using val or test would let the
evaluation folds influence what the generator pays attention to.
"""

import os
from collections import Counter
from typing import Callable, Dict, List, Sequence

from utils.cohort import load_manifest, read_trajectories

#: How much more a patient can count than one carrying no rare code at all.
#: 20 gives the tail a strong voice while leaving the common cohort a real
#: share of the gradient; at 1.0 this reduces exactly to unweighted training.
DEFAULT_MAX_RATIO = 20.0


def local_prevalence(cache_dir: str, fold: str = "train"
                     ) -> Dict[str, Dict[str, float]]:
    """``{hospital: {code: prevalence at that hospital}}`` on one fold.

    The manifest carries each hospital's rare code *list* but not the
    prevalences behind it, and rarity is the whole quantity here -- so it is
    recomputed from the trajectories rather than approximated by the cohort-wide
    number, which would erase exactly the per-site difference being tested.
    """
    out: Dict[str, Dict[str, float]] = {}
    for hid, patients in read_trajectories(cache_dir, fold).items():
        n = max(1, len(patients))
        c = Counter()
        for visits in patients.values():
            for code in {x for v in visits for x in v}:   # per PATIENT, not per visit
                c[code] += 1
        out[hid] = {code: k / n for code, k in c.items()}
    return out


def patient_weights(cache_dir: str, fold: str = "train",
                    max_ratio: float = DEFAULT_MAX_RATIO,
                    log: Callable[[str], None] = print
                    ) -> Dict[str, float]:
    """``{patient_id: weight}`` over every patient in ``fold``, mean 1 per site.

    Normalisation is per HOSPITAL, not pooled: each client's own batches should
    average to 1 so no site silently trains at a different effective learning
    rate from the others, which would confound FedAvg's averaging with a step
    size difference.

    Args:
        cache_dir: Cohort cache.
        fold: Fold whose patients get weights, and whose prevalences define
            rarity. Keep this ``train``.
        max_ratio: Ceiling on how many times more a patient may count than one
            carrying no local-rare code. 1.0 reduces to unweighted training.
        log: Progress sink.

    Returns:
        ``{patient_id: weight}``. Patients carrying no local-rare code get the
        smallest weight in their site, not zero -- they still carry the common
        structure the model needs in order to place a rare code in context.
    """
    manifest = load_manifest(cache_dir)
    rare_by_hosp = {h: set(v["rare_codes"])
                    for h, v in manifest["per_hospital"].items()}
    # The threshold rarity is measured against -- read from the cohort rather
    # than hardcoded, so a cache built with a different rare_prevalence_max
    # still produces weights on the intended scale.
    thresh = float(manifest.get("rare_prevalence_max") or 0.05)
    prev = local_prevalence(cache_dir, fold)
    traj = read_trajectories(cache_dir, fold)

    weights: Dict[str, float] = {}
    for hid, patients in traj.items():
        rare, p_h = rare_by_hosp.get(hid, set()), prev.get(hid, {})
        raw: Dict[str, float] = {}
        for pid, visits in patients.items():
            codes = {x for v in visits for x in v} & rare
            raw[pid] = 1.0 + sum(thresh / p_h[c] for c in codes
                                 if p_h.get(c, 0) > 0)
        capped = {pid: min(w, max_ratio) for pid, w in raw.items()}
        mean = sum(capped.values()) / max(1, len(capped))
        for pid, w in capped.items():
            weights[pid] = w / mean
        n_rare = sum(1 for pid, w in raw.items() if w > 1.0)
        n_cap = sum(1 for w in raw.values() if w > max_ratio)
        lo, hi = min(capped.values()) / mean, max(capped.values()) / mean
        log(f"  {hid}: {len(patients)} patients, {n_rare} carry a local-rare "
            f"code, {n_cap} hit the {max_ratio:g}x cap; "
            f"weight {lo:.2f}-{hi:.2f} (spread {hi / max(lo, 1e-9):.0f}x)")
    return weights


def make_weight_fn(weights: Dict[str, float], device: str = "cpu"):
    """``batch -> (batch,)`` tensor, looking each patient up by id.

    Keyed on ``patient_id`` rather than derived from the batch's ``visits``
    tensor on purpose: the same lookup then serves every regime, including the
    pooled ``centralized`` arm whose batches mix hospitals and where a single
    per-site weight vector could not be applied.
    """
    import torch

    def fn(batch):
        pids = batch.get("patient_id")
        if pids is None:
            return None
        return torch.tensor([weights.get(str(p), 1.0) for p in pids],
                            dtype=torch.float, device=device)
    return fn
