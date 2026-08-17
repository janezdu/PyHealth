"""Build and read the frozen cohort cache: the one file that defines the data.

Everything downstream -- all four training regimes, Test 1, Test 2 -- reads its
patients from a cache directory built by this script. Run it once; nothing else
ever touches eICU.

    export EICU_ROOT=/path/to/eicu-crd/2.0
    python cohort.py --hospitals 420,199,345,79,259,253,438,201 --out $CACHE
    python cohort.py --bands 0-199,200-499,500-1999,2000- --per-band 2 --out $CACHE

What it writes
--------------
::

    <out>/vocab.json            the fitted NestedSequenceProcessor state
    <out>/<hospital>.<fold>.parquet    8 hospitals x 3 folds = 24 files
    <out>/manifest.json         hospitals, rare codes, fold sizes, provenance

The split IS the file layout. ``420.train.parquet`` is hospital 420's train
fold -- there are no patient-id arrays to resolve against a dataset and no way
for the recorded split and the loaded data to disagree, because they are the
same object. Pooled folds are not written: ``load_fold(dir, "val")``
concatenates the 8 val files at read time, so there is no second copy to drift.

Each hospital is split 70/10/20 by iterative multilabel stratification over its
own rare codes. Train and test are guaranteed at least one patient of every rare
code; **validation is not**, because 22% of this cohort's rare (hospital, code)
pairs have only two patients and a three-fold guarantee is arithmetically
impossible for them. Val is thin on rare codes at the small hospitals by
construction -- fine for loss curves, not for rare-code claims.

The vocabulary is fitted over all ~200 eICU hospitals and then **pinned**, not
refitted on the cohort. Refitting would shrink it, changing the model's output
dimension and the denominator of every "all codes" metric.

Codes are stored as strings rather than vocabulary indices: strings survive a
vocabulary change, so a stale cache can be detected instead of silently decoded
against the wrong indices.
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

# --- where the data lives --------------------------------------------------
# From the environment, never hardcoded: eICU is credentialed data whose
# location differs per machine, and a personal path baked into a committed file
# is both unusable for anyone else and a data-safety problem (see
# .llms/rules/03-data-safety.md). Set these once, e.g. in ~/.bashrc:
#
#     export EICU_ROOT=/path/to/eicu-crd/2.0
#     export FEDCOHORT_CACHE=/fast/scratch/fedcohort     # optional
#
# The cache root falls back to a repo-relative gitignored directory, so a fresh
# clone works with no configuration at all -- just slower storage.
EICU_ROOT = os.environ.get("EICU_ROOT", "")
CACHE_ROOT = os.environ.get("FEDCOHORT_CACHE",
                            os.path.join("_outputs", "cache", "fedcohort"))

# --- what the cohort IS (not scale knobs -- changing these changes the data) --
MIN_VISITS = 1                 # eICU patients are often single-stay; keep them
TASK_NAME = "ehr_generation_eicu"

FOLDS = ("train", "val", "test")
FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}
GUARANTEED_FOLDS = ("train", "test")

# A code is RARE if <= 5% of that hospital's patients carry it. Rarity is
# per-hospital: a code rare at one hospital counts as rare even if it is common
# at another, which is the point of a federated rare-code study.
RARE_PREVALENCE_MAX = 0.05
RARE_MIN_PATIENTS = 2          # a 1-patient code cannot be split at all

# A stricter, cohort-wide subset: codes rare across ALL 8 hospitals pooled.
# Test 2 reports on this separately, because a code that is rare at one site but
# common elsewhere is a much easier target than one that is rare everywhere.
GLOBAL_RARE_PREVALENCE_MAX = 0.01

DEFAULT_BANDS = "0-199,200-499,500-1999,2000-"
DEFAULT_PER_BAND = 2
# Which cohort every job uses unless told otherwise. strat8_randsplit is the
# plain 70/10/20 shuffle: same size-banded 8 hospitals, but the patient split
# makes no rare-code coverage promise. See scripts/run_cohort.sh.
DEFAULT_COHORT = os.environ.get("FEDCOHORT_NAME", "strat8_random")
DEFAULT_CACHE_DIR = os.path.join(CACHE_ROOT, DEFAULT_COHORT)

MANIFEST_FILE = "manifest.json"
VOCAB_FILE = "vocab.json"


# --------------------------------------------------------------------------- #
# Reading -- what train.py / test1 / test2 import                             #
# --------------------------------------------------------------------------- #
def load_manifest(cache_dir: str) -> dict:
    """Read a cache's manifest, or explain that the cache was never built."""
    path = os.path.join(cache_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cohort cache at {cache_dir} (missing {MANIFEST_FILE}). "
            f"Build one with: python cohort.py --out {cache_dir}"
        )
    with open(path) as fh:
        return json.load(fh)


def hospitals(cache_dir: str) -> List[str]:
    """Cohort hospital ids, in the order they were selected."""
    return list(load_manifest(cache_dir)["hospitals"])


def rare_codes(cache_dir: str) -> Dict[str, List[str]]:
    """``{hospital_id: [rare code, ...]}`` -- rare *at that hospital*."""
    per = load_manifest(cache_dir)["per_hospital"]
    return {hid: list(h["rare_codes"]) for hid, h in per.items()}


def load_processor(cache_dir: str):
    """Rebuild the fitted ``NestedSequenceProcessor`` exactly as it was.

    ``_max_inner_len`` and ``_padding`` are as load-bearing as the vocabulary:
    they set the width of every emitted tensor, so restoring only ``code_vocab``
    would give correctly-labelled samples of the wrong shape.
    """
    from pyhealth.processors.nested_sequence_processor import (
        NestedSequenceProcessor,
    )

    with open(os.path.join(cache_dir, VOCAB_FILE)) as fh:
        state = json.load(fh)
    proc = NestedSequenceProcessor(padding=state["padding"])
    proc.code_vocab = state["code_vocab"]
    proc._next_index = max(state["code_vocab"].values()) + 1
    proc._max_inner_len = state["max_inner_len"]
    if proc.vocab_size() != state["vocab_size"]:
        raise ValueError(
            f"restored vocab size {proc.vocab_size()} != cached "
            f"{state['vocab_size']}; this PyHealth build and the cache disagree "
            "about the processor. Rebuild the cache."
        )
    return proc


def fold_path(cache_dir: str, hid: str, fold: str) -> str:
    """Path of one (hospital, fold) Parquet file."""
    return os.path.join(cache_dir, f"{hid}.{fold}.parquet")


def read_trajectories(cache_dir: str, fold: str,
                      only: Sequence[str] = None) -> Dict[str, Dict[str, list]]:
    """``{hospital_id: {patient_id: [[code, ...], ...]}}`` for one fold.

    The plain-Python view of the cache: code strings, no tensors, no PyHealth.
    Test 1 and Test 2 work from this.
    """
    import polars as pl

    out: Dict[str, Dict[str, list]] = {}
    for hid in (only or hospitals(cache_dir)):
        path = fold_path(cache_dir, str(hid), fold)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing. The cache at {cache_dir} does not hold fold "
                f"{fold!r} for hospital {hid}; rebuild it."
            )
        # Sorted before grouping so reconstruction never depends on row order.
        frame = pl.read_parquet(path).sort(["patient_id", "visit_idx", "pos"])
        patients: Dict[str, list] = {}
        for row in frame.iter_rows(named=True):
            visits = patients.setdefault(row["patient_id"], [])
            while len(visits) <= row["visit_idx"]:
                visits.append([])
            visits[row["visit_idx"]].append(row["code"])
        out[str(hid)] = patients
    return out


def _to_sample_dataset(patients_by_hospital: Dict[str, Dict[str, list]],
                       processor, name: str):
    """Wrap trajectories in a fitted ``SampleDataset`` without refitting."""
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {"patient_id": pid, "hospital_id": hid, "visits": visits}
        for hid, patients in patients_by_hospital.items()
        for pid, visits in patients.items()
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"visits": "nested_sequence"},
        output_schema={},
        input_processors={"visits": processor},   # pinned, never refitted
        dataset_name=name,
        task_name=TASK_NAME,
        in_memory=True,
    )


def load_fold(cache_dir: str, fold: str, only: Sequence[str] = None,
              processor=None):
    """One pooled ``SampleDataset`` over every hospital's ``fold``.

    This is the "+2" pooled view -- concatenated from the per-hospital files at
    read time rather than stored as its own copy.
    """
    processor = processor or load_processor(cache_dir)
    return _to_sample_dataset(read_trajectories(cache_dir, fold, only),
                              processor, f"cohort_{fold}")


def load_clients(cache_dir: str, folds: Sequence[str] = FOLDS,
                 only: Sequence[str] = None) -> Dict[str, Dict[str, object]]:
    """``{hospital_id: {fold: SampleDataset}}`` -- the federated clients.

    Every returned dataset shares one pinned processor, which is what makes the
    per-client model weights compatible for FedAvg averaging.
    """
    processor = load_processor(cache_dir)
    ids = [str(h) for h in (only or hospitals(cache_dir))]
    per_fold = {f: read_trajectories(cache_dir, f, ids) for f in folds}
    return {
        hid: {f: _to_sample_dataset({hid: per_fold[f][hid]}, processor,
                                    f"cohort_{hid}_{f}")
              for f in folds}
        for hid in ids
    }


def load_synthetic(save_dir: str) -> Dict[str, List[dict]]:
    """Read a finished run's per-hospital synthetic patients.

    Raises:
        FileNotFoundError: If the run has no ``synthetic.json`` -- a job that
            died before its final save step has nothing to score.
    """
    path = os.path.join(save_dir, "synthetic.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written at the very end of train.py; a "
            "run that timed out first has no synthetic data to score."
        )
    with open(path) as fh:
        blob = json.load(fh)
    if "per_hospital" not in blob:
        raise ValueError(f"{path} has no 'per_hospital' block; re-run the regime.")
    return blob["per_hospital"]


def load_shared_generator(save_dir: str) -> bool:
    """Did ONE generator produce every hospital's synthetic set?

    Reads the ``shared_generator`` flag written at generation time. Older files
    predate the flag, so this falls back to comparing the *content* of each
    hospital's set -- never the patient ids. Every generator numbers its output
    ``synthetic_0..N`` independently, so eight genuinely different fine-tuned
    models emit eight identical id lists over completely different patients;
    comparing ids reports "shared" and collapses eight per-hospital classifiers
    into one.

    Args:
        save_dir: The run's ``_outputs/<run_name>_save/`` folder.

    Returns:
        True when a single generator served every hospital.
    """
    with open(os.path.join(save_dir, "synthetic.json")) as fh:
        blob = json.load(fh)
    if "shared_generator" in blob:
        return bool(blob["shared_generator"])

    per_hospital = blob["per_hospital"]
    if len(per_hospital) < 2:
        return False
    signatures = {
        json.dumps([p["visits"] for p in patients], sort_keys=True)
        for patients in per_hospital.values()
    }
    return len(signatures) == 1


def manifest_sha256(manifest: dict) -> str:
    """Stable identity of the cohort cache, used in the resume fingerprint.

    Client sizes alone would not do: two different splits of the same cohort
    have identical sizes, so resuming across them would silently train on one
    partition and score on another. Lives here rather than in train.py so the
    training, generation and evaluation modules all stamp the same identity.
    """
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()


def load_pooled_synthetic(save_dir: str, mix: str = "proportional") -> List[dict]:
    """Read a finished run's pooled synthetic patients under one mixing rule.

    The two views answer different questions and are not interchangeable:

    - ``"proportional"`` -- hospital ``h`` contributes in proportion to its real
      train size, so the mix matches the real pooled cohort. This is the only
      view that may be scored against real pooled data: pooled prevalence is a
      hospital-weighted average, so a differently mixed set scores badly even
      when the generator is perfect.
    - ``"uniform"`` -- every hospital contributes equally (the concatenation of
      the per-hospital sets). With this cohort that lifts hospital 438 from 1.4%
      to 12.5% of the set, ~9x its real weight. Use it to describe how the
      federation represents its smallest members; do NOT use it as a fidelity
      metric against the real cohort.

    Note that for the single-model regimes (``centralized``, ``fedavg``) the
    distinction is vacuous -- one generator emits one distribution, and every
    hospital key holds the same list. For ``fedavg_ft`` the two views come from
    *different models*: the stored proportional set is the shared global model's
    output, while the uniform view concatenates the eight fine-tuned models.
    Do not read a difference between them as a mixing effect.

    Args:
        save_dir: The run's ``_outputs/<run_name>_save/`` folder.
        mix: ``"proportional"`` or ``"uniform"``.

    Returns:
        A flat list of ``{"patient_id", "visits"}`` dicts.

    Raises:
        ValueError: If ``mix`` is not one of the two supported rules, or the
            file predates this field and holds no pooled set at all.
    """
    if mix not in ("proportional", "uniform"):
        raise ValueError(
            f"mix must be 'proportional' or 'uniform', got {mix!r}")

    if mix == "uniform":
        per_hospital = load_synthetic(save_dir)
        # Single-model regimes file one global set under every hospital key.
        # Concatenating those would return the same patients eight times and
        # report a 40k "uniform" set that is really 5k of distinct data.
        ids = {h: tuple(str(p["patient_id"]) for p in v)
               for h, v in per_hospital.items()}
        if len(set(ids.values())) == 1 and len(per_hospital) > 1:
            return list(next(iter(per_hospital.values())))
        return [p for hid in sorted(per_hospital) for p in per_hospital[hid]]

    with open(os.path.join(save_dir, "synthetic.json")) as fh:
        blob = json.load(fh)
    # "pooled" is the pre-rename alias, kept so the runs finished before this
    # split stay readable.
    for key in ("pooled_proportional", "pooled"):
        if key in blob:
            return blob[key]
    raise ValueError(
        f"{save_dir}/synthetic.json has neither 'pooled_proportional' nor the "
        "legacy 'pooled' block; re-run the regime."
    )


def parse_run_specs(specs: Sequence[str]) -> Dict[str, str]:
    """Turn ``["fedavg=_outputs/x_save", ...]`` into ``{name: save_dir}``."""
    runs: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--run {spec!r} must look like NAME=SAVE_DIR")
        name, save_dir = spec.split("=", 1)
        runs[name.strip()] = save_dir.strip()
    return runs


# --------------------------------------------------------------------------- #
# Selecting the hospitals                                                      #
# --------------------------------------------------------------------------- #
def parse_bands(spec: str) -> List[Tuple[int, int]]:
    """Parse ``'0-199,200-499,2000-'`` into inclusive ``(lo, hi)`` pairs."""
    bands: List[Tuple[int, int]] = []
    for chunk in (c.strip() for c in spec.split(",") if c.strip()):
        if "-" not in chunk:
            raise ValueError(f"band {chunk!r} must look like 'lo-hi' or 'lo-'")
        lo_s, hi_s = chunk.split("-", 1)
        lo = int(lo_s) if lo_s.strip() else 0
        hi = int(hi_s) if hi_s.strip() else 10 ** 9
        if lo > hi:
            raise ValueError(f"band {chunk!r} has lo > hi")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("--bands parsed to zero bands")
    return bands


def draw_bands(sizes: Dict[str, int], bands: List[Tuple[int, int]],
               per_band: int, seed: int) -> List[Tuple[str, int]]:
    """Draw ``per_band`` hospitals uniformly from each size band.

    Size-banded rather than top-K by design: the top-K hospitals are all large,
    so a model that works there says nothing about whether federation helps the
    small sites, which is the whole question.

    Returns:
        ``[(hospital_id, band_index), ...]``, largest band first.

    Raises:
        ValueError: If any band has fewer than ``per_band`` hospitals in it.
    """
    pools = {b: sorted((hid for hid, n in sizes.items() if lo <= n <= hi),
                       key=lambda h: (-sizes[h], h))
             for b, (lo, hi) in enumerate(bands)}
    short = [(bands[b], len(pools[b])) for b in pools if len(pools[b]) < per_band]
    if short:
        detail = "; ".join(f"band {lo}-{hi}: {n} eligible" for (lo, hi), n in short)
        raise ValueError(f"cannot draw {per_band} per band -- {detail}. Widen "
                         f"the band or lower --per-band.")

    rng = np.random.default_rng(seed)
    chosen: List[Tuple[str, int]] = []
    for b in sorted(pools, key=lambda i: bands[i][0], reverse=True):
        picks = rng.choice(len(pools[b]), size=per_band, replace=False)
        chosen.extend((pools[b][int(p)], b) for p in sorted(picks))
    return chosen


def cross_hospital_patients(eicu_root: str) -> Set[str]:
    """Patients with unit stays at more than one hospital.

    They are dropped: a patient present at two clients breaks the premise that
    hospitals hold disjoint data, and would leak across the federation.
    """
    path = os.path.join(eicu_root, "patient.csv")
    try:
        seen: Dict[str, Set[str]] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                seen.setdefault(row["uniquepid"], set()).add(row["hospitalid"])
        return {pid for pid, hosp in seen.items() if len(hosp) > 1}
    except (OSError, KeyError) as exc:
        print(f"WARNING: cannot read {path} ({exc}); cross-hospital patients "
              "will NOT be filtered", flush=True)
        return set()


# --------------------------------------------------------------------------- #
# Rare codes + the stratified split                                            #
# --------------------------------------------------------------------------- #
def compute_rare_codes(patient_codes: Dict[str, Set[str]],
                       prevalence_max: float = RARE_PREVALENCE_MAX,
                       min_patients: int = RARE_MIN_PATIENTS) -> Dict[str, dict]:
    """Rare codes of ONE hospital: carried by <= ``prevalence_max`` of it."""
    n_h = len(patient_codes)
    counts: Dict[str, int] = {}
    for codes in patient_codes.values():
        for code in codes:
            counts[code] = counts.get(code, 0) + 1

    rare = {code: {"n_patients": counts[code],
                   "prevalence": round(counts[code] / n_h, 8)}
            for code in sorted(counts)
            if counts[code] >= min_patients
            and counts[code] / n_h <= prevalence_max}
    if not rare:
        raise ValueError(
            f"empty rare set: {n_h} patients, prevalence_max={prevalence_max}, "
            f"min_patients={min_patients}. At this hospital size a code needs "
            f"<= {prevalence_max * n_h:.1f} patients to be rare but >= "
            f"{min_patients} to qualify -- the thresholds are incompatible."
        )
    return rare


def _rare_quotas(n: int, fracs: Dict[str, float],
                 guaranteed: Sequence[str]) -> Dict[str, int]:
    """How many of a rare code's ``n`` patients each fold should get.

    Every guaranteed fold gets at least one, taken from the largest fold with
    slack. A 2-patient code therefore splits 1 train / 1 test / 0 val --
    measurable where the claim is made, absent where nothing depends on it.
    """
    if n < len(guaranteed):
        raise ValueError(
            f"a rare code carried by {n} patient(s) cannot reach all of "
            f"{list(guaranteed)}. Keep --rare-min-patients >= "
            f"{len(guaranteed)} so codes this thin are not scored at all."
        )
    quotas = {f: int(np.floor(fracs[f] * n + 0.5)) for f in fracs}
    for f in guaranteed:
        quotas[f] = max(1, quotas[f])
    while sum(quotas.values()) > n:
        donor = max((f for f in quotas
                     if quotas[f] > (1 if f in guaranteed else 0)),
                    key=lambda f: (quotas[f] - (1 if f in guaranteed else 0), f))
        quotas[donor] -= 1
    while sum(quotas.values()) < n:
        quotas[max(fracs, key=lambda f: (fracs[f], f))] += 1
    return quotas


def _pick_fold(desired, slots, code, folds, rng) -> str:
    """The fold that most needs one more patient carrying ``code``."""
    best = max(desired[f][code] for f in folds)
    cands = [f for f in folds if desired[f][code] == best]
    if len(cands) > 1:
        best_slots = max(slots[f] for f in cands)
        cands = [f for f in cands if slots[f] == best_slots]
    return cands[0] if len(cands) == 1 else str(rng.choice(cands))


def _repair_guarantee(assignment: Dict[str, str], labels: Dict[str, List[str]],
                      rare: Sequence[str], folds: Sequence[str],
                      guaranteed: Sequence[str]) -> int:
    """Move patients until every rare code reaches every guaranteed fold.

    The greedy pass alone does NOT guarantee this, which is easy to miss because
    it usually achieves it. A code is starved when every patient carrying it
    also carries a code processed earlier: those patients are all placed by the
    earlier code's deficits, the starved code's unassigned count reaches zero,
    and it is never considered on its own terms. Real ICD-9 data is full of such
    co-occurrence, so this is not a corner case.

    The repair moves one patient at a time, and only when the move is safe --
    the donor fold must keep at least one patient of every code that patient
    carries, or fixing one code would break another.

    Returns:
        The number of patients moved (0 when the greedy pass already sufficed).

    Raises:
        ValueError: If a code cannot reach a guaranteed fold by any safe move.
    """
    holders: Dict[Tuple[str, str], Set[str]] = {}
    for pid, fold in assignment.items():
        for code in labels[pid]:
            holders.setdefault((fold, code), set()).add(pid)

    def held(fold, code):
        return holders.get((fold, code), set())

    moves = 0
    for code in sorted(rare):
        for target in guaranteed:
            if held(target, code):
                continue
            # Donor folds that can spare one: a guaranteed fold must keep >= 1.
            donors = sorted(
                (f for f in folds if f != target
                 and len(held(f, code)) >= (2 if f in guaranteed else 1)),
                key=lambda f: (-len(held(f, code)), f),
            )
            moved = False
            for donor in donors:
                for pid in sorted(held(donor, code)):
                    # Safe only if the donor keeps every OTHER code of this
                    # patient covered too.
                    if donor in guaranteed and any(
                            len(held(donor, d)) < 2 for d in labels[pid]):
                        continue
                    for d in labels[pid]:
                        holders[(donor, d)].discard(pid)
                        holders.setdefault((target, d), set()).add(pid)
                    assignment[pid] = target
                    moves += 1
                    moved = True
                    break
                if moved:
                    break
            if not moved:
                raise ValueError(
                    f"rare code {code!r} cannot reach fold {target!r}: every "
                    f"patient carrying it is the sole carrier of some other "
                    f"code in its fold. Raise --rare-min-patients so codes this "
                    f"thin are not scored at all."
                )
    return moves


def stratified_split(patient_codes: Dict[str, Set[str]],
                     rare: Iterable[str],
                     fracs: Dict[str, float] = None,
                     guaranteed: Sequence[str] = GUARANTEED_FOLDS,
                     seed: int = 0) -> Dict[str, List[str]]:
    """Split one hospital's patients, stratified on every rare code.

    Iterative multilabel stratification (Sechidis et al. 2011): the rare code
    with the fewest still-unassigned patients is placed first, and each of its
    patients goes to whichever fold has the largest remaining deficit for that
    code. Committing the scarcest codes while every fold still has slack is what
    makes the per-fold guarantee achievable at all.

    That greedy pass sets the proportions; :func:`_repair_guarantee` then
    enforces the >=1-per-guaranteed-fold rule, which the greedy pass gets right
    most of the time but does not actually guarantee.

    Returns:
        ``({fold: sorted patient ids}, n_repaired)``. Deterministic for a given
        seed and invariant to the input ordering of ``patient_codes``.
        ``n_repaired`` is how many patients the repair pass had to move; it is
        returned rather than swallowed, so a split that needed heavy
        rearrangement is visible instead of silent.
    """
    fracs = dict(fracs or FRACS)
    folds = list(fracs)
    rare = sorted(rare)
    rare_set = set(rare)
    pids = sorted(patient_codes)          # sorted in => order-invariant out
    labels = {p: sorted(rare_set & patient_codes[p]) for p in pids}

    patients_with = {j: set() for j in rare}
    for p in pids:
        for j in labels[p]:
            patients_with[j].add(p)
    n_j = {j: len(patients_with[j]) for j in rare}

    n_total = len(pids)
    slots = {f: int(round(fracs[f] * n_total)) for f in folds}
    slots[folds[0]] += n_total - sum(slots.values())      # absorb rounding

    desired = {f: {} for f in folds}
    for j in rare:
        quotas = _rare_quotas(n_j[j], fracs, guaranteed)
        for f in folds:
            desired[f][j] = quotas[f]

    remaining = dict(n_j)
    unassigned = set(pids)
    assignment: Dict[str, str] = {}
    rng = np.random.default_rng(seed)

    while True:
        active = [j for j in rare if remaining[j] > 0]
        if not active:
            break
        code = min(active, key=lambda k: (remaining[k], k))
        for pid in sorted(patients_with[code] & unassigned):
            fold = _pick_fold(desired, slots, code, folds, rng)
            assignment[pid] = fold
            unassigned.discard(pid)
            slots[fold] -= 1
            for k in labels[pid]:
                desired[fold][k] -= 1
                remaining[k] -= 1

    # Patients carrying no rare code fill whichever fold is furthest from quota.
    leftovers = sorted(unassigned)
    for pos in rng.permutation(len(leftovers)):
        pid = leftovers[int(pos)]
        fold = max(folds, key=lambda f: (slots[f], f))
        assignment[pid] = fold
        slots[fold] -= 1

    # The greedy pass gets the proportions right but does not actually enforce
    # the >=1-per-guaranteed-fold rule; this does. It moves real patients
    # between folds, so the count is reported rather than applied silently.
    moved = _repair_guarantee(assignment, labels, rare, folds, guaranteed)

    splits = {f: sorted(p for p in pids if assignment[p] == f) for f in folds}
    return splits, moved


def random_split(patient_codes: Dict[str, Set[str]],
                 fracs: Dict[str, float] = None,
                 seed: int = 0) -> Tuple[Dict[str, List[str]], int]:
    """Plain seeded 70/10/20, blind to rare codes. The control condition.

    This is what most work does, and it is the honest baseline for asking
    whether the stratified split is worth its complexity. It makes no promise
    that a rare code appears in any particular fold -- a code carried by two
    patients has a ~0.64 chance of missing test entirely -- so codes drop out
    of the scoring pool by luck rather than by design.

    Returns:
        ``({fold: sorted patient ids}, 0)``. The trailing 0 mirrors
        :func:`stratified_split`'s repair count so callers need no special case.
    """
    fracs = dict(fracs or FRACS)
    folds = list(fracs)
    pids = sorted(patient_codes)              # sorted in => reproducible out
    rng = np.random.default_rng(seed)
    order = [pids[int(i)] for i in rng.permutation(len(pids))]

    n_total = len(pids)
    slots = {f: int(round(fracs[f] * n_total)) for f in folds}
    slots[folds[0]] += n_total - sum(slots.values())      # absorb rounding

    out, start = {}, 0
    for f in folds:
        out[f] = sorted(order[start:start + slots[f]])
        start += slots[f]
    return out, 0


def verify_split(patient_codes: Dict[str, Set[str]], rare: Iterable[str],
                 splits: Dict[str, List[str]], fracs: Dict[str, float] = None,
                 guaranteed: Sequence[str] = GUARANTEED_FOLDS,
                 tol: float = 0.10) -> None:
    """Assert the split invariants. This runs on every build -- it is the guard.

    Raises:
        ValueError: If folds overlap, miss a patient, drop a rare code from a
            guaranteed fold, or drift more than ``tol`` from a target fraction.
    """
    fracs = dict(fracs or FRACS)
    names = list(splits)
    sets = {f: set(splits[f]) for f in names}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"{a}/{b} overlap: {sorted(sets[a] & sets[b])[:5]}")
    covered = set().union(*sets.values()) if sets else set()
    if covered != set(patient_codes):
        missing = set(patient_codes) - covered
        raise ValueError(f"{len(missing)} patients unassigned, e.g. "
                         f"{sorted(missing)[:5]}")

    for code in sorted(rare):
        for f in guaranteed:
            if not any(code in patient_codes[p] for p in splits[f]):
                raise ValueError(
                    f"rare code {code!r} is absent from {f}; the >=1-per-fold "
                    f"guarantee for {list(guaranteed)} was violated")

    n_total = max(1, len(patient_codes))
    for f in names:
        realized = len(splits[f]) / n_total
        if abs(realized - fracs[f]) > tol:
            raise ValueError(
                f"realized {f} fraction {realized:.4f} drifts more than {tol} "
                f"from the {fracs[f]} target -- inspect before relaxing")


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #
def write_fold(patients: Dict[str, list], path: str) -> None:
    """Write one (hospital, fold) as a long-format Parquet table.

    One row per (patient, visit, position)::

        patient_id | visit_idx | pos | code
        002-10001  |         0 |   0 | 428.0

    ``pos`` is not decoration. The processor pads each visit in the order codes
    appear, so a visit's code ORDER is part of the emitted tensor; without an
    explicit position column a groupby could reorder them and silently produce
    different tensors from the same data.

    Long format rather than a nested ``list<list<string>>`` column because it is
    the shape Parquet is actually good at -- the code column dictionary-encodes
    to almost nothing -- and it sidesteps nested-type quirks between Arrow
    versions.
    """
    import polars as pl

    rows = {"patient_id": [], "visit_idx": [], "pos": [], "code": []}
    for pid, visits in patients.items():
        for v, visit in enumerate(visits):
            for i, code in enumerate(visit):
                rows["patient_id"].append(str(pid))
                rows["visit_idx"].append(v)
                rows["pos"].append(i)
                rows["code"].append(str(code))
    pl.DataFrame(rows, schema={"patient_id": pl.Utf8, "visit_idx": pl.Int32,
                               "pos": pl.Int32, "code": pl.Utf8}
                 ).write_parquet(path, compression="zstd")


def write_processor(processor, path: str) -> None:
    """Persist the fitted processor's full state (vocab + tensor geometry)."""
    with open(path, "w") as fh:
        json.dump({"code_vocab": processor.code_vocab,
                   "max_inner_len": processor._max_inner_len,
                   "padding": processor._padding,
                   "vocab_size": processor.vocab_size()}, fh)


def _sha256_list(items: Sequence[str]) -> str:
    """Content hash of an id list -- lets a rebuild be verified without storing
    the ids anywhere."""
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode())
        h.update(b"\n")
    return h.hexdigest()


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _describe(values: Sequence[float]) -> dict:
    """mean/p50/p90/max of a list of counts."""
    if not values:
        return {"mean": 0.0, "p50": 0, "p90": 0, "max": 0, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4),
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "max": int(arr.max()), "n": int(arr.size)}


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
def build(args) -> None:
    """Walk eICU once, select, split, and write the cache."""
    if not args.eicu_root:
        raise SystemExit(
            "eICU location unknown: set EICU_ROOT in your environment, or pass "
            "--eicu-root /path/to/eicu-crd/2.0.\n"
            "  export EICU_ROOT=/path/to/eicu-crd/2.0"
        )
    if not os.path.isdir(args.eicu_root):
        raise SystemExit(
            f"--eicu-root {args.eicu_root!r} is not a directory. It should be "
            "the folder holding patient.csv and diagnosis.csv."
        )

    from pyhealth.datasets import eICUDataset
    from pyhealth.tasks import EHRGenerationEICU

    fracs = {"train": 1.0 - args.val_frac - args.test_frac,
             "val": args.val_frac, "test": args.test_frac}
    if min(fracs.values()) <= 0:
        raise ValueError(f"fold fractions must all be positive, got {fracs}")

    if args.dev:
        print("!! --dev loads a SUBSET of eICU: hospital sizes and rare codes "
              "are NOT representative. Never build a real cache with it.",
              flush=True)
    print(f"Loading eICU from {args.eicu_root} (dev={args.dev})...", flush=True)
    base = eICUDataset(root=args.eicu_root, tables=["diagnosis"], dev=args.dev)
    samples = base.set_task(EHRGenerationEICU(min_visits=MIN_VISITS))
    processor = samples.input_processors["visits"]
    index_to_code = {v: k for k, v in processor.code_vocab.items()}
    print(f"Total samples: {len(samples)}   code vocab: "
          f"{processor.vocab_size()}", flush=True)

    dropped = cross_hospital_patients(args.eicu_root) if args.drop_cross else set()
    if dropped:
        print(f"Excluding {len(dropped)} patients seen at >1 hospital")

    # ONE pass over every sample. Decoded code strings are shared references
    # into the vocabulary, so holding all ~119k patients costs tens of MB, not
    # gigabytes -- which is what lets band selection happen AFTER the walk, on
    # exact sizes, with no CSV-based size estimation to calibrate.
    print("Walking samples (the slow part)...", flush=True)
    traj: Dict[str, Dict[str, list]] = {}
    n_dropped = 0
    for i in range(len(samples)):
        sample = samples[i]
        hid = str(sample.get("hospital_id", "NA"))
        pid = str(sample["patient_id"])
        if pid in dropped:
            n_dropped += 1
            continue
        visits = []
        for visit in sample["visits"].tolist():
            codes = [index_to_code.get(int(c)) for c in visit]
            codes = [c for c in codes if c not in (None, "<pad>", "<unk>")]
            if codes:
                visits.append(codes)
        if not visits:
            continue
        at_hospital = traj.setdefault(hid, {})
        if pid in at_hospital:
            raise ValueError(f"duplicate patient_id {pid!r} at hospital {hid}; "
                             "this pipeline assumes one sample per patient")
        at_hospital[pid] = visits
        if (i + 1) % 20000 == 0:
            print(f"  ...{i + 1}/{len(samples)}", flush=True)

    sizes = {hid: len(p) for hid, p in traj.items()}
    print(f"{len(sizes)} hospitals seen; {n_dropped} cross-hospital patients "
          f"dropped")

    # --- select the cohort ---
    bands = parse_bands(args.bands)
    if args.hospitals:
        cohort = [h.strip() for h in args.hospitals.split(",") if h.strip()]
        missing = [h for h in cohort if h not in sizes]
        if missing:
            raise ValueError(f"hospitals absent from this eICU build: {missing}")
        band_of = {h: next((b for b, (lo, hi) in enumerate(bands)
                            if lo <= sizes[h] <= hi), -1) for h in cohort}
        chosen = [(h, band_of[h]) for h in cohort]
        how = "explicit"
    else:
        chosen = draw_bands(sizes, bands, args.per_band, args.seed)
        how = "bands"
    cohort = [h for h, _ in chosen]
    print(f"\nCohort ({how}): " + ", ".join(
        f"{h} (n={sizes[h]}, band {b})" for h, b in chosen))
    print(f"Pin this draw with:  --hospitals {','.join(cohort)}")

    # --- rare codes + split, per hospital ---
    os.makedirs(args.out, exist_ok=True)
    per_hospital: Dict[str, dict] = {}
    pooled_rare: Set[str] = set()
    splits: Dict[str, Dict[str, List[str]]] = {}
    print()
    for hid, band in chosen:
        patients = traj[hid]
        codes_of = {pid: {c for visit in v for c in visit}
                    for pid, v in patients.items()}
        rare = compute_rare_codes(codes_of, args.rare_prevalence_max,
                                  args.rare_min_patients)
        # The random split makes no coverage promise, so verifying one would
        # always fail. Overlap, coverage and fraction checks still apply.
        guaranteed = GUARANTEED_FOLDS if args.split == "stratified" else ()
        if args.split == "stratified":
            split, n_repaired = stratified_split(
                codes_of, rare, fracs=fracs, guaranteed=guaranteed,
                seed=args.seed)
        else:
            split, n_repaired = random_split(codes_of, fracs=fracs,
                                             seed=args.seed)
        verify_split(codes_of, rare, split, fracs=fracs, guaranteed=guaranteed)
        splits[hid] = split
        pooled_rare.update(rare)

        for fold in FOLDS:
            write_fold({p: patients[p] for p in split[fold]},
                       fold_path(args.out, hid, fold))
        per_hospital[hid] = {
            "size_band": band,
            "n_total": len(patients),
            **{f"n_{f}": len(split[f]) for f in FOLDS},
            "n_rare_codes": len(rare),
            "n_repaired": n_repaired,
            "min_rare_prevalence": min(r["prevalence"] for r in rare.values()),
            "rare_codes": sorted(rare),
            "fold_sha256": {f: _sha256_list(split[f]) for f in FOLDS},
        }
        print(f"  [{hid}] {len(patients)} patients -> "
              + " / ".join(f"{len(split[f])} {f}" for f in FOLDS)
              + f"   {len(rare)} rare codes"
              + (f"   ({n_repaired} moved to hold the guarantee)"
                 if n_repaired else ""))

    # --- manifest ---
    write_processor(processor, os.path.join(args.out, VOCAB_FILE))
    pooled = sorted(pooled_rare)
    # Per-fold support of every pooled rare code: Test 2 needs it to band its
    # results by how much real signal a code actually had.
    support = {f: {c: 0 for c in pooled} for f in FOLDS}
    for hid in cohort:
        for fold in FOLDS:
            for pid in splits[hid][fold]:
                for code in {c for visit in traj[hid][pid] for c in visit}:
                    if code in support[fold]:
                        support[fold][code] += 1
    n_cohort = sum(per_hospital[h]["n_total"] for h in cohort)
    global_rare = [c for c in pooled
                   if sum(support[f][c] for f in FOLDS) / max(1, n_cohort)
                   <= args.global_rare_prevalence_max]
    stays = [len(traj[h][p]) for h in cohort for p in traj[h]]
    cohort_codes = {c for h in cohort for p in traj[h]
                    for visit in traj[h][p] for c in visit}
    manifest = {
        "kind": "cohort_cache",
        "cohort_name": args.name,
        "hospitals": cohort,
        "selected_by": how,
        "bands": args.bands,
        "per_band": args.per_band,
        "seed": args.seed,
        "fracs": fracs,
        "split_method": args.split,
        "guaranteed_folds": (list(GUARANTEED_FOLDS)
                             if args.split == "stratified" else []),
        "rare_prevalence_max": args.rare_prevalence_max,
        "rare_min_patients": args.rare_min_patients,
        "rare_scope": "per-hospital: rare at any one hospital counts as rare",
        "eicu_root": args.eicu_root,
        "dev": args.dev,
        "task_name": TASK_NAME,
        "min_visits": MIN_VISITS,
        "vocab_size": processor.vocab_size(),
        "cohort_distinct_codes": len(cohort_codes),
        "dataset_total_samples": len(samples),
        "n_hospitals_total": len(sizes),
        "n_cross_hospital_patients_dropped": n_dropped,
        "unit_stays_per_patient": _describe(stays),
        "per_hospital": per_hospital,
        "pooled_rare_codes": pooled,
        "n_pooled_rare_codes": len(pooled),
        "pooled_rare_support": support,
        "global_rare_prevalence_max": args.global_rare_prevalence_max,
        "global_rare_codes": global_rare,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    with open(os.path.join(args.out, MANIFEST_FILE), "w") as fh:
        json.dump(manifest, fh, indent=2)

    totals = {f: sum(per_hospital[h][f"n_{f}"] for h in cohort) for f in FOLDS}
    n_all = sum(totals.values())
    print(f"\n{n_all} patients: " + " / ".join(
        f"{totals[f]} {f} ({totals[f] / n_all:.3f})" for f in FOLDS))
    print(f"{len(pooled)} pooled rare codes ({len(global_rare)} of them rare "
          f"cohort-wide at <= {args.global_rare_prevalence_max}); cohort uses "
          f"{len(cohort_codes)} distinct codes of the {processor.vocab_size()} "
          f"in the vocabulary")
    print(f"Wrote {len(cohort) * len(FOLDS)} Parquet files -> {args.out}")

    if args.no_verify:
        print("--no-verify: skipped the read-back check")
        return

    # The check that makes the cache trustworthy: rebuild from disk and compare
    # emitted tensors against the ones the eICU path just produced. A cache that
    # is merely plausible is worse than none -- it would silently train every
    # regime on subtly different data.
    print("\nVerifying: rebuilding from cache and comparing tensors...",
          flush=True)
    want = {}
    for i in range(len(samples)):
        s = samples[i]
        pid = str(s["patient_id"])
        if str(s.get("hospital_id")) in set(cohort):
            want[pid] = s["visits"].tolist()
    checked = 0
    for fold in FOLDS:
        cached = load_fold(args.out, fold, only=cohort, processor=processor)
        if len(cached) != totals[fold]:
            raise ValueError(f"{fold}: cache has {len(cached)} samples, "
                             f"expected {totals[fold]}")
        for i in range(len(cached)):
            s = cached[i]
            if s["visits"].tolist() != want.get(str(s["patient_id"])):
                raise ValueError(
                    f"{fold}: patient {s['patient_id']} does not reproduce the "
                    "eICU tensor. Do NOT train on this cache.")
            checked += 1
    print(f"OK: all {checked} cached samples reproduce byte-identical tensors.")
    print(f"\nUse it with:  --cohort-cache {args.out}")


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _table(header: Sequence[str], rows: Sequence[Sequence]) -> None:
    """Right-aligned numeric table with a left-aligned first column."""
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(str(header[i])), *(len(r[i]) for r in cells)) if cells
         else len(str(header[i])) for i in range(len(header))]
    print("  " + "  ".join(str(h).ljust(w[0]) if i == 0 else str(h).rjust(w[i])
                           for i, h in enumerate(header)))
    print("  " + "-" * (sum(w) + 2 * (len(w) - 1)))
    for r in cells:
        print("  " + "  ".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i])
                               for i, c in enumerate(r)))


def report(cache_dir: str) -> None:
    """Print fold coverage: which codes and patients land where.

    The code table is the one that matters for interpreting a rare-code result.
    A code present in train but absent from the scoring fold cannot be measured
    however good the generator is, and a code in the scoring fold but not train
    is being asked for blind -- both look like ordinary rows in a metric table.
    """
    manifest = load_manifest(cache_dir)
    hosp = list(manifest["hospitals"])
    rare = set(manifest["pooled_rare_codes"])
    traj = {f: read_trajectories(cache_dir, f, hosp) for f in FOLDS}

    codes_in = {f: {c for pats in traj[f].values() for visits in pats.values()
                    for visit in visits for c in visit} for f in FOLDS}
    every = set().union(*codes_in.values())

    # Total carriers per code across the whole cohort. Needed because "not in
    # the rare set" is NOT the same as "common": a code carried by one patient
    # fails the >=RARE_MIN_PATIENTS floor and so never qualifies as rare, even
    # though it is the rarest thing in the data. Without this the non-rare
    # column reads as "common codes" and badly misleads.
    carriers: Dict[str, int] = {}
    for pats in traj.values():
        for patients in pats.values():
            for visits in patients.values():
                for code in {c for visit in visits for c in visit}:
                    carriers[code] = carriers.get(code, 0) + 1

    # Which folds each code appears in -> one row per non-empty combination.
    where: Dict[Tuple[str, ...], List[str]] = {}
    for code in every:
        key = tuple(f for f in FOLDS if code in codes_in[f])
        where.setdefault(key, []).append(code)

    print(f"\ncohort '{manifest.get('cohort_name')}'   split="
          f"{manifest.get('split_method', 'stratified')}"
          f"   guaranteed={manifest.get('guaranteed_folds') or 'none'}")
    print(f"  {len(hosp)} hospitals, {len(every)} distinct codes of "
          f"{manifest['vocab_size']} in vocab, {len(rare)} pooled rare")

    print("\nCODES by fold presence")
    order = sorted(where, key=lambda k: (-len(k), [FOLDS.index(f) for f in k]))
    rows = []
    for key in order:
        found = where[key]
        n_rare = sum(1 for c in found if c in rare)
        rows.append(["+".join(key), n_rare, len(found) - n_rare, len(found)])
    rows.append(["TOTAL", len(rare & every), len(every - rare), len(every)])
    thin = {c for c in every - rare if carriers[c] < RARE_MIN_PATIENTS}
    # A code in the vocabulary that no cohort patient carries is not a gap in
    # the split -- it belongs to one of the other ~200 hospitals.
    absent = manifest["vocab_size"] - len(every)
    _table(["present in", "rare", "not rare", "all"], rows)
    print(f"  ({absent} more codes exist in the pinned vocabulary but no cohort "
          f"patient carries them)")
    print(f"  NOTE: 'not rare' is not the same as common -- {len(thin)} of those "
          f"{len(every - rare)} codes have < {RARE_MIN_PATIENTS} carriers in the "
          f"whole cohort\n        and were excluded from the rare set by the "
          f"minimum-carriers floor, not by prevalence.")

    scoreable = {f: sum(1 for c in rare if c in codes_in[f]) for f in FOLDS}
    print("\n  rare codes reachable per fold: " + "   ".join(
        f"{f} {scoreable[f]}/{len(rare)} ({scoreable[f] / len(rare):.1%})"
        for f in FOLDS))

    print("\nPATIENTS by fold")
    rows = []
    for hid in hosp:
        n = {f: len(traj[f][hid]) for f in FOLDS}
        total = sum(n.values())
        rows.append([hid, *(n[f] for f in FOLDS), total,
                     " ".join(f"{n[f] / total:.2f}" for f in FOLDS)])
    tot = {f: sum(len(traj[f][h]) for h in hosp) for f in FOLDS}
    n_all = sum(tot.values())
    rows.append(["TOTAL", *(tot[f] for f in FOLDS), n_all,
                 " ".join(f"{tot[f] / n_all:.2f}" for f in FOLDS)])
    _table(["hospital", *FOLDS, "all", "fractions"], rows)

    # Patients are assigned to exactly one fold, so every cross term must be
    # empty. If one is not, folds overlap and every metric is contaminated.
    ids = {f: {p for pats in traj[f].values() for p in pats} for f in FOLDS}
    overlaps = {f"{a}+{b}": len(ids[a] & ids[b])
                for i, a in enumerate(FOLDS) for b in FOLDS[i + 1:]}
    bad = {k: v for k, v in overlaps.items() if v}
    print(f"  patient overlap between folds: "
          + ("NONE (as it must be)" if not bad else f"!! {bad} !!"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the frozen cohort cache read by every downstream job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--out", default=DEFAULT_CACHE_DIR,
                   help="cache directory (put it on fast local storage)")
    p.add_argument("--report", action="store_true",
                   help="do not build: read the cache at --out and print fold "
                        "coverage for codes and patients, then exit")
    p.add_argument("--name", default="strat8", help="cohort name, for the record")
    p.add_argument("--eicu-root", default=EICU_ROOT,
                   help="eICU CRD root (default: $EICU_ROOT)")
    p.add_argument("--hospitals",
                   help="comma-separated hospital ids to use verbatim; omit to "
                        "draw --per-band from each of --bands")
    p.add_argument("--bands", default=DEFAULT_BANDS,
                   help="inclusive patient-count bands, e.g. "
                        f"'{DEFAULT_BANDS}' (default)")
    p.add_argument("--per-band", type=int, default=DEFAULT_PER_BAND,
                   help="hospitals to draw from each band (ignored with "
                        "--hospitals)")
    p.add_argument("--val-frac", type=float, default=FRACS["val"])
    p.add_argument("--test-frac", type=float, default=FRACS["test"])
    p.add_argument("--rare-prevalence-max", type=float,
                   default=RARE_PREVALENCE_MAX,
                   help="a code is rare at <= this prevalence IN A HOSPITAL")
    p.add_argument("--rare-min-patients", type=int, default=RARE_MIN_PATIENTS)
    p.add_argument("--global-rare-prevalence-max", type=float,
                   default=GLOBAL_RARE_PREVALENCE_MAX,
                   help="stricter subset: rare across the whole pooled cohort")
    p.add_argument("--split", choices=["stratified", "random"],
                   default="stratified",
                   help="stratified: iterative multilabel stratification over "
                        "rare codes, with train/test coverage guaranteed. "
                        "random: a plain seeded shuffle, blind to rare codes "
                        "-- the control condition")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the band draw and the split")
    p.add_argument("--dev", action="store_true",
                   help="load a small eICU subset -- for wiring checks ONLY")
    p.add_argument("--keep-cross-hospital", dest="drop_cross",
                   action="store_false",
                   help="keep patients seen at more than one hospital")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the read-back check (not recommended: it is the "
                        "only thing proving the cache reproduces the data)")
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    report(_args.out) if _args.report else build(_args)
