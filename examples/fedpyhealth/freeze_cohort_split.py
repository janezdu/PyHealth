"""Freeze a fixed 8-hospital cohort with rare-code-stratified 80/20 splits.

Run this ONCE. It produces the manifest that every baseline (centralized /
local / fedavg / fedavg_ft) loads via ``ehr_eicu.py --cohort-file``, so all four
regimes train and evaluate on byte-identical data. Re-deriving the split at run
time would make it a function of the numpy version, dict ordering and any future
edit to the stratifier -- which silently breaks cross-regime comparability and
invalidates FedAvg resume checkpoints.

What "rare" means here
----------------------
A code is rare **within one hospital** if it is carried by at most
``--rare-prevalence-max`` (default 5%) of that hospital's patients, and by at
least ``--rare-min-patients`` (default 2) of them. Rare sets are therefore
hospital-specific: a code can be rare at hospital A and common at hospital B.
This is deliberately different from the cross-hospital "present in <= k
hospitals" notion in ``hospital_stats.py``.

Why the split is stratified
---------------------------
A plain random 80/20 leaves many rare codes entirely on one side. Any code with
zero validation patients has a *real* prevalence of exactly 0, so the "correct"
behaviour for a generator becomes never emitting it -- rewarding exactly the
tail-dropping failure this experiment exists to detect. We therefore stratify:
every rare code is its own stratum, and the splitter guarantees each rare code
lands >= 1 patient on **both** sides.

Because a patient usually carries several rare codes at once, "one bin per rare
code" does not assign patients uniquely. We use iterative multilabel
stratification (Sechidis et al. 2011): repeatedly take the rare code with the
fewest unassigned patients left, and place each of its patients into whichever
fold most needs that code.

This is a CPU job (Delta rejects zero-GPU jobs on a *-delta-gpu account, so the
wrapper requests one idle GPU it never touches). Run it on a compute node, NOT
the login node -- see ``run_freeze_split.sh``.

Example:
    python examples/fedpyhealth/freeze_cohort_split.py \
        --out examples/fedpyhealth/cohorts/rare8_v1.json
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

# Kept in sync with ehr_eicu.py so the frozen cohort matches what training uses.
EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1

# The fixed cohort. Order is cosmetic (logging / JSON key order) -- unlike the
# legacy standard_8 manifest, each hospital's split is fully determined by its
# own frozen patient-id lists, not by its position in this list.
COHORT: Tuple[str, ...] = ("458", "443", "148", "79", "388", "244", "202", "272")

SCHEMA_VERSION = 2


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--eicu-root", default=EICU_ROOT, help="path to eICU CRD root")
    p.add_argument("--cohort", default=",".join(COHORT),
                   help="comma-separated hospital ids to freeze")
    p.add_argument("--cohort-name", default="rare8_v1",
                   help="name recorded in the manifest meta block")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="per-hospital validation fraction")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed; only breaks ties in the stratifier")
    p.add_argument("--rare-prevalence-max", type=float, default=0.05,
                   help="a code is rare if carried by <= this fraction of the "
                        "hospital's patients")
    p.add_argument("--rare-min-patients", type=int, default=2,
                   help="a code must be carried by >= this many patients to "
                        "count as rare (2 is the floor that makes the "
                        ">=1-per-fold guarantee satisfiable)")
    p.add_argument("--val-frac-tol", type=float, default=0.10,
                   help="how far the realized validation fraction may drift "
                        "from --val-frac before failing. The >=1-per-fold "
                        "guarantee pushes it up when rare codes rarely "
                        "co-occur, so this is a check, not a tight contract")
    p.add_argument("--out", default="examples/fedpyhealth/cohorts/rare8_v1.json",
                   help="manifest path to write")
    p.add_argument("--dev", action="store_true",
                   help="load a small eICU subset (smoke only -- see the note "
                        "about --rare-prevalence-max below)")
    p.add_argument("--no-admission-diagnostics", action="store_true",
                   help="skip the patient-vs-admission diagnostic pass")
    return p


# --------------------------------------------------------------------------- #
# Stage 1: one walk over the dataset                                           #
# --------------------------------------------------------------------------- #
def collect_cohort_patients(
    eicu_root: str, cohort: Sequence[str], dev: bool = False
) -> Tuple[Dict[str, Dict[str, List[str]]], dict]:
    """Walk every task sample once, keeping code sets for the cohort hospitals.

    Args:
        eicu_root: Path to the eICU CRD 2.0 root (the folder with the CSVs).
        cohort: Hospital ids to keep.
        dev: Load a small development subset instead of the full dataset.

    Returns:
        ``(patients, meta)`` where ``patients`` maps
        ``hospital_id -> {patient_id: sorted list of distinct codes}`` and
        ``meta`` carries provenance used to detect a stale manifest later.

    Raises:
        ValueError: If any requested hospital has no samples in the dataset.
    """
    from pyhealth.datasets import eICUDataset
    from pyhealth.tasks import EHRGenerationEICU

    if dev:
        print("!! --dev: loading a SUBSET of eICU. Hospital sizes and rare "
              "codes are NOT representative.", flush=True)
    print(f"Loading eICU from {eicu_root} (dev={dev})...", flush=True)
    base = eICUDataset(root=eicu_root, tables=["diagnosis"], dev=dev)
    samples = base.set_task(EHRGenerationEICU(min_visits=MIN_VISITS))
    index_to_code = {
        v: k for k, v in samples.input_processors["visits"].code_vocab.items()
    }
    n = len(samples)
    vocab_size = samples.input_processors["visits"].vocab_size()
    print(f"Total samples: {n}   code vocab: {vocab_size}", flush=True)
    print("Walking samples (this is the slow part)...", flush=True)

    wanted = set(str(h) for h in cohort)
    patients: Dict[str, Dict[str, List[str]]] = {h: {} for h in wanted}
    visits_per_patient: Dict[str, int] = {}
    n_hospitals_seen: Set[str] = set()

    for i in range(n):
        sample = samples[i]
        hid = str(sample.get("hospital_id", "NA"))
        n_hospitals_seen.add(hid)
        if hid not in wanted:
            continue
        pid = str(sample["patient_id"])
        codes: Set[str] = set()
        visits = sample["visits"].tolist()
        for visit in visits:
            for code_idx in visit:
                code = index_to_code.get(int(code_idx))
                if code in (None, "<pad>", "<unk>"):
                    continue
                codes.add(code)
        if not codes:
            continue  # no usable diagnosis codes -> cannot be stratified
        if pid in patients[hid]:
            raise ValueError(
                f"duplicate patient_id {pid!r} at hospital {hid} -- the "
                "manifest keys on patient id and assumes one sample per patient"
            )
        patients[hid][pid] = sorted(codes)
        visits_per_patient[pid] = len(visits)

        if (i + 1) % 20000 == 0:
            print(f"  ...{i + 1}/{n} samples", flush=True)

    missing = [h for h in cohort if not patients[str(h)]]
    if missing:
        raise ValueError(
            f"cohort hospitals absent or empty in the dataset: {missing}. "
            "Check --eicu-root and whether --dev is filtering them out."
        )

    stays = list(visits_per_patient.values())
    meta = {
        "eicu_root": eicu_root,
        "dev": dev,
        "min_visits": MIN_VISITS,
        "task_name": "ehr_generation_eicu",
        "code_attr": "icd9code",
        "dataset_total_samples": n,
        "code_vocab_size": vocab_size,
        "n_hospitals_total": len(n_hospitals_seen),
        "unit_stays_per_patient": _describe(stays),
    }
    return patients, meta


def _describe(values: Sequence[float]) -> dict:
    """Summarise a list of counts (mean/p50/p90/max) for the manifest."""
    if not values:
        return {"mean": 0.0, "p50": 0, "p90": 0, "max": 0, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "max": int(arr.max()),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# Stage 2: hospital-specific rare codes                                        #
# --------------------------------------------------------------------------- #
def compute_rare_codes(
    patient_codes: Dict[str, List[str]],
    prevalence_max: float = 0.05,
    min_patients: int = 2,
) -> Dict[str, dict]:
    """Find the rare codes of a single hospital.

    A code is rare when it is carried by at least ``min_patients`` patients and
    by no more than ``prevalence_max`` of the hospital's patients.

    Args:
        patient_codes: ``{patient_id: [code, ...]}`` for one hospital.
        prevalence_max: Upper bound on patient-level prevalence (0-1).
        min_patients: Lower bound on the number of patients carrying the code.

    Returns:
        ``{code: {"n_patients": int, "prevalence": float}}``, sorted by code.

    Raises:
        ValueError: If the resulting rare set is empty, which means the
            thresholds are incompatible with this hospital's size (see the
            ``--dev`` note in the module docstring).
    """
    n_h = len(patient_codes)
    counts: Dict[str, int] = {}
    for codes in patient_codes.values():
        for code in codes:
            counts[code] = counts.get(code, 0) + 1

    rare = {}
    for code in sorted(counts):
        n_pat = counts[code]
        prev = n_pat / n_h
        if n_pat >= min_patients and prev <= prevalence_max:
            rare[code] = {"n_patients": n_pat, "prevalence": round(prev, 8)}

    if not rare:
        raise ValueError(
            f"empty rare set: {n_h} patients, prevalence_max={prevalence_max}, "
            f"min_patients={min_patients}. At this hospital size a code needs "
            f"<= {prevalence_max * n_h:.1f} patients to be rare but >= "
            f"{min_patients} to qualify -- the thresholds are incompatible. "
            "Raise --rare-prevalence-max (smoke/--dev runs typically need 0.5)."
        )
    return rare


# --------------------------------------------------------------------------- #
# Stage 3: iterative multilabel stratification                                 #
# --------------------------------------------------------------------------- #
def _pick_fold(
    desired: Dict[str, Dict[str, int]],
    slots: Dict[str, int],
    code: str,
    rng: np.random.Generator,
) -> str:
    """Choose the fold that most needs one more patient carrying ``code``."""
    best = max(desired["train"][code], desired["val"][code])
    cands = [f for f in ("train", "val") if desired[f][code] == best]
    if len(cands) == 1:
        return cands[0]
    best_slots = max(slots[f] for f in cands)
    cands = [f for f in cands if slots[f] == best_slots]
    if len(cands) == 1:
        return cands[0]
    return str(rng.choice(cands))


def iterative_stratified_split(
    patient_codes: Dict[str, List[str]],
    rare_codes: Iterable[str],
    val_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split one hospital's patients 80/20, stratified on every rare code.

    Implements iterative multilabel stratification: the rare code with the
    fewest still-unassigned patients is handled first, and each of its patients
    goes to whichever fold has the largest remaining deficit for that code.
    Committing the scarcest codes while both folds still have slack is what
    makes the >= 1-per-fold guarantee achievable.

    Quotas per rare code ``j`` carried by ``n_j`` patients are
    ``val_j = min(n_j - 1, max(1, round(val_frac * n_j)))``, so every rare code
    keeps at least one patient in each fold. This costs a small upward bias in
    the realized validation fraction and buys a prevalence metric that is not
    dominated by structural zeros.

    Args:
        patient_codes: ``{patient_id: [code, ...]}`` for one hospital.
        rare_codes: The hospital's rare codes.
        val_frac: Target validation fraction.
        seed: Seed used only to break exact ties.

    Returns:
        ``(train_ids, val_ids)``, each sorted. Deterministic for a given seed
        and invariant to the input ordering of ``patient_codes``.
    """
    rare = sorted(rare_codes)
    rare_set = set(rare)
    pids = sorted(patient_codes)  # sorted input => order-invariant output
    labels = {p: sorted(rare_set.intersection(patient_codes[p])) for p in pids}

    patients_with: Dict[str, Set[str]] = {j: set() for j in rare}
    for p in pids:
        for j in labels[p]:
            patients_with[j].add(p)
    n_j = {j: len(patients_with[j]) for j in rare}

    n_total = len(pids)
    n_val = int(round(val_frac * n_total))
    slots = {"train": n_total - n_val, "val": n_val}

    desired = {"train": {}, "val": {}}
    for j in rare:
        n = n_j[j]
        want_val = int(np.floor(val_frac * n + 0.5))
        want_val = max(1, min(n - 1, want_val)) if n >= 2 else 0
        desired["val"][j] = want_val
        desired["train"][j] = n - want_val

    remaining = dict(n_j)
    unassigned = set(pids)
    assignment: Dict[str, str] = {}
    rng = np.random.default_rng(seed)

    while True:
        active = [j for j in rare if remaining[j] > 0]
        if not active:
            break
        # rarest remaining label first; code string breaks ties deterministically
        code = min(active, key=lambda k: (remaining[k], k))
        for pid in sorted(patients_with[code] & unassigned):
            fold = _pick_fold(desired, slots, code, rng)
            assignment[pid] = fold
            unassigned.discard(pid)
            slots[fold] -= 1
            for k in labels[pid]:
                desired[fold][k] -= 1
                remaining[k] -= 1

    # Patients carrying no rare code: fill whichever fold is furthest from quota.
    leftovers = sorted(unassigned)
    for pos in rng.permutation(len(leftovers)):
        pid = leftovers[int(pos)]
        fold = "train" if slots["train"] >= slots["val"] else "val"
        assignment[pid] = fold
        slots[fold] -= 1

    train = sorted(p for p in pids if assignment[p] == "train")
    val = sorted(p for p in pids if assignment[p] == "val")
    return train, val


def verify_split(
    patient_codes: Dict[str, List[str]],
    rare_codes: Iterable[str],
    train: Sequence[str],
    val: Sequence[str],
    val_frac: float = 0.2,
    tol: float = 0.08,
) -> Dict[str, dict]:
    """Assert the split invariants and return per-code train/val counts.

    Raises:
        ValueError: If the folds overlap, do not cover every patient, if any
            rare code is missing from a fold, or if the realized validation
            fraction drifts more than ``tol`` from ``val_frac``.
    """
    train_set, val_set = set(train), set(val)
    if train_set & val_set:
        raise ValueError(f"train/val overlap: {sorted(train_set & val_set)[:5]}")
    if train_set | val_set != set(patient_codes):
        missing = set(patient_codes) - (train_set | val_set)
        raise ValueError(f"{len(missing)} patients unassigned, e.g. "
                         f"{sorted(missing)[:5]}")

    per_code = {}
    for code in sorted(rare_codes):
        n_tr = sum(1 for p in train if code in patient_codes[p])
        n_va = sum(1 for p in val if code in patient_codes[p])
        if n_tr < 1 or n_va < 1:
            raise ValueError(
                f"rare code {code!r} has train={n_tr}, val={n_va}; the "
                ">=1-per-fold guarantee was violated"
            )
        per_code[code] = {"n_train": n_tr, "n_val": n_va}

    realized = len(val) / max(1, len(patient_codes))
    if abs(realized - val_frac) > tol:
        raise ValueError(
            f"realized val fraction {realized:.4f} drifts more than {tol} from "
            f"the {val_frac} target; with many 2-patient rare codes the "
            ">=1-per-fold guarantee can push it up -- inspect before relaxing"
        )
    return per_code


# --------------------------------------------------------------------------- #
# Stage 4: patient-vs-admission diagnostics (informational; gates nothing)     #
# --------------------------------------------------------------------------- #
def admission_diagnostics(eicu_root: str, cohort: Sequence[str]) -> dict:
    """Count hospital admissions per patient for the cohort hospitals.

    Recorded as evidence for the deferred question of whether to re-run the
    experiment with the admission (``patienthealthsystemstayid``) as the sample
    unit instead of the patient. Reads ``patient.csv`` directly because the
    generation task does not surface admission ids. Best-effort: any failure
    returns a ``{"error": ...}`` stub rather than sinking the freeze job.
    """
    try:
        import polars as pl

        path = os.path.join(eicu_root, "patient.csv")
        frame = pl.read_csv(
            path,
            columns=["uniquepid", "patienthealthsystemstayid", "hospitalid"],
            schema_overrides={"uniquepid": pl.Utf8, "hospitalid": pl.Utf8},
        ).filter(pl.col("hospitalid").is_in([str(h) for h in cohort]))

        per_patient = frame.group_by("uniquepid").agg(
            pl.col("patienthealthsystemstayid").n_unique().alias("n_adm")
        )
        counts = per_patient["n_adm"].to_list()
        multi = sum(1 for c in counts if c >= 2)
        return {
            "admissions_per_patient": _describe(counts),
            "frac_patients_multi_admission": round(multi / max(1, len(counts)), 4),
            "n_patients_in_patient_csv": len(counts),
            "note": "informational only; the frozen split uses sample_unit=patient",
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
        return {"error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def _sha256_list(items: Sequence[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode())
        h.update(b"\0")
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest(args) -> dict:
    """Run every stage and assemble the manifest dict."""
    cohort = [h.strip() for h in args.cohort.split(",") if h.strip()]
    patients, meta = collect_cohort_patients(args.eicu_root, cohort, args.dev)

    hospitals = []
    pooled_rare: Set[str] = set()
    for order, hid in enumerate(cohort):
        pc = patients[hid]
        rare = compute_rare_codes(
            pc, args.rare_prevalence_max, args.rare_min_patients
        )
        train, val = iterative_stratified_split(
            pc, rare, val_frac=args.val_frac, seed=args.seed
        )
        # Print before verifying: if the fraction check trips, the log already
        # shows every hospital's realized numbers instead of just the failure.
        print(f"  [{hid}] {len(pc)} patients -> {len(train)} train / "
              f"{len(val)} val (frac={len(val) / len(pc):.4f}), "
              f"{len(rare)} rare codes", flush=True)
        per_code = verify_split(
            pc, rare, train, val, val_frac=args.val_frac, tol=args.val_frac_tol
        )
        for code, stats in per_code.items():
            rare[code].update(stats)
        pooled_rare.update(rare)

        support = [v["n_patients"] for v in rare.values()]
        hospitals.append({
            "hospital_id": hid,
            "order": order,
            "n_total": len(pc),
            "n_train": len(train),
            "n_val": len(val),
            "n_unique_codes": len({c for cs in pc.values() for c in cs}),
            "n_rare_codes": len(rare),
            "rare_support_min": int(min(support)),
            "rare_support_median": int(np.percentile(support, 50)),
            "min_rare_prevalence": min(v["prevalence"] for v in rare.values()),
            "rare_codes": rare,
            "train_patient_ids": train,
            "val_patient_ids": val,
            "train_patient_ids_sha256": _sha256_list(train),
            "val_patient_ids_sha256": _sha256_list(val),
        })

    pooled = sorted(pooled_rare)
    total = sum(h["n_total"] for h in hospitals)
    n_train = sum(h["n_train"] for h in hospitals)
    n_val = sum(h["n_val"] for h in hospitals)

    diagnostics = ({} if args.no_admission_diagnostics
                   else admission_diagnostics(args.eicu_root, cohort))

    meta.update({
        "cohort_name": args.cohort_name,
        "selection": "explicit_hospital_ids",
        "sample_unit": "patient",
        "split_algorithm": "iterative_multilabel_stratification",
        "split_algorithm_version": "1.0",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "rare_prevalence_max": args.rare_prevalence_max,
        "rare_min_patients": args.rare_min_patients,
        "guarantee_val_per_rare_code": True,
        "pyhealth_git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cohort_total": total,
        "cohort_train": n_train,
        "cohort_val": n_val,
        "realized_val_frac": round(n_val / max(1, total), 4),
        "pooled_rare_codes": pooled,
        "pooled_rare_codes_sha256": _sha256_list(pooled),
        "n_pooled_rare_codes": len(pooled),
        "unit_diagnostics": diagnostics,
    })
    return {"schema_version": SCHEMA_VERSION, "meta": meta,
            "hospitals": hospitals}


def summarize(manifest: dict) -> dict:
    """Strip patient ids, keeping aggregates + hashes for a committable file."""
    keep = ("hospital_id", "order", "n_total", "n_train", "n_val",
            "n_unique_codes", "n_rare_codes", "rare_support_min",
            "rare_support_median", "min_rare_prevalence",
            "train_patient_ids_sha256", "val_patient_ids_sha256")
    meta = {k: v for k, v in manifest["meta"].items()
            if k != "pooled_rare_codes"}
    return {
        "schema_version": manifest["schema_version"],
        "meta": meta,
        "hospitals": [{k: h[k] for k in keep} for h in manifest["hospitals"]],
    }


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    manifest = build_manifest(args)
    meta = manifest["meta"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    summary = summarize(manifest)
    summary["meta"]["manifest_sha256"] = manifest_sha
    summary_path = args.out.replace(".json", ".summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nWrote {args.out}")
    print(f"Wrote {summary_path}  (no patient ids -- safe to commit)")
    print(f"manifest sha256: {manifest_sha}")
    print(f"\ncohort: {meta['cohort_total']} patients -> "
          f"{meta['cohort_train']} train / {meta['cohort_val']} val "
          f"(val_frac={meta['realized_val_frac']})")
    print(f"pooled rare codes: {meta['n_pooled_rare_codes']}")

    smallest = min(h["min_rare_prevalence"] for h in manifest["hospitals"])
    floor = int(np.ceil(10 / smallest)) if smallest > 0 else 0
    print(f"\nsmallest rare prevalence: {smallest:.6f}")
    print(f"=> a synthetic set resolves prevalence only to 1/num_synth; use "
          f"--num-synth >= {floor} to keep rare R^2 out of the quantization "
          f"noise floor.")

    diag = meta.get("unit_diagnostics", {})
    if "frac_patients_multi_admission" in diag:
        print(f"\n[deferred] {diag['frac_patients_multi_admission']:.1%} of "
              f"cohort patients have >=2 hospital admissions "
              f"(mean {diag['admissions_per_patient']['mean']}). Below ~10% "
              "means an admission-unit re-run would mostly reproduce this one.")


if __name__ == "__main__":
    main()
