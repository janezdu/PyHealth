"""Prepare the eICU dataset for the federated generation task.

Everything between "raw eICU" and "data the experiment can train on" lives here,
as two steps you run once:

    # 1. preview -- seconds. Which 8 hospitals? Reads patient.csv only, draws
    #    two per size band, writes the selection file.
    python examples/fedpyhealth/prepare_dataset.py preview \
        --size-bands 0-199,200-499,500-1999,2000- --per-band 2 --seed 2

    # 2. freeze -- ~1-2h. Splits each of those hospitals 80/20 and writes the
    #    manifest every regime then loads.
    python examples/fedpyhealth/prepare_dataset.py freeze

The freeze is the part that matters for correctness. A selection file names
hospitals; it does not say which patients are train and which are validation.
If each run decided that for itself, fedavg / local / centralized would each
train on a different 80% and score against a different 20%, and comparing them
would mean nothing. So the split is computed once, written down as explicit
patient-id lists, and loaded by every run.

The split is also not a plain random 80/20: it is stratified so every rare code
lands at least one patient on *both* sides. A rare code with zero validation
patients has a real prevalence of exactly 0, which makes "never emit it" the
optimal behaviour -- rewarding precisely the tail-dropping this experiment
exists to detect.

Neither step needs a GPU. Outputs go to ``cohorts/``: ``<name>.cohort.json``
(selection, committed) and ``<name>.json`` (the frozen manifest, which carries
patient ids and is gitignored) plus ``<name>.summary.json`` (aggregates and
hashes, committed).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1
DEFAULT_BANDS = "0-199,200-499,500-1999,2000-"
DEFAULT_PREVIEW_CALIBRATE = "examples/fedpyhealth/cohorts/*.summary.json"
DEFAULT_SELECTION_FILE = "examples/fedpyhealth/cohorts/strat8.cohort.json"
DEFAULT_PREVIEW_OUT = "examples/fedpyhealth/cohorts/strat8.cohort.json"
DEFAULT_FREEZE_OUT = "examples/fedpyhealth/cohorts/strat8.json"


# --------------------------------------------------------------------------- #
# Shared preview helpers                                                       #
# --------------------------------------------------------------------------- #
def parse_size_bands(spec: str) -> List[Tuple[int, int]]:
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
        raise ValueError("--size-bands parsed to zero bands")
    return bands


def band_candidates(
    sizes: Dict[str, int],
    bands: List[Tuple[int, int]],
    min_samples: int,
) -> Dict[int, List[Tuple[str, int]]]:
    """Group eligible hospitals into the bands they fall in, largest first."""
    eligible = sorted(
        ((hid, n) for hid, n in sizes.items() if n >= min_samples),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {b: [(hid, n) for hid, n in eligible if lo <= n <= hi]
            for b, (lo, hi) in enumerate(bands)}


def band_draw(
    sizes: Dict[str, int],
    bands: List[Tuple[int, int]],
    per_band: int,
    min_samples: int,
    seed: int,
) -> List[dict]:
    """Draw ``per_band`` hospitals from each explicit size band."""
    candidates = band_candidates(sizes, bands, min_samples)
    short = [(bands[b], len(candidates[b])) for b in range(len(bands))
             if len(candidates[b]) < per_band]
    if short:
        detail = "; ".join(f"band {lo}-{hi}: {n} eligible" for (lo, hi), n in short)
        raise ValueError(
            f"cannot draw {per_band} per band -- {detail}. Widen the band, lower "
            f"--min-hospital-samples (now {min_samples}), or drop --per-band."
        )

    rng = np.random.default_rng(seed)
    chosen: List[dict] = []
    for b in sorted(range(len(bands)), key=lambda i: bands[i][0], reverse=True):
        pool = candidates[b]
        picks = rng.choice(len(pool), size=per_band, replace=False)
        for pos in sorted(int(p) for p in picks):
            hid, n = pool[pos]
            chosen.append({"hospital_id": hid, "size_bin": b, "n_total": n})
    return chosen


def csv_hospital_sizes(eicu_root: str) -> Dict[str, int]:
    """Count distinct patients per hospital straight from ``patient.csv``."""
    import polars as pl

    path = os.path.join(eicu_root, "patient.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no patient.csv under {eicu_root!r} -- point --eicu-root at the "
            "folder holding the eICU CRD 2.0 CSVs"
        )
    frame = pl.read_csv(
        path, columns=["uniquepid", "hospitalid"],
        schema_overrides={"uniquepid": pl.Utf8, "hospitalid": pl.Utf8},
    )
    counts = frame.group_by("hospitalid").agg(
        pl.col("uniquepid").n_unique().alias("n")
    )
    return {row["hospitalid"]: int(row["n"]) for row in counts.to_dicts()}


def calibration_ratio(csv_sizes: Dict[str, int], pattern: str) -> Tuple[float, list]:
    """Estimate task-count / csv-count from already-frozen manifests."""
    pairs = []
    for path in sorted(glob.glob(pattern)) if pattern else []:
        try:
            with open(path) as fh:
                manifest = json.load(fh)
            for h in manifest.get("hospitals", []):
                hid, task_n = str(h["hospital_id"]), int(h["n_total"])
                csv_n = csv_sizes.get(hid)
                if csv_n:
                    pairs.append((hid, csv_n, task_n, task_n / csv_n))
        except Exception as exc:  # noqa: BLE001 - calibration is best-effort
            print(f"  (skipped {path}: {type(exc).__name__}: {exc})")
    if not pairs:
        return 1.0, []
    return float(np.median([p[3] for p in pairs])), pairs


def _near_edge(n: int, bands: List[Tuple[int, int]], margin: float) -> bool:
    """True if ``n`` sits within ``margin`` (relative) of any band boundary."""
    for lo, hi in bands:
        for edge in (lo, hi):
            if 0 < edge < 10 ** 9 and abs(n - edge) <= margin * edge:
                return True
    return False


def build_preview_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--eicu-root", default=EICU_ROOT, help="path to eICU CRD root")
    p.add_argument("--size-bands", default=DEFAULT_BANDS,
                   help="inclusive patient-count bands, e.g. "
                        "'0-199,200-499,500-1999,2000-'; a trailing '-' means "
                        "no upper bound")
    p.add_argument("--per-band", type=int, default=2,
                   help="hospitals to draw from each band")
    p.add_argument("--min-hospital-samples", type=int, default=100,
                   help="global floor; smaller hospitals are never eligible")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the within-band draw")
    p.add_argument("--list-only", action="store_true",
                   help="print every eligible hospital per band and exit, so "
                        "you can hand-pick instead of drawing")
    p.add_argument("--calibrate", default=DEFAULT_PREVIEW_CALIBRATE,
                   help="glob of frozen summary manifests used to estimate the "
                        "csv-count -> task-count ratio; empty to skip")
    p.add_argument("--edge-margin", type=float, default=0.15,
                   help="flag a hospital whose estimate is within this relative "
                        "distance of a band edge")
    p.add_argument("--top", type=int, default=25,
                   help="how many hospitals to print per band")
    p.add_argument("--out", default=DEFAULT_PREVIEW_OUT,
                   help="selection file to write (hospital ids + the bands, "
                        "seed and calibration they came from). This is what "
                        "the freeze step reads")
    p.add_argument("--cohort-name", default="strat8",
                   help="name recorded in the selection file")
    return p


def preview_main(argv: Sequence[str] | None = None) -> None:
    args = build_preview_parser().parse_args(argv)
    bands = parse_size_bands(args.size_bands)

    csv_sizes = csv_hospital_sizes(args.eicu_root)
    print(f"patient.csv: {len(csv_sizes)} hospitals, "
          f"{sum(csv_sizes.values())} patients")

    ratio, pairs = calibration_ratio(csv_sizes, args.calibrate)
    if pairs:
        spread = [f"{hid}:{r:.2f}" for hid, _, _, r in sorted(pairs)[:8]]
        print(f"calibration from {len(pairs)} already-frozen hospitals: "
              f"task/csv median = {ratio:.3f}  ({', '.join(spread)}...)")
    else:
        print("calibration: no frozen manifest matched --calibrate; "
              "using csv counts as-is (they overshoot the task counts)")

    est = {hid: int(round(n * ratio)) for hid, n in csv_sizes.items()}

    print(f"\nEstimated task patients per hospital, by band "
          f"(>= {args.min_hospital_samples}):")
    candidates = band_candidates(est, bands, args.min_hospital_samples)
    for b, (lo, hi) in enumerate(bands):
        pool = candidates[b]
        hi_s = "inf" if hi >= 10 ** 9 else str(hi)
        shown = pool[:args.top]
        more = f" ... +{len(pool) - len(shown)} more" if len(pool) > len(shown) else ""
        print(f"  band {b} [{lo}-{hi_s}]: {len(pool)} hospitals")
        print("    " + ", ".join(
            f"{hid}({n}{'*' if _near_edge(n, bands, args.edge_margin) else ''})"
            for hid, n in shown) + more)
    print("  (* = within {:.0%} of a band edge; the estimate may be on the "
          "wrong side)".format(args.edge_margin))

    if args.list_only:
        print("\n--list-only: no draw. Pass ids you like straight to "
              "the freeze step with --cohort.")
        return

    chosen = band_draw(est, bands, args.per_band, args.min_hospital_samples,
                       args.seed)
    print(f"\nDrawn cohort (seed={args.seed}, {args.per_band} per band):")
    for c in chosen:
        lo, hi = bands[c["size_bin"]]
        hi_s = "inf" if hi >= 10 ** 9 else str(hi)
        print(f"  band {c['size_bin']} [{lo}-{hi_s}]  hospital {c['hospital_id']:>5}"
              f"  ~{c['n_total']} patients (est)")
    total = sum(c["n_total"] for c in chosen)
    print(f"  estimated cohort total: ~{total} patients "
          f"(~{int(total * 0.8)} train)")

    ids = ",".join(c["hospital_id"] for c in chosen)
    selection = {
        "kind": "selection",
        "cohort_name": args.cohort_name,
        "meta": {
            "selection": "size_bands",
            "size_bands": args.size_bands,
            "per_band": args.per_band,
            "min_hospital_samples": args.min_hospital_samples,
            "seed": args.seed,
            "eicu_root": args.eicu_root,
            "calibration_ratio": round(ratio, 4),
            "n_patients_is_estimate": True,
            "produced_by": "prepare_dataset.py preview",
        },
        "hospitals": [
            {"hospital_id": c["hospital_id"], "size_band": c["size_bin"],
             "n_patients_est": c["n_total"]} for c in chosen
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(selection, fh, indent=2)

    print(f"\nhospital ids: {ids}")
    print(f"Wrote {args.out}")
    print("Freeze the real (rare-code-stratified) split with:")
    print(f"  python examples/fedpyhealth/prepare_dataset.py freeze "
          f"--cohort-file {args.out}")
    print("\nThe freeze step records each hospital's TRUE patient count; if one "
          "lands outside its band there, re-draw with another --seed.")


# --------------------------------------------------------------------------- #
# Freeze workflow                                                              #
# --------------------------------------------------------------------------- #
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


def cross_hospital_patients(eicu_root: str) -> Set[str]:
    """Find patients with unit stays at more than one hospital."""
    path = os.path.join(eicu_root, "patient.csv")
    try:
        seen: Dict[str, Set[str]] = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                seen.setdefault(row["uniquepid"], set()).add(row["hospitalid"])
        return {pid for pid, hospitals in seen.items() if len(hospitals) > 1}
    except (OSError, KeyError) as exc:
        print(f"WARNING: cannot read {path} ({exc}); cross-hospital patients "
              "will NOT be filtered", flush=True)
        return set()


def collect_cohort_patients(
    eicu_root: str, cohort: Sequence[str], dev: bool = False,
    exclude: Set[str] = frozenset(),
) -> Tuple[Dict[str, Dict[str, List[str]]], dict]:
    """Walk every task sample once, keeping code sets for the cohort hospitals."""
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
    n_excluded = 0

    for i in range(n):
        sample = samples[i]
        hid = str(sample.get("hospital_id", "NA"))
        n_hospitals_seen.add(hid)
        if hid not in wanted:
            continue
        pid = str(sample["patient_id"])
        if pid in exclude:
            n_excluded += 1
            continue
        codes: Set[str] = set()
        visits = sample["visits"].tolist()
        for visit in visits:
            for code_idx in visit:
                code = index_to_code.get(int(code_idx))
                if code in (None, "<pad>", "<unk>"):
                    continue
                codes.add(code)
        if not codes:
            continue
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
        "n_cross_hospital_patients_dropped": n_excluded,
    }
    return patients, meta


def compute_rare_codes(
    patient_codes: Dict[str, List[str]],
    prevalence_max: float = 0.05,
    min_patients: int = 2,
) -> Dict[str, dict]:
    """Find the rare codes of a single hospital."""
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
    """Split one hospital's patients 80/20, stratified on every rare code."""
    rare = sorted(rare_codes)
    rare_set = set(rare)
    pids = sorted(patient_codes)
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
        code = min(active, key=lambda k: (remaining[k], k))
        for pid in sorted(patients_with[code] & unassigned):
            fold = _pick_fold(desired, slots, code, rng)
            assignment[pid] = fold
            unassigned.discard(pid)
            slots[fold] -= 1
            for k in labels[pid]:
                desired[fold][k] -= 1
                remaining[k] -= 1

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
    """Assert the split invariants and return per-code train/val counts."""
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


def admission_diagnostics(eicu_root: str, cohort: Sequence[str]) -> dict:
    """Count hospital admissions per patient for the cohort hospitals."""
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
    cohort = resolve_cohort(args)

    exclude: Set[str] = set()
    if args.cross_hospital == "drop":
        print("Scanning patient.csv for cross-hospital patients...", flush=True)
        exclude = cross_hospital_patients(args.eicu_root)
        print(f"  {len(exclude)} patients (all hospitals) have stays at >1 "
              "site; cohort members among them will be dropped", flush=True)

    patients, meta = collect_cohort_patients(
        args.eicu_root, cohort, args.dev, exclude=exclude
    )
    print(f"  dropped {meta['n_cross_hospital_patients_dropped']} cohort "
          "patients as cross-hospital", flush=True)

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

    val_support: Dict[str, int] = {c: 0 for c in pooled}
    train_support: Dict[str, int] = {c: 0 for c in pooled}
    for h in hospitals:
        pc = patients[h["hospital_id"]]
        for pid in h["val_patient_ids"]:
            for code in pc[pid]:
                if code in val_support:
                    val_support[code] += 1
        for pid in h["train_patient_ids"]:
            for code in pc[pid]:
                if code in train_support:
                    train_support[code] += 1

    global_max = args.global_rare_prevalence_max
    global_rare = sorted(
        c for c in pooled
        if (train_support[c] + val_support[c]) / max(1, total) <= global_max
    )
    print(f"\npooled rare codes: {len(pooled)}   of which globally rare "
          f"(cohort prevalence <= {global_max}): {len(global_rare)}", flush=True)

    hist = {}
    for thresh in (1, 2, 3, 5, 10, 20, 50, 100):
        hist[f"ge_{thresh}"] = sum(1 for c in pooled if val_support[c] >= thresh)
    print("scorable pooled rare codes by validation support: "
          + "  ".join(f"{k}={v}" for k, v in hist.items()), flush=True)

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
        "global_rare_prevalence_max": args.global_rare_prevalence_max,
        "cross_hospital": args.cross_hospital,
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
        "global_rare_codes": global_rare,
        "global_rare_codes_sha256": _sha256_list(global_rare),
        "n_global_rare_codes": len(global_rare),
        "pooled_rare_val_support": val_support,
        "pooled_rare_train_support": train_support,
        "val_support_histogram": hist,
        "unit_diagnostics": diagnostics,
    })
    return {"meta": meta,
            "hospitals": hospitals}


def summarize(manifest: dict) -> dict:
    """Strip patient ids, keeping aggregates + hashes for a committable file."""
    keep = ("hospital_id", "order", "n_total", "n_train", "n_val",
            "n_unique_codes", "n_rare_codes", "rare_support_min",
            "rare_support_median", "min_rare_prevalence",
            "train_patient_ids_sha256", "val_patient_ids_sha256")
    bulky = ("pooled_rare_codes", "global_rare_codes",
             "pooled_rare_val_support", "pooled_rare_train_support")
    meta = {k: v for k, v in manifest["meta"].items() if k not in bulky}
    root = meta.get("eicu_root")
    if root:
        meta["eicu_root"] = os.path.join("<redacted>",
                                         *os.path.normpath(root).split(os.sep)[-2:])
    return {
        "meta": meta,
        "hospitals": [{k: h[k] for k in keep} for h in manifest["hospitals"]],
    }


def build_freeze_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--eicu-root", default=EICU_ROOT, help="path to eICU CRD root")
    p.add_argument("--cohort", default="",
                   help="comma-separated hospital ids to freeze; usually left "
                        "empty in favour of --cohort-file")
    p.add_argument("--cohort-file", default=DEFAULT_SELECTION_FILE,
                   help="selection file written by the preview step; its "
                        "hospital ids are what gets frozen. Ignored when "
                        "--cohort is given")
    p.add_argument("--cohort-name", default="strat8",
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
    p.add_argument("--global-rare-prevalence-max", type=float, default=0.01,
                   help="strict pool: a pooled rare code also counts as "
                        "globally rare if carried by <= this fraction of the "
                        "whole cohort. Separates true long-tail codes from "
                        "codes that are merely rare at one site")
    p.add_argument("--cross-hospital", choices=("drop", "keep"), default="drop",
                   help="what to do with patients who have stays at more than "
                        "one hospital. The task assigns them to a single "
                        "client but keeps every stay, so 'keep' leaks one "
                        "site's visits into another site's client")
    p.add_argument("--val-frac-tol", type=float, default=0.10,
                   help="how far the realized validation fraction may drift "
                        "from --val-frac before failing. The >=1-per-fold "
                        "guarantee pushes it up when rare codes rarely "
                        "co-occur, so this is a check, not a tight contract")
    p.add_argument("--out", default=DEFAULT_FREEZE_OUT,
                   help="manifest path to write")
    p.add_argument("--dev", action="store_true",
                   help="load a small development subset (smoke only)")
    p.add_argument("--no-admission-diagnostics", action="store_true",
                   help="skip the patient-vs-admission diagnostic pass")
    p.add_argument("--self-test", action="store_true",
                   help="validate the stratifier on synthetic data and exit; "
                        "no eICU, no pytest, runs in under a second")
    return p


def self_test() -> None:
    """Assert the splitter's invariants on hand-built data."""
    def disjoint(sizes):
        return {f"p{n}_{i:03d}": [f"c{n}", "common"]
                for n in sizes for i in range(n)}

    expected = {2: (1, 1), 3: (2, 1), 4: (3, 1), 5: (4, 1),
                7: (6, 1), 8: (6, 2), 9: (7, 2), 10: (8, 2)}
    for n, want in expected.items():
        pc = disjoint([n])
        train, val = iterative_stratified_split(pc, [f"c{n}"], 0.2, seed=0)
        got = (sum(1 for p in train if f"c{n}" in pc[p]),
               sum(1 for p in val if f"c{n}" in pc[p]))
        assert got == want, f"quota n={n}: got {got}, want {want}"

    sizes = list(range(2, 11))
    pc = disjoint(sizes)
    rare = [f"c{n}" for n in sizes]
    train, val = iterative_stratified_split(pc, rare, 0.2, seed=0)

    assert not set(train) & set(val), "folds overlap"
    assert set(train) | set(val) == set(pc), "some patients unassigned"
    for code in rare:
        assert any(code in pc[p] for p in train), f"{code} missing from train"
        assert any(code in pc[p] for p in val), f"{code} missing from val"

    assert iterative_stratified_split(pc, rare, 0.2, seed=0) == (train, val)
    items = list(pc.items())
    np.random.default_rng(7).shuffle(items)
    assert iterative_stratified_split(dict(items), rare, 0.2, seed=0) == (
        train, val), "split changed when input order changed"
    assert iterative_stratified_split(
        pc, list(reversed(rare)), 0.2, seed=0) == (train, val)

    verify_split(pc, rare, train, val, 0.2, tol=1.0)
    try:
        verify_split(pc, rare, sorted(pc), [], 0.2, tol=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("verify_split accepted an empty validation fold")

    try:
        compute_rare_codes({f"p{i}": ["a", "b"] for i in range(20)}, 0.05, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("compute_rare_codes accepted an empty rare set")

    pc2 = {f"p{i:03d}": ["common"] for i in range(100)}
    for i in range(4):
        pc2[f"p{i:03d}"].append("rare4")
    for i in range(5):
        pc2[f"p{i:03d}"].append("exactly5pct")
    for i in range(6):
        pc2[f"p{i:03d}"].append("over5pct")
    pc2["p000"].append("singleton")
    found = set(compute_rare_codes(pc2, 0.05, 2))
    assert found == {"rare4", "exactly5pct"}, f"threshold semantics: {found}"

    print("self-test: all stratifier invariants hold")



def resolve_cohort(args) -> List[str]:
    """Hospital ids to freeze: ``--cohort`` if given, else the selection file.

    Keeping the ids in a selection file rather than retyping them is what makes
    the freeze reproducible -- the file records the bands, seed and calibration
    the draw came from.

    Raises:
        SystemExit: If neither source yields any hospital id.
    """
    if args.cohort.strip():
        return [h.strip() for h in args.cohort.split(",") if h.strip()]

    path = args.cohort_file
    if not path or not os.path.exists(path):
        raise SystemExit(
            f"no hospitals to freeze: --cohort is empty and {path!r} does not "
            "exist. Run the preview step first "
            "(python examples/fedpyhealth/prepare_dataset.py preview ...), or "
            "pass --cohort 420,199,..."
        )
    with open(path) as fh:
        selection = json.load(fh)
    ids = [str(h["hospital_id"]) for h in selection.get("hospitals", [])]
    if not ids:
        raise SystemExit(f"{path} lists no hospitals")
    print(f"cohort from {path}: {', '.join(ids)}")
    return ids


def freeze_main(argv: Sequence[str] | None = None) -> None:
    args = build_freeze_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return
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
    print(f"pooled rare codes: {meta['n_pooled_rare_codes']}  "
          f"(globally rare: {meta['n_global_rare_codes']})")
    hist = meta["val_support_histogram"]
    print("scorable by validation support: "
          + "  ".join(f"{k.replace('ge_', '>=')}: {v}" for k, v in hist.items()))
    if meta["cross_hospital"] == "drop":
        print(f"cross-hospital patients dropped: "
              f"{meta['n_cross_hospital_patients_dropped']}")
    else:
        print("WARNING: --cross-hospital keep -- clients are NOT disjoint")

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


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("choose 'preview' or 'freeze'")
    mode, rest = args[0], args[1:]
    if mode == "preview":
        preview_main(rest)
    elif mode == "freeze":
        freeze_main(rest)
    elif mode in ("-h", "--help"):
        print(__doc__ or "")
    else:
        raise SystemExit(f"unknown mode {mode!r}; use 'preview' or 'freeze'")


if __name__ == "__main__":
    main()
