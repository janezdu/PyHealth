"""Select and freeze a fixed federated cohort of K hospitals (stratified by size).

Run this ONCE to materialize the "standard run" cohort used by every baseline in
the federated-EHR table (Centralized / Local / FedAvg / FedAvg+FT / FedEHR-Gen).
Rather than re-drawing clients at the start of each experiment, we draw a
*size-spread* sample of K hospitals a single time, cache the chosen ids + basic
per-hospital EDA to a JSON manifest, and have ``ehr_eicu.py --cohort-file`` load
that manifest. This guarantees all baselines compare on the identical client set.

The draw is **stratified by hospital size**: eligible hospitals (>= ``min_samples``)
are sorted by sample count and split into ``n_clients`` contiguous size-bins, then
one hospital is drawn at random from each bin. The result deterministically spans
big and small hospitals (one per stratum) instead of collapsing onto the K
largest -- a more realistic, heterogeneous federation. It is reproducible via
``--seed``.

This is a CPU job (no GPU): it loads eICU, applies the task, does the draw and a
pandas-free EDA pass over the chosen hospitals' samples. Run it on a compute node
(NOT the login node) -- see the sample command at the bottom of this file.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from pyhealth.datasets import eICUDataset
from pyhealth.tasks import EHRGenerationEICU

# Kept in sync with ehr_eicu.py so the frozen cohort matches what training uses.
EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1
# Calibrated from run 19227134 (warm cache): seconds per (sample x local-epoch).
# Used only to print suggested n_rounds for a wall-clock budget; not authoritative.
SEC_PER_SAMPLE_EPOCH = 0.117


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--eicu-root", default=EICU_ROOT, help="path to eICU CRD root")
    p.add_argument("--n-clients", type=int, default=8,
                   help="number of hospitals to draw (one per size-bin)")
    p.add_argument("--min-hospital-samples", type=int, default=50,
                   help="exclude hospitals with fewer samples than this")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="per-hospital held-out fraction (matches ehr_eicu.py)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the bin draws and per-hospital splits")
    p.add_argument("--out", default="examples/fedpyhealth/cohorts/standard_8.json",
                   help="manifest path to write")
    p.add_argument("--field-index-cache",
                   default="examples/fedpyhealth/cohorts/.field_index.json",
                   help="cache file for the hospital->indices map (skips the slow "
                        "sample walk on re-runs); set empty to disable")
    p.add_argument("--rebuild-index", action="store_true",
                   help="force rebuilding the hospital index cache")
    return p


def build_field_index(
    dataset,
    field: str = "hospital_id",
    cache_path: str = None,
    eicu_root: str = None,
    rebuild: bool = False,
) -> Dict[str, List[int]]:
    """Map each hospital id to its sample indices (one streaming pass).

    The pass walks every sample single-threaded (``dataset[i]`` deserializes from
    the on-disk cache), which dominates this script's runtime. When ``cache_path``
    is given, the index is cached to JSON keyed by ``(len(dataset), eicu_root)``;
    a later run with a matching dataset loads it instantly instead of re-walking.
    Pass ``rebuild=True`` to force a fresh pass (e.g. after the dataset changes).
    """
    n = len(dataset)
    if cache_path and not rebuild and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("n_samples") == n and cached.get("eicu_root") == eicu_root:
            print(f"Loaded hospital index from cache {cache_path} "
                  f"({len(cached['field_index'])} hospitals, {n} samples)")
            return cached["field_index"]
        print(f"Index cache {cache_path} stale "
              f"(n_samples/eicu_root mismatch); rebuilding")

    field_index: Dict[str, List[int]] = {}
    for i in range(n):
        key = str(dataset[i].get(field, "NA"))
        field_index.setdefault(key, []).append(i)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"n_samples": n, "eicu_root": eicu_root,
                       "field_index": field_index}, f)
        print(f"Cached hospital index -> {cache_path}")
    return field_index


def stratified_draw(
    field_index: Dict[str, List[int]],
    n_clients: int,
    min_samples: int,
    seed: int,
) -> List[dict]:
    """Draw one hospital per size-bin; return ordered records (largest bin first).

    Eligible hospitals are sorted by size descending and split into ``n_clients``
    contiguous bins; one hospital is drawn at random from each bin. Returns a list
    of ``{"hospital_id", "size_bin", "n_total"}`` in bin order, so downstream
    per-hospital train/test splitting (seeded by enumerate index) is reproducible.
    """
    eligible = sorted(
        ((hid, idxs) for hid, idxs in field_index.items() if len(idxs) >= min_samples),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    if len(eligible) < n_clients:
        raise ValueError(
            f"Only {len(eligible)} hospitals have >= {min_samples} samples; "
            f"need >= {n_clients}."
        )
    rng = np.random.default_rng(seed)
    bins = np.array_split(np.arange(len(eligible)), n_clients)
    chosen: List[dict] = []
    for b, bin_positions in enumerate(bins):
        pick = int(rng.choice(bin_positions))
        hid, idxs = eligible[pick]
        chosen.append({"hospital_id": hid, "size_bin": b, "n_total": len(idxs),
                       "_idxs": idxs})
    return chosen


def hospital_eda(dataset, idxs: List[int], index_to_code: Dict[int, str],
                 test_frac: float, split_seed: int) -> dict:
    """Compute basic EDA for one hospital and its reproducible train/test sizes.

    The split mirrors ``ehr_eicu.partition_by_hospital`` exactly (shuffle with
    ``default_rng(split_seed)``, ``n_test = max(1, round(n * test_frac))``) so the
    recorded sizes match what training will see.
    """
    # Reproduce the per-hospital split (must match ehr_eicu.partition_by_hospital).
    rng = np.random.default_rng(split_seed)
    idx = list(idxs)
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * test_frac)))

    n_visits_total = 0
    n_codes_total = 0
    unique_codes = set()
    for i in idxs:
        sample = dataset[i]
        visits = sample["visits"].tolist()
        n_visits_total += len(visits)
        for visit in visits:
            for code_idx in visit:
                code = index_to_code.get(int(code_idx))
                if code in (None, "<pad>", "<unk>"):
                    continue
                n_codes_total += 1
                unique_codes.add(code)

    n = len(idxs)
    return {
        "n_total": n,
        "n_train": n - n_test,
        "n_test": n_test,
        "avg_visits_per_patient": round(n_visits_total / n, 3) if n else 0.0,
        "avg_codes_per_visit": (
            round(n_codes_total / n_visits_total, 3) if n_visits_total else 0.0
        ),
        "n_unique_codes": len(unique_codes),
    }


def main():
    args = _build_arg_parser().parse_args()
    print(f"Loading eICU from {args.eicu_root} (full, dev=False)...")
    base = eICUDataset(root=args.eicu_root, tables=["diagnosis"], dev=False)
    sample_dataset = base.set_task(EHRGenerationEICU(min_visits=MIN_VISITS))
    index_to_code = {
        v: k for k, v in sample_dataset.input_processors["visits"].code_vocab.items()
    }
    total_samples = len(sample_dataset)
    print(f"Total samples: {total_samples}  "
          f"code vocab: {sample_dataset.input_processors['visits'].vocab_size()}")

    field_index = build_field_index(
        sample_dataset,
        cache_path=args.field_index_cache or None,
        eicu_root=args.eicu_root,
        rebuild=args.rebuild_index,
    )
    n_eligible = sum(1 for v in field_index.values()
                     if len(v) >= args.min_hospital_samples)
    print(f"Hospitals: {len(field_index)} total, {n_eligible} eligible "
          f"(>= {args.min_hospital_samples} samples)")

    chosen = stratified_draw(
        field_index, args.n_clients, args.min_hospital_samples, args.seed,
    )

    # Per-hospital EDA. split_seed = args.seed + position, matching the
    # enumerate(client_seed) convention in ehr_eicu.partition_by_hospital.
    hospitals: List[dict] = []
    for pos, rec in enumerate(chosen):
        eda = hospital_eda(
            sample_dataset, rec["_idxs"], index_to_code,
            test_frac=args.test_frac, split_seed=args.seed + pos,
        )
        hospitals.append({
            "hospital_id": rec["hospital_id"],
            "size_bin": rec["size_bin"],
            **eda,
        })

    cohort_train = sum(h["n_train"] for h in hospitals)
    cohort_total = sum(h["n_total"] for h in hospitals)
    cohort_test = sum(h["n_test"] for h in hospitals)

    manifest = {
        "meta": {
            "selection": "stratified",
            "seed": args.seed,
            "n_clients": args.n_clients,
            "min_hospital_samples": args.min_hospital_samples,
            "test_frac": args.test_frac,
            "eicu_root": args.eicu_root,
            "n_hospitals_total": len(field_index),
            "n_hospitals_eligible": n_eligible,
            "dataset_total_samples": total_samples,
            "cohort_total": cohort_total,
            "cohort_train": cohort_train,
            "cohort_test": cohort_test,
            "calibration_sec_per_sample_epoch": SEC_PER_SAMPLE_EPOCH,
        },
        # Ordered (largest size-bin first); ehr_eicu.py MUST consume in this order
        # so per-hospital train/test seeds line up.
        "hospitals": hospitals,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    # Report.
    print(f"\nFrozen cohort ({args.n_clients} hospitals, seed={args.seed}) "
          f"-> {args.out}")
    hdr = f"  {'bin':>3}  {'hospital':>10}  {'n_train':>8}  {'visits/pt':>9}  " \
          f"{'codes/visit':>11}  {'uniq_codes':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for h in hospitals:
        print(f"  {h['size_bin']:>3}  {h['hospital_id']:>10}  {h['n_train']:>8}  "
              f"{h['avg_visits_per_patient']:>9}  {h['avg_codes_per_visit']:>11}  "
              f"{h['n_unique_codes']:>10}")
    print(f"\n  cohort train samples: {cohort_train}  (total {cohort_total}, "
          f"test {cohort_test})")

    # Suggested n_rounds for a training budget, from the calibration.
    budget_h = 9.5
    print(f"\n  Suggested n_rounds for ~{budget_h}h training "
          f"(sec/sample-epoch={SEC_PER_SAMPLE_EPOCH}):")
    for E in (2, 5, 10):
        h_per_round = cohort_train * E * SEC_PER_SAMPLE_EPOCH / 3600.0
        rounds = max(1, int(budget_h / h_per_round)) if h_per_round else 0
        print(f"    local_epochs={E:>2}:  ~{h_per_round:.2f} h/round  "
              f"->  n_rounds ~ {rounds}")


if __name__ == "__main__":
    main()


# --- how to run (compute node, NOT login) ----------------------------------
#   srun --account=bgyw-delta-gpu --partition=cpu --time=00:30:00 \
#        --mem=32g --cpus-per-task=8 --pty \
#        python examples/fedpyhealth/select_cohort.py
#   # or wrap in a short SLURM batch script. Writes examples/fedpyhealth/
#   # cohorts/standard_8.json; commit that manifest so every baseline reuses it.
