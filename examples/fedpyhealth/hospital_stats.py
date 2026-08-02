"""Per-hospital EDA for the eICU federated cohort: sizes, code histograms, rare codes.

Answers, for the eICU task samples that the federated runs actually train on
(``eICUDataset`` + ``EHRGenerationEICU``, so one **record = one patient/sample**
with >= 1 unit-stay "visit"):

1. How many hospitals are there?
2. What is the distribution of records per hospital?
3. Which are the ``--n-smallest`` (default 8) smallest hospitals, and how many
   records do they have?  Two answers are printed: the *true* smallest (no floor,
   often uselessly tiny) and the smallest that clear ``--min-hospital-samples``
   (default 50, the same floor the federated pipeline uses).  The floored set is
   what the code analysis below runs on.
4. Within those hospitals, a histogram/count of the ICD-9 codes each one has.
5. Per hospital, its "rare codes" -- codes present in at most
   ``--rare-max-hospitals`` (default half, i.e. 4 of 8) of that 8-hospital set.

The expensive part is a single streaming pass over every sample (``dataset[i]``
deserializes from the on-disk cache).  That pass is cached to JSON, so re-running
the report with a different threshold is instant::

    # smoke test on a small subset first (minutes; numbers NOT representative)
    python examples/fedpyhealth/hospital_stats.py --dev --eicu-root /path/to/eicu
    # the real thing, once (slow: builds/walks every sample)
    python examples/fedpyhealth/hospital_stats.py --eicu-root /path/to/eicu
    # again with a different rare-code rule (seconds, no eICU load)
    python examples/fedpyhealth/hospital_stats.py --from-cache --rare-max-hospitals 2

This is a CPU-only job (no GPU, no training).  Do NOT run it on the login node --
see the sample commands at the bottom of this file, or use the companion
``run_hospital_stats.sh`` launcher.

Outputs (under ``--out-dir``, gitignored):
  hospital_stats_cache.json  full per-hospital counts (the reusable pass cache)
  hospital_sizes.csv         every hospital: n_records, n_visits, n_codes, n_unique_codes
  smallest_code_counts.csv   the analysed hospitals: one row per (hospital, code)
  rare_codes.csv             the analysed hospitals: one row per (hospital, rare code)
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

# Kept in sync with ehr_eicu.py / select_cohort.py so the numbers describe the
# same dataset the federated runs train on.
EICU_ROOT = "/work/hdd/bgyw/janezdu/data/eicu/eicu-crd/2.0"
MIN_VISITS = 1

# Size buckets for the "distribution of records per hospital" histogram.
SIZE_BINS: List[Tuple[int, int]] = [
    (1, 9), (10, 24), (25, 49), (50, 99), (100, 249),
    (250, 499), (500, 999), (1000, 2499), (2500, 10 ** 9),
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--eicu-root", default=EICU_ROOT, help="path to eICU CRD root")
    p.add_argument("--n-smallest", type=int, default=8,
                   help="how many small hospitals to analyse in detail")
    p.add_argument("--min-hospital-samples", type=int, default=50,
                   help="floor for the analysed set; hospitals below it are "
                        "reported as a warning but not analysed")
    p.add_argument("--rare-max-hospitals", type=int, default=None,
                   help="a code is 'rare' if present in <= this many of the "
                        "analysed hospitals (default: half of --n-smallest)")
    p.add_argument("--top-codes", type=int, default=20,
                   help="how many top codes / rare codes to print per hospital "
                        "(the CSVs always contain the full lists)")
    p.add_argument("--out-dir", default="_outputs/eda",
                   help="directory for the JSON cache and CSV tables")
    p.add_argument("--dev", action="store_true",
                   help="load eICU with dev=True (small subset). Use for a fast "
                        "smoke test only -- the resulting hospital counts are "
                        "NOT representative of the full dataset")
    p.add_argument("--from-cache", action="store_true",
                   help="skip the eICU load and re-render the report from the "
                        "JSON cache in --out-dir")
    p.add_argument("--field-index-cache",
                   default="examples/fedpyhealth/cohorts/.field_index.json",
                   help="also write select_cohort.py's hospital->indices cache "
                        "during the pass (free); set empty to skip")
    return p


# --------------------------------------------------------------------------- #
# Stage 1: the expensive pass                                                  #
# --------------------------------------------------------------------------- #
def collect_hospital_stats(eicu_root: str, dev: bool = False) -> dict:
    """Walk every task sample once and tally per-hospital records and codes.

    One record = one sample = one patient kept by ``EHRGenerationEICU``.  Codes
    are ICD-9 as stored in eICU's ``icd9code`` (first token of a comma-separated
    ICD9/ICD10 pair), recovered from the fitted vocabulary; ``<pad>``/``<unk>``
    are skipped.

    Args:
        eicu_root: Path to the eICU CRD 2.0 root (the folder with the CSVs).
        dev: Load a small development subset instead of the full dataset. Fast,
            but the hospital counts are then a biased subset -- smoke test only.

    Returns:
        A JSON-serialisable dict with a ``meta`` block and a ``hospitals`` map
        ``hospital_id -> {n_records, n_visits, n_code_occurrences,
        code_occurrences: {code: n}, code_patients: {code: n}}``, plus
        ``field_index`` (hospital -> sample indices) for reuse by
        ``select_cohort.py``.
    """
    from pyhealth.datasets import eICUDataset
    from pyhealth.tasks import EHRGenerationEICU

    if dev:
        print("!! --dev: loading a SUBSET of eICU. Hospital counts, the "
              "'smallest' set\n!! and rare codes below are NOT representative "
              "of the full dataset.", flush=True)
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

    hospitals: Dict[str, dict] = {}
    field_index: Dict[str, List[int]] = {}
    for i in range(n):
        sample = samples[i]
        hid = str(sample.get("hospital_id", "NA"))
        field_index.setdefault(hid, []).append(i)
        rec = hospitals.setdefault(hid, {
            "n_records": 0, "n_visits": 0, "n_code_occurrences": 0,
            "code_occurrences": {}, "code_patients": {},
        })
        rec["n_records"] += 1

        occ = rec["code_occurrences"]
        seen_in_patient = set()
        visits = sample["visits"].tolist()
        rec["n_visits"] += len(visits)
        for visit in visits:
            for code_idx in visit:
                code = index_to_code.get(int(code_idx))
                if code in (None, "<pad>", "<unk>"):
                    continue
                occ[code] = occ.get(code, 0) + 1
                rec["n_code_occurrences"] += 1
                seen_in_patient.add(code)
        pat = rec["code_patients"]
        for code in seen_in_patient:
            pat[code] = pat.get(code, 0) + 1

        if (i + 1) % 20000 == 0:
            print(f"  ...{i + 1}/{n} samples", flush=True)

    return {
        "meta": {
            "eicu_root": eicu_root,
            "dev": dev,
            "min_visits": MIN_VISITS,
            "dataset_total_samples": n,
            "code_vocab_size": vocab_size,
            "record_unit": "task sample = one patient with >=1 unit-stay visit",
            "code_field": "icd9code (as stored, first token)",
        },
        "hospitals": hospitals,
        "field_index": field_index,
    }


def count_hospitals_in_raw_csv(eicu_root: str) -> int:
    """Count hospital ids listed in eICU's ``hospital.csv`` (context only).

    Some listed hospitals contribute no qualifying patients, so this is >= the
    number of hospitals that appear in the task samples.

    Args:
        eicu_root: Path to the eICU CRD 2.0 root.

    Returns:
        The number of rows in ``hospital.csv``, or -1 if the file is unreadable.
    """
    path = os.path.join(eicu_root, "hospital.csv")
    try:
        with open(path, newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except OSError:
        return -1


# --------------------------------------------------------------------------- #
# Stage 2: reporting (pure functions over the cached counts)                    #
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: List[int], q: float) -> float:
    """Return the ``q``-quantile (0-1) of an ascending list, linear interpolation."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def print_size_distribution(sizes: Dict[str, int]) -> None:
    """Print summary stats and a text histogram of records per hospital.

    Args:
        sizes: Map of hospital id -> number of records.
    """
    vals = sorted(sizes.values())
    total = sum(vals)
    print("\n" + "=" * 78)
    print("2. DISTRIBUTION OF RECORDS PER HOSPITAL")
    print("=" * 78)
    print(f"  hospitals with >=1 record : {len(vals)}")
    print(f"  total records             : {total}")
    print(f"  mean                      : {total / len(vals):.1f}")
    print(f"  min / p10 / q1            : {vals[0]} / {_percentile(vals, .10):.0f}"
          f" / {_percentile(vals, .25):.0f}")
    print(f"  median                    : {_percentile(vals, .50):.0f}")
    print(f"  q3 / p90 / max            : {_percentile(vals, .75):.0f}"
          f" / {_percentile(vals, .90):.0f} / {vals[-1]}")

    print(f"\n  {'records/hospital':>18}  {'hospitals':>9}  {'records':>9}  share")
    print("  " + "-" * 58)
    for lo, hi in SIZE_BINS:
        members = [v for v in vals if lo <= v <= hi]
        if not members:
            continue
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        bar = "#" * max(1, round(40 * len(members) / len(vals)))
        print(f"  {label:>18}  {len(members):>9}  {sum(members):>9}  {bar}")

    print(f"\n  5 largest hospitals: " + ", ".join(
        f"{h}({n})" for h, n in
        sorted(sizes.items(), key=lambda kv: -kv[1])[:5]))


def print_smallest(sizes: Dict[str, int], n_smallest: int, floor: int
                   ) -> List[str]:
    """Print the true smallest hospitals and the floored analysis set.

    Args:
        sizes: Map of hospital id -> number of records.
        n_smallest: How many hospitals to list / analyse.
        floor: Minimum records for a hospital to enter the analysis set.

    Returns:
        The hospital ids of the analysis set (smallest ``n_smallest`` at or above
        ``floor``), ascending by size.
    """
    ascending = sorted(sizes.items(), key=lambda kv: (kv[1], kv[0]))
    true_smallest = ascending[:n_smallest]
    eligible = [(h, v) for h, v in ascending if v >= floor]

    print("\n" + "=" * 78)
    print(f"3. THE {n_smallest} SMALLEST HOSPITALS")
    print("=" * 78)
    print(f"\n  !! WARNING -- the TRUE {n_smallest} smallest hospitals (no floor) "
          f"are tiny:\n")
    print(f"  {'hospital':>10}  {'records':>8}")
    print("  " + "-" * 20)
    for h, v in true_smallest:
        print(f"  {h:>10}  {v:>8}")
    print(f"\n  Code histograms and rare-code lists on hospitals this small are "
          f"\n  dominated by noise, and the federated pipeline excludes them "
          f"(min_hospital_samples).")
    print(f"  The analysis below therefore uses the {n_smallest} smallest "
          f"hospitals with >= {floor} records.")

    if len(eligible) < n_smallest:
        raise SystemExit(
            f"\nERROR: only {len(eligible)} hospitals have >= {floor} records; "
            f"need {n_smallest}. Lower --min-hospital-samples."
        )
    chosen = eligible[:n_smallest]
    print(f"\n  ANALYSIS SET -- {n_smallest} smallest hospitals with >= {floor} "
          f"records:\n")
    print(f"  {'hospital':>10}  {'records':>8}")
    print("  " + "-" * 20)
    for h, v in chosen:
        print(f"  {h:>10}  {v:>8}")
    print(f"  {'TOTAL':>10}  {sum(v for _, v in chosen):>8}")
    n_below = sum(1 for _, v in ascending if v < floor)
    print(f"\n  ({n_below} of {len(ascending)} hospitals fall below the "
          f"{floor}-record floor and are excluded.)")
    return [h for h, _ in chosen]


def print_code_histograms(stats: dict, chosen: List[str], top_codes: int) -> None:
    """Print a per-hospital ICD-9 code histogram for the analysed hospitals.

    Args:
        stats: The cached stats dict from :func:`collect_hospital_stats`.
        chosen: Hospital ids to report on.
        top_codes: How many top codes to print per hospital.
    """
    print("\n" + "=" * 78)
    print(f"4. ICD-9 CODE HISTOGRAM PER HOSPITAL (top {top_codes}; "
          f"full lists in smallest_code_counts.csv)")
    print("=" * 78)
    for hid in chosen:
        rec = stats["hospitals"][hid]
        occ = rec["code_occurrences"]
        pat = rec["code_patients"]
        n_rec = rec["n_records"]
        print(f"\n  hospital {hid}:  {n_rec} records, {rec['n_visits']} visits, "
              f"{rec['n_code_occurrences']} code occurrences, "
              f"{len(occ)} unique codes")
        print(f"    {'code':>10}  {'count':>7}  {'pts':>6}  {'%pts':>6}")
        print("    " + "-" * 34)
        for code, cnt in sorted(occ.items(), key=lambda kv: (-kv[1], kv[0]))[:top_codes]:
            n_pat = pat.get(code, 0)
            print(f"    {code:>10}  {cnt:>7}  {n_pat:>6}  "
                  f"{100.0 * n_pat / n_rec:>5.1f}%")


def compute_rare_codes(stats: dict, chosen: List[str], max_hospitals: int
                       ) -> Tuple[Dict[str, int], Dict[str, List[tuple]]]:
    """Find codes present in at most ``max_hospitals`` of the analysed hospitals.

    Args:
        stats: The cached stats dict from :func:`collect_hospital_stats`.
        chosen: Hospital ids forming the comparison set.
        max_hospitals: A code is rare if it appears in <= this many of them.

    Returns:
        ``(presence, rare_by_hospital)`` where ``presence`` maps code -> number of
        analysed hospitals containing it, and ``rare_by_hospital`` maps hospital
        id -> list of ``(code, n_occurrences, n_patients, n_hospitals_present)``
        sorted by occurrence descending.
    """
    presence: Dict[str, int] = {}
    for hid in chosen:
        for code in stats["hospitals"][hid]["code_occurrences"]:
            presence[code] = presence.get(code, 0) + 1

    rare_by_hospital: Dict[str, List[tuple]] = {}
    for hid in chosen:
        rec = stats["hospitals"][hid]
        rows = [
            (code, cnt, rec["code_patients"].get(code, 0), presence[code])
            for code, cnt in rec["code_occurrences"].items()
            if presence[code] <= max_hospitals
        ]
        rows.sort(key=lambda r: (-r[1], r[0]))
        rare_by_hospital[hid] = rows
    return presence, rare_by_hospital


def print_rare_codes(stats: dict, chosen: List[str], presence: Dict[str, int],
                     rare_by_hospital: Dict[str, List[tuple]],
                     max_hospitals: int, top_codes: int) -> None:
    """Print the rare-code summary and per-hospital rare-code lists.

    Args:
        stats: The cached stats dict.
        chosen: Hospital ids forming the comparison set.
        presence: Code -> number of analysed hospitals containing it.
        rare_by_hospital: Output of :func:`compute_rare_codes`.
        max_hospitals: The rarity threshold used.
        top_codes: How many rare codes to print per hospital.
    """
    n_h = len(chosen)
    print("\n" + "=" * 78)
    print(f"5. RARE CODES -- present in <= {max_hospitals} of the {n_h} analysed "
          f"hospitals")
    print("=" * 78)
    print(f"\n  Codes seen anywhere in the {n_h} hospitals: {len(presence)}")
    print(f"  {'in # hospitals':>15}  {'codes':>7}")
    print("  " + "-" * 25)
    for k in range(1, n_h + 1):
        n_codes = sum(1 for v in presence.values() if v == k)
        mark = "  <- rare" if k <= max_hospitals else ""
        print(f"  {k:>15}  {n_codes:>7}{mark}")
    n_rare = sum(1 for v in presence.values() if v <= max_hospitals)
    print(f"\n  rare codes (<= {max_hospitals} hospitals): {n_rare} of "
          f"{len(presence)} ({100.0 * n_rare / max(1, len(presence)):.1f}%)")

    print(f"\n  Per hospital (top {top_codes} by occurrence; full lists in "
          f"rare_codes.csv):")
    for hid in chosen:
        rows = rare_by_hospital[hid]
        rec = stats["hospitals"][hid]
        n_uniq = len(rec["code_occurrences"])
        share = 100.0 * len(rows) / max(1, n_uniq)
        print(f"\n  hospital {hid}:  {len(rows)} rare of {n_uniq} unique codes "
              f"({share:.1f}%)")
        if not rows:
            print("    (none)")
            continue
        print(f"    {'code':>10}  {'count':>7}  {'pts':>6}  {'#hosp':>6}")
        print("    " + "-" * 34)
        for code, cnt, n_pat, n_hosp in rows[:top_codes]:
            print(f"    {code:>10}  {cnt:>7}  {n_pat:>6}  {n_hosp:>6}")


# --------------------------------------------------------------------------- #
# CSV writers                                                                  #
# --------------------------------------------------------------------------- #
def write_csvs(out_dir: str, stats: dict, chosen: List[str],
               rare_by_hospital: Dict[str, List[tuple]], floor: int) -> None:
    """Write the three CSV tables (sizes, code counts, rare codes).

    Args:
        out_dir: Directory to write into (created if missing).
        stats: The cached stats dict.
        chosen: Analysed hospital ids.
        rare_by_hospital: Output of :func:`compute_rare_codes`.
        floor: The record floor used, recorded in the sizes table.
    """
    os.makedirs(out_dir, exist_ok=True)

    sizes_path = os.path.join(out_dir, "hospital_sizes.csv")
    with open(sizes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hospital_id", "n_records", "n_visits", "n_code_occurrences",
                    "n_unique_codes", "meets_floor", "in_analysis_set"])
        for hid, rec in sorted(stats["hospitals"].items(),
                               key=lambda kv: -kv[1]["n_records"]):
            w.writerow([hid, rec["n_records"], rec["n_visits"],
                        rec["n_code_occurrences"], len(rec["code_occurrences"]),
                        int(rec["n_records"] >= floor), int(hid in chosen)])

    codes_path = os.path.join(out_dir, "smallest_code_counts.csv")
    with open(codes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hospital_id", "icd9code", "n_occurrences", "n_patients"])
        for hid in chosen:
            rec = stats["hospitals"][hid]
            for code, cnt in sorted(rec["code_occurrences"].items(),
                                    key=lambda kv: (-kv[1], kv[0])):
                w.writerow([hid, code, cnt, rec["code_patients"].get(code, 0)])

    rare_path = os.path.join(out_dir, "rare_codes.csv")
    with open(rare_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hospital_id", "icd9code", "n_occurrences", "n_patients",
                    "n_hospitals_present"])
        for hid in chosen:
            for code, cnt, n_pat, n_hosp in rare_by_hospital[hid]:
                w.writerow([hid, code, cnt, n_pat, n_hosp])

    print(f"\nWrote:\n  {sizes_path}\n  {codes_path}\n  {rare_path}")


def main():
    args = _build_arg_parser().parse_args()
    # dev results are a biased subset -- keep them in a separate cache file so
    # they can never be mistaken for (or overwrite) the full-dataset numbers.
    cache_name = ("hospital_stats_cache_dev.json" if args.dev
                  else "hospital_stats_cache.json")
    cache_path = os.path.join(args.out_dir, cache_name)

    if args.from_cache:
        if not os.path.exists(cache_path):
            raise SystemExit(
                f"--from-cache given but {cache_path} does not exist. Run once "
                f"without --from-cache first (needs a compute node)."
            )
        with open(cache_path) as f:
            stats = json.load(f)
        print(f"Loaded cached stats from {cache_path} "
              f"({len(stats['hospitals'])} hospitals, "
              f"{stats['meta']['dataset_total_samples']} samples)")
    else:
        stats = collect_hospital_stats(args.eicu_root, dev=args.dev)
        os.makedirs(args.out_dir, exist_ok=True)
        field_index = stats.pop("field_index")
        with open(cache_path, "w") as f:
            json.dump(stats, f)
        print(f"Cached per-hospital stats -> {cache_path}")
        # Free gift for select_cohort.py: the same hospital->indices map it
        # caches. Never write it from a --dev run: the indices would not match
        # the full dataset the cohort selection runs on.
        if args.field_index_cache and not args.dev:
            os.makedirs(os.path.dirname(args.field_index_cache) or ".",
                        exist_ok=True)
            with open(args.field_index_cache, "w") as f:
                json.dump({"n_samples": stats["meta"]["dataset_total_samples"],
                           "eicu_root": args.eicu_root,
                           "field_index": field_index}, f)
            print(f"Cached hospital index -> {args.field_index_cache}")

    sizes = {hid: rec["n_records"] for hid, rec in stats["hospitals"].items()}
    rare_max = (args.rare_max_hospitals if args.rare_max_hospitals is not None
                else args.n_smallest // 2)

    # 1. how many hospitals
    print("\n" + "=" * 78)
    print("1. DATASET SUMMARY")
    print("=" * 78)
    meta = stats["meta"]
    print(f"  eICU root                 : {meta['eicu_root']}")
    if meta.get("dev"):
        print("  !! dev=True SUBSET -- these numbers are not representative of "
              "the full dataset")
    print(f"  record unit               : {meta['record_unit']}")
    print(f"  code field                : {meta['code_field']}")
    print(f"  total records (samples)   : {meta['dataset_total_samples']}")
    print(f"  code vocabulary size      : {meta['code_vocab_size']}")
    print(f"  HOSPITALS with >=1 record : {len(sizes)}")
    n_listed = count_hospitals_in_raw_csv(meta["eicu_root"])
    if n_listed >= 0:
        print(f"  hospitals in hospital.csv : {n_listed}  (some contribute no "
              f"qualifying patients)")

    # 2-3. size distribution and the smallest hospitals
    print_size_distribution(sizes)
    chosen = print_smallest(sizes, args.n_smallest, args.min_hospital_samples)

    # 4-5. code histograms and rare codes within the analysed hospitals
    print_code_histograms(stats, chosen, args.top_codes)
    presence, rare_by_hospital = compute_rare_codes(stats, chosen, rare_max)
    print_rare_codes(stats, chosen, presence, rare_by_hospital, rare_max,
                     args.top_codes)

    write_csvs(args.out_dir, stats, chosen, rare_by_hospital,
               args.min_hospital_samples)


if __name__ == "__main__":
    sys.exit(main())


# --- how to run (compute node, NOT login) ----------------------------------
#   sbatch examples/fedpyhealth/run_hospital_stats.sh
#   # or interactively:
#   srun --account=bgyw-delta-gpu --partition=gpuA100x4-interactive \
#        --gpus-per-node=1 --time=01:00:00 --mem=64g --cpus-per-task=16 --pty \
#        bash -c "source .venv/bin/activate && \
#                 python examples/fedpyhealth/hospital_stats.py"
