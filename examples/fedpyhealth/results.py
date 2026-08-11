#!/usr/bin/env python3
"""Aggregate per-run ``results.json`` files into a sortable leaderboard.

Reads every run file ``train.py`` wrote under ``_outputs/results/runs/``
(or the files/globs you pass), pulls each run's macro-averaged metrics plus its
key knobs, and prints one row per run sorted by a chosen metric -- the quick way
to read a sweep's outcome without re-parsing SLURM logs. Optionally dumps the
same table to CSV.

Usage (from the repo root)::

    .venv/bin/python examples/fedpyhealth/results.py
    .venv/bin/python examples/fedpyhealth/results.py --sort Prevalence_R2 --desc
    .venv/bin/python examples/fedpyhealth/results.py --csv leaderboard.csv
    .venv/bin/python examples/fedpyhealth/results.py '_outputs/results/runs/fedavg_*.json'
"""
import argparse
import csv
import glob
import json
import os
import sys

RUNS_GLOB = "_outputs/results/runs/*.json"
TESTS_DIR = "_outputs/results/tests"

KNOB_COLS = ["regime", "weighting", "lr", "local_epochs", "n_rounds", "ft_epochs",
             "num_synth", "cohort_cache"]


def load_rows(paths):
    rows, metric_cols = [], []
    for p in sorted(set(paths)):
        with open(p) as f:
            d = json.load(f)
        knobs = d.get("key_knobs", {}) or {}
        row = {"run_name": d.get("run_name", os.path.basename(p))}
        for c in KNOB_COLS:
            row[c] = d.get(c, knobs.get(c))
        for name, stats in (d.get("macro_avg_metrics") or {}).items():
            row[name] = stats.get("mean")
            if name not in metric_cols:
                metric_cols.append(name)
        # Pooled (non-per-hospital) numbers, e.g. Test 2 ML efficacy, land in
        # global_metrics. Prefixed so they never collide with a macro average.
        for name, stats in (d.get("global_metrics") or {}).items():
            col = f"g_{name}"
            row[col] = stats.get("mean")
            if col not in metric_cols:
                metric_cols.append(col)
        rows.append(row)
    return rows, metric_cols


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


# --------------------------------------------------------------------------- #
# The three experiment tables (joined from tests/)                             #
# --------------------------------------------------------------------------- #
# Reference arms are bounds, not methods: real_pooled is the ceiling (a
# classifier trained on all eight hospitals' real data), real_local the "each
# site alone, with REAL data" bar that federation has to beat, and prior the
# floor (predict base rates). Printed below a rule so they are never read as
# competing methods.
REFERENCE_ARMS = ("real_pooled", "real_pooled_budgeted", "real_local", "prior")

# Support bands in increasing order, matching test2_rare_efficacy.SUPPORT_BANDS.
# Sorting the names alphabetically puts "5_9" last, which reads as if the tail
# improved -- the exact misreading these bands exist to prevent.
BAND_ORDER = ("2_4", "5_9", "10_19", "20_49", "50_plus")


def _load_test(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean_std(values):
    vals = [v for v in values if v is not None and v == v]     # drop None/NaN
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var ** 0.5


def test1_by_regime(t1):
    """``{regime: {metric: (mean, spread)}}`` -- spread is ACROSS HOSPITALS.

    Hospital variance is the quantity of interest in a deliberately
    heterogeneous cohort, and it is much larger than the bootstrap error the
    per-hospital numbers already carry, so the +/- here is the between-hospital
    standard deviation rather than a confidence interval on any one of them.
    """
    out = {}
    for regime, per_hospital in (t1 or {}).get("runs", {}).items():
        scores = {}
        for label, key in (("t1_r2_all", "PrevVal_All_Prevalence_R2"),
                           ("t1_r2_rare", "PrevVal_Rare_Prevalence_R2")):
            vals = [m.get(key, [None])[0] for m in per_hospital.values()]
            scores[label] = _mean_std(vals)
        out[regime] = scores
    return out


def test2_by_family(t2):
    """``{family: {"t2_ap_global_rare": (mean, None), ...}}`` from families."""
    out = {}
    for family, entry in (t2 or {}).get("families", {}).items():
        out[family] = {
            "t2_ap_global_rare": (entry.get("global_rare_ap_macro", {}).get("mean"),
                                  None),
            "t2_ap_all_rare": (entry.get("overall_ap_macro", {}).get("mean"), None),
            "n_degenerate": entry.get("n_degenerate", 0),
        }
    return out


def test2_by_hospital(t2):
    """``{family: {hospital: ap_global_rare}}`` from the per-classifier arms."""
    out = {}
    for entry in (t2 or {}).get("arms", {}).values():
        family, hospital = entry.get("family"), entry.get("hospital")
        if not family or not hospital or entry.get("degenerate"):
            continue
        ap = (entry.get("global_rare") or {}).get("ap_macro")
        out.setdefault(family, {})[str(hospital)] = ap
    return out


def _print_table(header, rows, rule_before=None):
    cols = list(header)
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(c)) for i, c in enumerate(cols)]
    print("  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 2 * len(cols)))
    for r in rows:
        if rule_before is not None and r[0] == rule_before:
            print("-" * (sum(widths) + 2 * len(cols)))
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def _pm(pair):
    """Format a ``(mean, spread)`` pair; spread omitted when there is none."""
    if not pair or pair[0] is None:
        return "-"
    mean, sd = pair
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ±{sd:.3f}"


def print_experiment_tables(t1, t2, hospital_order):
    """Headline table, per-hospital-by-size table, and support bands."""
    t1r, t2f = test1_by_regime(t1), test2_by_family(t2)
    regimes = [r for r in t2f if r not in REFERENCE_ARMS] or list(t1r)

    print("\n=== Headline: prevalence fidelity and rare-code ML efficacy ===")
    print("    (+/- is the spread ACROSS the 8 hospitals, not bootstrap error)")
    header = ["method", "T1 R2(all)", "T1 R2(rare)",
              "T2 AP(global-rare)", "T2 AP(all rare)"]
    rows = []
    for name in list(regimes) + [a for a in REFERENCE_ARMS if a in t2f]:
        s1, s2 = t1r.get(name, {}), t2f.get(name, {})
        rows.append([name, _pm(s1.get("t1_r2_all")), _pm(s1.get("t1_r2_rare")),
                     _pm(s2.get("t2_ap_global_rare")),
                     _pm(s2.get("t2_ap_all_rare"))])
    first_ref = next((a for a in REFERENCE_ARMS if a in t2f), None)
    _print_table(header, rows, rule_before=first_ref)
    degenerate = {n: t2f[n]["n_degenerate"] for n in t2f
                  if t2f[n].get("n_degenerate")}
    if degenerate:
        print(f"  NOTE: degenerate classifiers (no scored rare code in their "
              f"training data): {degenerate}")

    # --- the table the size-banded cohort exists to produce -----------------
    per_hosp = test2_by_hospital(t2)
    if per_hosp and hospital_order:
        print("\n=== Test 2 by hospital, largest to smallest ===")
        print("    A pooled average hides this. If federation does what it is "
              "supposed to,\n    the gap over `local` widens as hospitals get "
              "smaller.")
        fams = [f for f in ("local", "fedavg", "fedavg_ft", "centralized")
                if f in per_hosp]
        header = ["hospital", "n"] + fams
        if "local" in fams and "fedavg" in fams:
            header.append("fedavg-local")
        rows = []
        for hid, n in hospital_order:
            vals = [per_hosp.get(f, {}).get(hid) for f in fams]
            row = [hid, n] + [f"{v:.3f}" if v is not None else "-" for v in vals]
            if "local" in fams and "fedavg" in fams:
                lo = per_hosp.get("local", {}).get(hid)
                fa = per_hosp.get("fedavg", {}).get(hid)
                row.append(f"{fa - lo:+.3f}" if lo is not None and fa is not None
                           else "-")
            rows.append(row)
        _print_table(header, rows)

    # --- where in the tail each method breaks down --------------------------
    bands = {f: e.get("bands", {}) for f, e in (t2 or {}).get("families", {}).items()}
    seen = {b for e in bands.values() for b in e}
    band_names = ([b for b in BAND_ORDER if b in seen]
                  + sorted(seen - set(BAND_ORDER)))
    if band_names:
        print("\n=== Test 2 AP by validation support (codes with N positives) ===")
        header = ["method"] + band_names
        rows = [[f] + [f"{bands[f][b]['mean']:.3f}"
                       if bands[f].get(b, {}).get("mean") == bands[f].get(b, {}).get("mean")
                       else "-" for b in band_names]
                for f in bands]
        _print_table(header, rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                    help=f"run files/globs (default: {RUNS_GLOB})")
    ap.add_argument("--sort", default="Prevalence_R2",
                    help="macro-avg metric (or knob column) to sort by")
    ap.add_argument("--desc", action="store_true",
                    help="sort descending (use for higher-is-better metrics)")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    ap.add_argument("--tests-dir", default=TESTS_DIR,
                    help="where test1/test2 wrote their scores")
    ap.add_argument("--no-tests", action="store_true",
                    help="print only the per-run table, skip the joined tables")
    ap.add_argument("--summary", default="_outputs/results/summary.json",
                    help="write the joined tables here ('' to skip)")
    args = ap.parse_args(argv)

    patterns = args.paths or [RUNS_GLOB]
    paths = [m for pat in patterns for m in glob.glob(pat)]
    if not paths:
        print(f"no results.json matched {patterns}", file=sys.stderr)
        sys.exit(1)

    rows, metric_cols = load_rows(paths)
    cols = ["run_name"] + KNOB_COLS + metric_cols
    key = args.sort
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key) if r.get(key) is not None else 0),
              reverse=args.desc)

    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * len(cols)))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))
    print(f"\n{len(rows)} run(s), sorted by {key}{' desc' if args.desc else ' asc'}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print(f"wrote {args.csv}")

    if args.no_tests:
        return
    t1 = _load_test(os.path.join(args.tests_dir, "test1_prevalence.json"))
    t2 = _load_test(os.path.join(args.tests_dir, "test2_rare_efficacy.json"))
    if not t1 and not t2:
        print(f"\n(no test scores in {args.tests_dir} yet -- run main.py test1 "
              "/ test2 to fill the experiment tables)")
        return

    # Hospital order comes from the cohort cache, so the per-hospital table is
    # sorted by real cohort size rather than by whatever the tests emitted.
    hospital_order = []
    cohort_cache = (t1 or {}).get("cohort_cache") or (t2 or {}).get("cohort_cache")
    manifest_path = os.path.join(cohort_cache or "", "manifest.json")
    if cohort_cache and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        hospital_order = sorted(
            ((str(hid), h["n_total"])
             for hid, h in manifest["per_hospital"].items()),
            key=lambda kv: kv[1], reverse=True,
        )

    print_experiment_tables(t1, t2, hospital_order)

    if args.summary:
        summary = {
            "kind": "summary",
            "cohort_cache": cohort_cache,
            "runs": [r["run_name"] for r in rows],
            "headline": {
                "test1": test1_by_regime(t1),
                "test2": test2_by_family(t2),
            },
            "test2_by_hospital": test2_by_hospital(t2),
            "hospital_order": hospital_order,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
