#!/usr/bin/env python3
"""Aggregate per-run ``results.json`` files into a sortable leaderboard.

Reads every ``results.json`` ``ehr_eicu.py`` wrote under ``_outputs/results/``
(or the files/globs you pass), pulls each run's macro-averaged metrics plus its
key knobs, and prints one row per run sorted by a chosen metric -- the quick way
to read a sweep's outcome without re-parsing SLURM logs. Optionally dumps the
same table to CSV.

Usage (from the repo root)::

    .venv/bin/python examples/fedpyhealth/sweeps/aggregate_results.py
    .venv/bin/python examples/fedpyhealth/sweeps/aggregate_results.py --sort Prevalence_R2 --desc
    .venv/bin/python examples/fedpyhealth/sweeps/aggregate_results.py --csv leaderboard.csv
    .venv/bin/python examples/fedpyhealth/sweeps/aggregate_results.py '_outputs/results/fedavg_*.json'
"""
import argparse
import csv
import glob
import json
import os
import sys

KNOB_COLS = ["regime", "weighting", "lr", "local_epochs", "n_rounds", "ft_epochs"]


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
        rows.append(row)
    return rows, metric_cols


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                    help="results.json files/globs (default: _outputs/results/*.json)")
    ap.add_argument("--sort", default="Prevalence_R2",
                    help="macro-avg metric (or knob column) to sort by")
    ap.add_argument("--desc", action="store_true",
                    help="sort descending (use for higher-is-better metrics)")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    args = ap.parse_args(argv)

    patterns = args.paths or ["_outputs/results/*.json"]
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


if __name__ == "__main__":
    main()
