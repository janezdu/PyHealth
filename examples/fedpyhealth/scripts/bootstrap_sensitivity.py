"""Report whether --n-bootstraps changes the ANSWER, not just the number.

Reads the JSONs written by bootstrap_sensitivity.sh and prints, per metric:
how far the n=5 estimate wanders across seeds, and whether the arm ranking is
stable. Ranking is what matters -- every claim in this project is a comparison,
so an estimator that moves all arms together is harmless while one that
reorders them is not.

Usage:
    python examples/fedpyhealth/scripts/bootstrap_sensitivity.py <dir>
"""

import json
import statistics as st
import sys
from pathlib import Path

METRICS = {
    "Pearson (all)": "PrevVal_All_Prevalence_Pearson",
    "R2 (all)": "PrevVal_All_Prevalence_R2",
    "Pearson (rare)": "PrevVal_Rare_Prevalence_Pearson",
}


def arm_medians(path, key):
    """Median across hospitals, which is how every figure summarises an arm."""
    runs = json.load(open(path))["runs"]
    return {r: st.median(v[key][0] for v in runs.values()) for r, v in runs.items()}


def main(d):
    d = Path(d)
    n5 = sorted(d.glob("n5_seed*.json"))
    n200 = sorted(d.glob("n200_seed*.json"))
    if not n5 or not n200:
        sys.exit(f"no results in {d}; run bootstrap_sensitivity.sh first")

    for label, key in METRICS.items():
        five = [arm_medians(p, key) for p in n5]
        two = [arm_medians(p, key) for p in n200]
        arms = list(five[0])
        print(f"=== {label}")
        print(f"{'arm':16s} {'n=5 across seeds':>26s} {'spread':>8s} {'n=200':>9s}")
        for a in sorted(arms, key=lambda a: -two[0][a]):
            vals = [f[a] for f in five]
            print(f"{a:16s} {min(vals):8.4f} .. {max(vals):8.4f}"
                  f" {'':6s} {max(vals)-min(vals):8.4f} {two[0][a]:9.4f}")

        # The decision-relevant part: does the ORDER survive?
        orders = [tuple(sorted(arms, key=lambda a: -f[a])) for f in five]
        o200 = tuple(sorted(arms, key=lambda a: -two[0][a]))
        distinct = set(orders)
        print(f"  distinct rankings across {len(five)} n=5 seeds: {len(distinct)}")
        print(f"  any n=5 ranking equals the n=200 ranking: "
              f"{o200 in distinct}")
        if len(two) > 1:
            o200b = tuple(sorted(arms, key=lambda a: -two[1][a]))
            print(f"  n=200 ranking stable across its own two seeds: "
                  f"{o200 == o200b}")
        print(f"  VERDICT: n=5 is "
              f"{'ADEQUATE' if len(distinct) == 1 and o200 in distinct else 'NOT ADEQUATE'}"
              f" for ranking on this metric")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "_outputs/results/bootstrap_sensitivity")
