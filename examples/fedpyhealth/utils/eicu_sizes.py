"""How many patients does each eICU hospital have?

Answers the question that has to be settled before any cohort can be designed:
which hospitals exist, how big are they, and how many sit in a given size band.
Counts **unique patients** (``patienthealthsystemstayid``), not unit stays --
a patient can have several, and the cohort is built per patient.

Reads only ``patient.csv``, so it is seconds rather than the minutes a full
cohort build takes.

.. warning::
   This is a **survey, not a sizing tool**. It counts
   ``patienthealthsystemstayid`` and applies none of the task's filters, so its
   numbers run higher than the post-task count a cohort is built from. To pick a
   size band, use ``hospital_sizes.py``, which counts ``uniquepid``, drops stays
   with no usable code and patients seen at more than one hospital, and can
   verify itself against an existing cohort manifest. Prints aggregate counts only; eICU hospital IDs are already
de-identified in the source, and no patient-level field is read beyond the stay
and hospital identifiers needed to count.

Usage
-----
::

    export EICU_ROOT=/path/to/eicu-crd/2.0
    python examples/fedpyhealth/utils/eicu_sizes.py
    python examples/fedpyhealth/utils/eicu_sizes.py --band 1500 --top 4
    python examples/fedpyhealth/utils/eicu_sizes.py --markdown > notes/eicu-hospitals.md
"""

import argparse
import os
import statistics as st
from typing import Dict, List

# Observed on hospital 420 (in the retired strat8_random): 3005 patients
# survived of 3876 in
# the raw table, once MIN_VISITS and the cross-hospital-patient drop applied.
# A rule of thumb for sizing a cohort, not a guarantee.
POST_TASK_SHRINK = 3005 / 3876


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eicu-root", default=os.environ.get("EICU_ROOT", ""),
                   help="folder holding patient.csv (default: $EICU_ROOT)")
    p.add_argument("--band", type=int, default=1500,
                   help="the big/small boundary, in patients (default: 1500)")
    p.add_argument("--top", type=int, default=20,
                   help="how many of the largest hospitals to list")
    p.add_argument("--markdown", action="store_true",
                   help="emit the full notes/eicu-hospitals.md document")
    return p


def hospital_sizes(eicu_root: str) -> Dict[int, int]:
    """``{hospital_id: n_unique_patients}``, largest first."""
    import polars as pl

    path = os.path.join(eicu_root, "patient.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"no patient.csv at {path!r}. Set EICU_ROOT to the folder holding "
            "it, or pass --eicu-root.")
    df = pl.read_csv(path,
                     columns=["patienthealthsystemstayid", "hospitalid"],
                     schema_overrides={"hospitalid": pl.Int64})
    s = (df.unique(["patienthealthsystemstayid", "hospitalid"])
           .group_by("hospitalid").len().sort("len", descending=True))
    return dict(zip(s["hospitalid"].to_list(), s["len"].to_list()))


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    sizes = hospital_sizes(args.eicu_root)
    vals: List[int] = sorted(sizes.values(), reverse=True)
    big = [v for v in vals if v >= args.band]

    print(f"eICU: {len(sizes)} hospitals, {sum(vals):,} unique patients")
    print(f"  largest {vals[0]:,}   median {int(st.median(vals)):,}   "
          f"smallest {vals[-1]:,}")
    print(f"  >= {args.band}: {len(big)} hospitals holding "
          f"{100 * sum(big) / sum(vals):.0f}% of all patients")
    print(f"  <  {args.band}: {len(vals) - len(big)} hospitals\n")

    print(f"largest {args.top}:")
    for hid, n in list(sizes.items())[:args.top]:
        print(f"  {hid:>4}  {n:>6,}  ~{int(n * POST_TASK_SHRINK):>6,} post-task")

    top = list(sizes.items())[:4]
    tot = sum(n for _, n in top)
    print(f"\ntop-4 total: {tot:,} raw, ~{int(tot * POST_TASK_SHRINK):,} post-task")
    print("  (hilo8_random, the active cohort, is 12,156 post-task in total)")
    print("\nNOTE draw_bands() samples uniformly at random within a band, by "
          "design, so\n     that results generalise beyond the biggest sites. "
          "Maximising cohort size\n     means passing --hospitals explicitly "
          "rather than --bands.")


if __name__ == "__main__":
    main()
