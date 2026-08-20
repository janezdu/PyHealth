"""Write a cohort's git record from a built cache.

The cache lives outside the repo (it is large, and it is derived from
credentialed data), so the only thing version control can hold is a description
of WHICH hospitals a cohort drew and how. That is what this writes:
``cohorts/<name>.config.json``.

These files are documentation, not input -- nothing reads them at run time. The
build path is ``scripts/run_cohort*.sh`` -> ``utils/cohort.py`` -> a cache
directory holding ``manifest.json``, and every downstream job reads that
manifest. The config exists so that a year from now the repo can still answer
"which hospitals was this?" without the cache being around.

Run it after any cohort build::

    python examples/fedpyhealth/utils/record_cohort.py --cache $FEDCOHORT_CACHE/hilo8_random

Aggregate counts only: hospital ids (already de-identified in eICU) and fold
sizes. No patient-level field is read.
"""

import argparse
import json
import os
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT_DIR = os.path.join(os.path.dirname(HERE), "cohorts")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True,
                   help="a cohort cache directory holding manifest.json")
    p.add_argument("--out", default=None,
                   help="output path (default: cohorts/<cohort_name>.config.json)")
    p.add_argument("--description", default=None,
                   help="one-line description for the record")
    p.add_argument("--status", default="active",
                   help="active | superseded | abandoned (default: active)")
    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)
    mpath = os.path.join(args.cache, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"no manifest.json in {args.cache!r}; build the cohort first")
    m = json.load(open(mpath))
    ph: Dict[str, dict] = m["per_hospital"]
    hosp = list(m["hospitals"])

    # The rebuild command pins the hospitals EXPLICITLY rather than repeating
    # --bands/--per-band. A band draw depends on the size distribution, which
    # shifts with the eICU release and with MIN_VISITS; the explicit list is the
    # only form that reproduces the same cohort years later.
    rebuild = ("python examples/fedpyhealth/utils/cohort.py"
               f" --name {m['cohort_name']}"
               f" --hospitals {','.join(hosp)}"
               f" --split {m['split_method']}"
               f" --seed {m['seed']} --out <cache dir>")

    rec = {
        "kind": "cohort_selection",
        "cohort_name": m["cohort_name"],
        "description": args.description or (
            f"{len(hosp)} eICU hospitals, selected by {m['selected_by']}, split "
            f"{m['split_method']} {m['fracs']['train']:.0%}/"
            f"{m['fracs']['val']:.0%}/{m['fracs']['test']:.0%}. This file is the "
            "git record of WHICH hospitals; the data lives in a cohort cache "
            "built by utils/cohort.py."),
        "status": args.status,
        "selected_by": m["selected_by"],
        "bands": m.get("bands"),
        "per_band": m.get("per_band"),
        "seed": m["seed"],
        "split_method": m["split_method"],
        "fracs": m["fracs"],
        "rare_prevalence_max": m["rare_prevalence_max"],
        "rare_min_patients": m["rare_min_patients"],
        "global_rare_prevalence_max": m["global_rare_prevalence_max"],
        "eicu": "eICU CRD 2.0",
        "created_utc": m.get("created_utc"),
        "git_sha": m.get("git_sha"),
        "rebuild": rebuild,
        "totals": {
            "n_patients": sum(ph[h]["n_total"] for h in hosp),
            "n_train": sum(ph[h]["n_train"] for h in hosp),
            "n_val": sum(ph[h]["n_val"] for h in hosp),
            "n_test": sum(ph[h]["n_test"] for h in hosp),
            "n_pooled_rare_codes": m["n_pooled_rare_codes"],
            "n_global_rare_codes": len(m.get("global_rare_codes", [])),
            "vocab_size": m["vocab_size"],
            "cohort_distinct_codes": m["cohort_distinct_codes"],
        },
        "hospitals": [
            {"hospital_id": h,
             "n_patients": ph[h]["n_total"],
             "n_train": ph[h]["n_train"],
             "n_val": ph[h]["n_val"],
             "n_test": ph[h]["n_test"],
             "n_rare_codes": ph[h]["n_rare_codes"],
             "min_rare_prevalence": round(ph[h]["min_rare_prevalence"], 5)}
            for h in hosp],
    }

    out = args.out or os.path.join(COHORT_DIR, f"{m['cohort_name']}.config.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2)

    t = rec["totals"]
    print(f"wrote {out}")
    print(f"  {m['cohort_name']}: {len(hosp)} hospitals, "
          f"{t['n_patients']:,} patients "
          f"({t['n_train']:,} train / {t['n_val']:,} val / {t['n_test']:,} test)")
    print(f"  {t['n_pooled_rare_codes']} pooled rare codes, "
          f"{t['n_global_rare_codes']} globally rare")
    print("  sizes: " + ", ".join(
        f"{h}({ph[h]['n_total']})" for h in hosp))


if __name__ == "__main__":
    main()
