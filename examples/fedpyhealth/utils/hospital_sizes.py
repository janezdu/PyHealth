"""Exact post-task hospital sizes, straight from the eICU CSVs.

``cohort.py --list-sizes`` gives the same numbers, but only after loading eICU
through PyHealth and running the task -- an hour on a GPU node for two columns
of integers. This reads ``patient.csv`` and ``diagnosis.csv`` with the stdlib
and gets there in a couple of minutes on a login node, with no torch import and
no cluster job.

    python examples/fedpyhealth/utils/hospital_sizes.py --bands 100-1499 \\
        --per-band 8 --seed 1 --verify $FEDCOHORT_CACHE/hilo8_random/manifest.json

It prints every hospital's size, draws a cohort from the bands, and emits the
``--hospitals a,b,c`` line to paste into ``cohort.py``. Pinning the draw that way
is what makes the cohort reproducible: ``draw_bands`` samples uniformly within a
band, so re-drawing at the same seed from a different size table is a different
cohort.

Why this can be exact
---------------------
"Post-task size" is not ``patient.csv`` row count -- that runs 1.2-1.7x higher,
by a ratio that varies per hospital. Three things shrink it, and all three are
visible in the CSVs:

1. A patient is a ``uniquepid``, not a unit stay. eICU patients readmit.
2. ``EHRGenerationEICU`` keeps a unit stay only if it carries >= 1 non-empty
   ``icd9code``, and drops a patient with no surviving stay (``min_visits=1``).
3. ``cohort.py`` drops patients seen at more than one ``hospitalid``.

Apply those and the count matches. ``--verify`` proves it against a manifest
built the slow way rather than asking you to take it on faith -- run it once
against a known cohort before trusting a fresh draw.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cohort import EICU_ROOT, draw_bands, parse_bands  # noqa: E402


def _clean_icd9(raw: str) -> str:
    """Mirror of ``pyhealth.tasks.generate_ehr._clean_icd9``.

    Duplicated rather than imported: importing the task pulls in
    ``pyhealth.processors`` and therefore torch, which is a heavy import for a
    script whose whole point is to stay light enough for a login node. It is two
    lines and it is pinned by ``--verify`` -- if the real one ever changes, the
    verification against a real manifest stops matching.
    """
    if raw is None:
        return ""
    return str(raw).split(",")[0].strip()


def post_task_sizes(eicu_root: str, drop_cross: bool = True
                    ) -> Tuple[Dict[str, int], dict]:
    """``{hospital_id: n_patients}`` after the task's filters.

    Args:
        eicu_root: Folder holding ``patient.csv`` and ``diagnosis.csv``.
        drop_cross: Drop patients with stays at more than one hospital, as
            ``cohort.py`` does.

    Returns:
        ``(sizes, stats)`` -- stats records what each filter removed, so a
        surprising number can be traced to the step that caused it.
    """
    ppath = os.path.join(eicu_root, "patient.csv")
    dpath = os.path.join(eicu_root, "diagnosis.csv")
    for path in (ppath, dpath):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Point --eicu-root at the "
                             "folder holding patient.csv and diagnosis.csv.")

    # Pass 1: stay -> patient, and patient -> the hospitals it was seen at.
    print(f"reading {ppath} ...", flush=True)
    stay_to_pid: Dict[str, str] = {}
    pid_hospitals: Dict[str, Set[str]] = {}
    with open(ppath, newline="") as fh:
        for row in csv.DictReader(fh):
            pid, hid = row["uniquepid"], row["hospitalid"]
            stay_to_pid[row["patientunitstayid"]] = pid
            pid_hospitals.setdefault(pid, set()).add(hid)
    n_stays, n_pids = len(stay_to_pid), len(pid_hospitals)

    cross = {p for p, h in pid_hospitals.items() if len(h) > 1} if drop_cross \
        else set()

    # Pass 2: which patients keep at least one codeful stay. Only the patient
    # id is retained, so memory stays at one set of ~119k strings rather than
    # the ~2.7M diagnosis rows.
    print(f"reading {dpath} ...", flush=True)
    keep: Set[str] = set()
    n_rows = n_coded = 0
    with open(dpath, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            if not _clean_icd9(row["icd9code"]):
                continue
            n_coded += 1
            pid = stay_to_pid.get(row["patientunitstayid"])
            if pid is not None and pid not in cross:
                keep.add(pid)

    sizes: Dict[str, int] = {}
    for pid in keep:
        # Single-element set after the cross filter, so this is the patient's
        # one hospital.
        hid = next(iter(pid_hospitals[pid]))
        sizes[hid] = sizes.get(hid, 0) + 1

    stats = {
        "patient_csv_stays": n_stays,
        "patient_csv_patients": n_pids,
        "cross_hospital_dropped": len(cross),
        "diagnosis_rows": n_rows,
        "diagnosis_rows_with_code": n_coded,
        "patients_kept": len(keep),
        "hospitals": len(sizes),
    }
    return sizes, stats


def verify(sizes: Dict[str, int], manifest_path: str) -> bool:
    """Compare against a manifest built the slow way. Returns True on an exact
    match for every hospital in that cohort."""
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    print(f"\nverifying against {manifest_path}")
    print(f"  ({manifest.get('cohort_name')}, built "
          f"{manifest.get('created_utc')})\n")
    print(f"  {'hospital':>9} {'manifest':>9} {'from csv':>9}  ")
    ok = True
    for hid, h in manifest["per_hospital"].items():
        want, got = h["n_total"], sizes.get(hid, 0)
        flag = "ok" if want == got else f"MISMATCH ({got - want:+d})"
        ok = ok and want == got
        print(f"  {hid:>9} {want:>9,} {got:>9,}  {flag}")
    print("\n  " + ("exact match -- CSV sizes are trustworthy" if ok else
                    "DOES NOT MATCH: do not draw a cohort from these numbers"))
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eicu-root", default=EICU_ROOT,
                   help="eICU CRD root (default: $EICU_ROOT)")
    p.add_argument("--bands", default="100-1499",
                   help="inclusive size bands to draw from, e.g. "
                        "'100-1499,1500-' (default: %(default)s)")
    p.add_argument("--per-band", type=int, default=8,
                   help="hospitals to draw from each band")
    p.add_argument("--seed", type=int, default=1,
                   help="seeds the draw; must match the --seed you later pass "
                        "to cohort.py for the record to be consistent")
    p.add_argument("--verify", metavar="MANIFEST",
                   help="path to an existing cohort manifest.json; check these "
                        "CSV sizes reproduce its per-hospital n_total exactly")
    p.add_argument("--keep-cross-hospital", dest="drop_cross",
                   action="store_false",
                   help="keep patients seen at more than one hospital")
    p.add_argument("--list", action="store_true",
                   help="print every hospital's size and stop, without drawing")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.eicu_root:
        raise SystemExit("set EICU_ROOT or pass --eicu-root")

    sizes, stats = post_task_sizes(args.eicu_root, args.drop_cross)
    print("\nhow the count was reached:")
    for k, v in stats.items():
        print(f"  {k:28s} {v:>10,}")

    if args.verify and not verify(sizes, args.verify):
        raise SystemExit(1)

    if args.list:
        print(f"\n{len(sizes)} hospitals by post-task patient count:\n")
        for hid, n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {hid:>8} {n:>8,}")
        return

    bands = parse_bands(args.bands)
    for lo, hi in bands:
        pool = [h for h, n in sizes.items() if lo <= n <= hi]
        print(f"\nband {lo}-{hi if hi < 10 ** 9 else ''}: {len(pool)} eligible "
              f"hospitals")
    chosen = draw_bands(sizes, bands, args.per_band, args.seed)
    ids = [h for h, _ in chosen]
    print("\ndrawn: " + ", ".join(f"{h} (n={sizes[h]:,})" for h in ids))
    print(f"\n  --hospitals {','.join(ids)}")


if __name__ == "__main__":
    main()
