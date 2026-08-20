"""One table for the ft_epochs x lr sweep: weight drift, output drift, fidelity.

The sweep asks whether ``fedavg_ft`` was simply under-personalised at
``ft_epochs=2``. Answering that needs three numbers per config, because the
first two disagreed at ft=2 and the disagreement is the interesting part:

``rel drift``     ||theta_i - theta_global|| / ||theta_global||, averaged over
                  the 8 hospitals. At ft=2 this was under 1% -- which looked
                  like "nothing happened".
``upd cos``       mean pairwise cosine between the hospitals' update vectors.
                  Positive means the sites are personalising in compatible
                  directions; near zero or negative means they are pulling
                  apart, which is the client-drift regime.
``JS to own``     Jensen-Shannon distance from each hospital's generator output
                  to that hospital's own real distribution, averaged. At ft=2
                  this moved a LOT (0.524 -> 0.442) despite the tiny weight
                  movement, because a generator's output is far more sensitive
                  to small weight changes than the norm suggests. This is the
                  column that actually tracks personalisation.
``Pearson``       Test 1 prevalence fidelity, if test1_ftsweep.json exists.

Reference points, from the headline runs:

    fedavg          JS 0.524    the un-personalised starting point
    fedavg_ft ft=2  JS 0.442    two epochs of fine-tuning
    local           JS 0.275    trained from scratch on own data -- the target

A config whose JS falls below 0.442 personalised further; one approaching 0.275
has closed the gap to ``local``. If JS bottoms out well above 0.275 no matter
how much fine-tuning is applied, the federated warm start is the constraint, not
the fine-tuning budget.

Usage
-----
::

    python sweep_ft_report.py
    python sweep_ft_report.py --glob '_outputs/ftsweep_*_save'

Runs after the sweep array. Loads 9 x 9 checkpoints, so allow a few minutes.
"""

import argparse
import glob as globmod
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch

from utils.eda.drift import flatten, load_state
from utils.eda.generator_distance import (
    js_distance,
    prevalence,
    real_prevalence,
)
from utils.cohort import DEFAULT_CACHE_DIR, load_manifest, read_trajectories

# Measured on the headline runs; printed as reference rows so a sweep number is
# never read without the two points that bracket it.
REFERENCE = {"fedavg": 0.5235, "fedavg_ft (ft=2)": 0.4421, "local": 0.2747}
TAG_RE = re.compile(r"ftsweep_e(\d+)_lr([\w.+-]+)_save$")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="_outputs/ftsweep_*_save",
                   help="shell glob matching the sweep run directories")
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR)
    p.add_argument("--fold", default="train",
                   help="real fold the generators were trained on")
    p.add_argument("--test1", default="_outputs/results/tests/test1_ftsweep.json",
                   help="test1 results to merge in, if present")
    p.add_argument("--out", default=os.path.join("_outputs", "eda"))
    p.add_argument("--skip-drift", action="store_true",
                   help="skip the weight-space columns (they load 9 "
                        "checkpoints per config and dominate the runtime)")
    return p


def _table(header, rows) -> None:
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(str(header[i])), *(len(r[i]) for r in cells)) if cells
         else len(str(header[i])) for i in range(len(header))]
    print("  " + "  ".join(str(h).ljust(w[0]) if i == 0 else str(h).rjust(w[i])
                           for i, h in enumerate(header)))
    print("  " + "-" * (sum(w) + 2 * (len(w) - 1)))
    for r in cells:
        print("  " + "  ".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i])
                               for i, c in enumerate(r)))


def drift_stats(save_dir: str) -> Dict[str, float]:
    """Mean relative weight drift and mean pairwise update cosine.

    Valid here for the same reason it was valid for the headline fedavg_ft run:
    all eight fine-tuned models share one warm start, so their deltas live in a
    single basin and directions are comparable. It would NOT be valid across
    independently initialised models.
    """
    gpath = os.path.join(save_dir, "fedavg_state.pt")
    hids = sorted(f[3:-3] for f in os.listdir(save_dir)
                  if f.startswith("ft_") and f.endswith(".pt"))
    if not os.path.exists(gpath) or not hids:
        return {}
    glob_state = load_state(gpath)
    keys = sorted(glob_state)
    gvec = flatten(glob_state, keys)
    gnorm = float(torch.linalg.vector_norm(gvec))

    deltas, rels = {}, []
    for hid in hids:
        vec = flatten(load_state(os.path.join(save_dir, f"ft_{hid}.pt")), keys)
        d = vec - gvec
        deltas[hid] = d
        rels.append(float(torch.linalg.vector_norm(d)) / gnorm)

    cos = []
    for i, a in enumerate(hids):
        for b in hids[i + 1:]:
            cos.append(float(torch.nn.functional.cosine_similarity(
                deltas[a], deltas[b], dim=0)))
    return {"rel_drift_mean": float(np.mean(rels)),
            "rel_drift_min": float(np.min(rels)),
            "rel_drift_max": float(np.max(rels)),
            "update_cos_mean": float(np.mean(cos)) if cos else float("nan")}


def js_to_own(save_dir: str, real: Dict[str, np.ndarray],
              vocab: Dict[str, int]) -> Dict[str, float]:
    """Mean JS distance from each hospital's synthetic output to its own real."""
    path = os.path.join(save_dir, "synthetic.json")
    if not os.path.exists(path):
        return {}
    per = (json.load(open(path)).get("per_hospital") or {})
    per_h = {h: js_distance(prevalence(pats, vocab), real[h])
             for h, pats in per.items() if h in real}
    if not per_h:
        return {}
    return {"js_mean": float(np.mean(list(per_h.values()))),
            "js_per_hospital": per_h}


def load_test1(path: str) -> Dict[str, Dict[str, float]]:
    """``{run_name: {pearson, r2}}`` from a test1 results file, if present.

    test1 stores ``runs -> hospital -> metric -> [mean, std]``: each hospital is
    scored against its OWN validation split, so a run-level number is the macro
    mean over the 8 hospitals, weighting every site equally regardless of size.
    That is deliberate -- hospital 420 holds 36% of the cohort, and a
    size-weighted mean would report little more than how well 420 was fitted.
    """
    if not os.path.exists(path):
        return {}
    blob = json.load(open(path))
    out = {}
    for name, per_hospital in (blob.get("runs") or {}).items():
        acc: Dict[str, List[float]] = {}
        for metrics in per_hospital.values():
            if not isinstance(metrics, dict):
                continue
            for key, val in metrics.items():
                # [mean, std] pairs; take the mean.
                v = val[0] if isinstance(val, (list, tuple)) and val else val
                if isinstance(v, (int, float)) and v == v:
                    acc.setdefault(key, []).append(float(v))
        if not acc:
            continue
        out[name] = {
            "pearson": (float(np.mean(acc["PrevVal_All_Prevalence_Pearson"]))
                        if "PrevVal_All_Prevalence_Pearson" in acc else None),
            "r2": (float(np.mean(acc["PrevVal_All_Prevalence_R2"]))
                   if "PrevVal_All_Prevalence_R2" in acc else None),
            "rmse": (float(np.mean(acc["PrevVal_All_Prevalence_RMSE"]))
                     if "PrevVal_All_Prevalence_RMSE" in acc else None),
            "pearson_rare": (
                float(np.mean(acc["PrevVal_Rare_Prevalence_Pearson"]))
                if "PrevVal_Rare_Prevalence_Pearson" in acc else None),
        }
    return out


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    dirs = sorted(d for d in globmod.glob(args.glob) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(
            f"no sweep directories match {args.glob!r}. Has the array run?")

    manifest = load_manifest(args.cohort_cache)
    hospitals = list(manifest["hospitals"])
    codes = sorted(manifest["pooled_rare_codes"])
    vocab = {c: i for i, c in enumerate(codes)}
    traj = read_trajectories(args.cohort_cache, args.fold, hospitals)
    real = {h: real_prevalence(traj[h], vocab) for h in hospitals}
    print(f"cohort {manifest['cohort_name']}   {len(codes)} rare codes   "
          f"{len(dirs)} sweep dirs matched")

    t1 = load_test1(args.test1)
    if t1:
        print(f"merged Test 1 prevalence from {args.test1}")
    else:
        print(f"no Test 1 results at {args.test1} "
              "(run sweep_ft_score.sbatch to fill the Pearson column)")

    rows, blob = [], {}
    for d in dirs:
        m = TAG_RE.search(d)
        ft, lr = (int(m.group(1)), m.group(2)) if m else ("?", "?")
        tag = os.path.basename(d)[:-5]

        if not os.path.exists(os.path.join(d, "synthetic.json")):
            rows.append([ft, lr, "—", "—", "—", "—", "—", "not finished"])
            continue

        js = js_to_own(d, real, vocab)
        dr = {} if args.skip_drift else drift_stats(d)
        scores = t1.get(tag, {})
        blob[tag] = {"ft_epochs": ft, "lr": lr, **dr,
                     **{k: v for k, v in js.items() if k != "js_per_hospital"},
                     "js_per_hospital": js.get("js_per_hospital", {}),
                     **scores}

        def f(v, p=4):
            return "—" if v is None or (isinstance(v, float) and v != v) \
                else f"{v:.{p}f}"

        rows.append([
            ft, lr,
            f(dr.get("rel_drift_mean"), 5),
            f"{f(dr.get('rel_drift_min'),5)}/{f(dr.get('rel_drift_max'),5)}"
            if dr else "—",
            f(dr.get("update_cos_mean"), 3),
            f(js.get("js_mean")),
            f(scores.get("pearson")),
            "",
        ])

    # Sort lr NUMERICALLY. Lexicographic ordering of the labels puts 1e-4 before
    # 1e-5, which silently misreads a dose-response curve as non-monotone.
    def _lr_num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    rows.sort(key=lambda r: (r[0] if isinstance(r[0], int) else 999,
                             _lr_num(r[1])))
    print("\nSweep configs")
    _table(["ft", "lr", "rel drift", "drift min/max", "upd cos",
            "JS to own", "Pearson", ""], rows)

    print("\nReference (headline runs, same cohort)")
    _table(["arm", "JS to own"],
           [[k, f"{v:.4f}"] for k, v in REFERENCE.items()])
    print("  lower JS = the generator matches its own hospital more closely.")
    print("  local's 0.2747 is the target; fedavg's 0.5235 is the un-personalised"
          "\n  starting point. A config below 0.4421 personalised further than "
          "ft=2 did.")

    done = [b for b in blob.values() if b.get("js_mean") is not None]
    if done:
        best = min(done, key=lambda b: b["js_mean"])
        print(f"\nClosest to its own sites: ft={best['ft_epochs']} "
              f"lr={best['lr']}   JS {best['js_mean']:.4f}")
        gap = (REFERENCE["fedavg"] - best["js_mean"]) / (
            REFERENCE["fedavg"] - REFERENCE["local"])
        print(f"  that closes {gap:.0%} of the fedavg -> local gap "
              f"(ft=2 closed "
              f"{(REFERENCE['fedavg'] - REFERENCE['fedavg_ft (ft=2)']) / (REFERENCE['fedavg'] - REFERENCE['local']):.0%})")
        # Only a conclusion once the sweep is complete. With configs still
        # running, "nothing beat ft=2" may simply mean the configs that could
        # have are not in yet -- and an early partial table is exactly what
        # gets copied into notes as if it were the result.
        pending = [t for t, b in blob.items() if b.get("js_mean") is None]
        missing = len(dirs) - len(done)
        if missing or pending:
            print(f"\n  PARTIAL: {len(done)} of {len(dirs)} configs reported. "
                  "Draw no conclusion about\n  whether more fine-tuning helps "
                  "until the rest land -- the configs most\n  likely to beat "
                  "ft=2 are the ones that take longest to finish.")
        elif best["js_mean"] > REFERENCE["fedavg_ft (ft=2)"]:
            print("  NOTE: no config beat ft=2. More fine-tuning did not help, "
                  "so the\n  federated warm start is the constraint, not the "
                  "fine-tuning budget.")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ftsweep_report.json")
    with open(path, "w") as fh:
        json.dump({"reference": REFERENCE, "configs": blob}, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
