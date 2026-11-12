"""Which agreement metric should the frontier figure use?

The two metrics in use each fail in an opposite way:

* **Pearson** is scale-invariant, so it cannot see an arm emitting every code at
  2.2x the right rate -- unweighted ``centralized`` scores +0.58 while emitting
  34 codes per patient against a real 15.7. It also compresses every arm into
  [0.7, 1.0], where a frontier line through two anchors 0.11 apart is not
  determined well enough to rank anything.
* **R^2 vs identity** sees that failure, but unboundedly: the same arm scores
  -53. One bad arm then sets the axis and squashes every other point, and on
  rare codes the denominator sum((y - ybar)^2) is tiny, so the blow-up is worse
  exactly where the project's claim lives.

What is wanted is an agreement statistic that PENALISES scale error but stays
bounded. Candidates computed here, all on the same prevalence vectors:

=================  ===========================================================
metric             what it does
=================  ===========================================================
pearson            correlation only; blind to scale and offset
r2_identity        1 - SSE/SST against y = x; unbounded below
ccc                Lin's concordance correlation. Pearson multiplied by a bias
                   term that falls as the means or variances diverge. Bounded
                   in [-1, 1]. This is the standard "agreement with identity"
                   statistic and is what R^2-vs-identity is reaching for.
r2_log             R^2 on log10 prevalence. Prevalence error is multiplicative,
                   so a uniform 2.2x inflation becomes a constant +0.34 offset
                   rather than an error that scales with the value.
spearman           rank correlation; the floor for "ordering only"
slope              OLS slope through the origin -- not a score, a diagnostic.
                   1.0 is calibrated, 2.2 means every prevalence is 2.2x high.
=================  ===========================================================

Prints every arm under every metric so the choice can be made on evidence
rather than on which one flatters the result.

Example:
    python examples/fedpyhealth/scripts/compare_prevalence_metrics.py \\
        --run local=_outputs/local_full_hilo8_random_nes_save \\
        --run centralized=_outputs/centralized_full_hilo8_random_nes_save
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.cohort import (  # noqa: E402
    DEFAULT_CACHE_DIR, load_manifest, load_shared_generator, load_synthetic,
    read_trajectories, scored_codes,
)

EPS = 1e-6

# metric key -> the name plot_rw_frontier.py's --metric expects
EMIT_NAME = {"pearson": "Pearson", "r2_identity": "R2", "ccc": "CCC",
             "r2_log": "r2_log", "spearman": "spearman", "slope": "slope"}


def prevalence(carriers: Counter, n: int, codes) -> np.ndarray:
    return np.array([carriers.get(c, 0) / max(1, n) for c in codes])


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    ``2 * cov / (var_x + var_y + (mean_x - mean_y)^2)``. Equals Pearson when the
    two vectors have identical means and variances, and falls away from it as
    either diverges -- so it is Pearson with a calibration penalty, bounded.
    """
    vx, vy = x.var(), y.var()
    denom = vx + vy + (x.mean() - y.mean()) ** 2
    return float(2 * np.cov(x, y, bias=True)[0, 1] / denom) if denom else float("nan")


def r2_identity(x: np.ndarray, y: np.ndarray) -> float:
    """``sklearn.r2_score(y_true=x, y_pred=y)``.

    The total sum of squares comes from the REAL vector, not the synthetic one.
    Taking it from the synthetic side instead makes an arm that inflates every
    prevalence look better, because its own variance grows with the error --
    which is how an earlier version of this script reported centralized at
    -0.03 where test1_prevalence correctly reports -53.
    """
    sst = ((x - x.mean()) ** 2).sum()
    return float(1 - ((y - x) ** 2).sum() / sst) if sst else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v))
        return r
    return pearson(rank(x), rank(y))


def metrics(real: np.ndarray, syn: np.ndarray) -> dict:
    """``real`` is the reference (x); ``syn`` is what the generator produced."""
    lr, ls = np.log10(real + EPS), np.log10(syn + EPS)
    denom = (real ** 2).sum()
    return {
        "pearson": pearson(real, syn),
        "r2_identity": r2_identity(real, syn),
        "ccc": ccc(real, syn),
        "r2_log": r2_identity(lr, ls),
        "spearman": spearman(real, syn),
        "slope": float((real * syn).sum() / denom) if denom else float("nan"),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR)
    p.add_argument("--run", action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--fold", default="test", choices=["val", "test"])
    p.add_argument("--cap", type=int, default=8000,
                   help="synthetic patients per hospital")
    p.add_argument("--out", default="_outputs/results/prevalence_metrics.json")
    p.add_argument("--emit-test1", default="",
                   help="also write PREFIX_spec.json / PREFIX_gen.json in the "
                        "shape test1_prevalence produces, so plot_rw_frontier "
                        "can draw any of these metrics. The second value in "
                        "each pair is a standard error in the real files and "
                        "is written as 0 here -- these are point estimates, "
                        "not bootstrapped, so do not read error bars off them.")
    args = p.parse_args(argv)

    manifest = load_manifest(args.cohort_cache)
    hosp = [str(h) for h in manifest["hospitals"]]
    real_traj = read_trajectories(args.cohort_cache, args.fold, hosp)
    rare = set(scored_codes(manifest, args.fold, 1))
    allc = sorted({c for pats in real_traj.values() for v in pats.values()
                   for visit in v for c in visit})
    pools = {"All": allc, "Rare": sorted(rare & set(allc))}

    def carriers_traj(pats):
        out = Counter()
        for visits in pats.values():
            out.update({c for v in visits for c in v})
        return out

    def carriers_syn(recs):
        out = Counter()
        for r in recs:
            out.update({c for v in r["visits"] for c in v})
        return out

    pooled_real = {p: v for pats in real_traj.values() for p, v in pats.items()}

    rows = {}
    per_site = {"spec": {}, "gen": {}}
    for spec in args.run:
        name, save_dir = spec.split("=", 1)
        per = load_synthetic(save_dir)
        shared = load_shared_generator(save_dir)
        rows[name] = {}
        per_site["spec"][name] = {h: {} for h in hosp}
        per_site["gen"][name] = {h: {} for h in hosp}
        for scope, codes in pools.items():
            # Specialist: each site's synthetic against ITS OWN real fold.
            # Generalist: the same synthetic against the POOLED real fold.
            # Both averaged over sites, the reduction test1_prevalence reports.
            vals = []
            pooled_vec = prevalence(carriers_traj(pooled_real),
                                    len(pooled_real), codes)
            for h in hosp:
                src = (per[hosp[0]] if shared else per.get(h, []))[:args.cap]
                syn = prevalence(carriers_syn(src), len(src), codes)
                real = prevalence(carriers_traj(real_traj[h]),
                                  len(real_traj[h]), codes)
                m_spec = metrics(real, syn)
                m_gen = metrics(pooled_vec, syn)
                vals.append(m_spec)
                for view, mm in (("spec", m_spec), ("gen", m_gen)):
                    for k, v in mm.items():
                        key = f"PrevVal_{scope}_Prevalence_{EMIT_NAME[k]}"
                        per_site[view][name][h][key] = [v, 0.0]
            rows[name][scope] = {k: float(np.mean([v[k] for v in vals]))
                                 for k in vals[0]}

    order = ["pearson", "spearman", "ccc", "r2_log", "r2_identity", "slope"]
    for scope in pools:
        print(f"\n=== {scope} codes ({len(pools[scope])}) "
              f"-- specialist view, mean over 8 sites ===")
        print(f"{'arm':22s}" + "".join(f"{m:>13s}" for m in order))
        for name, sc in rows.items():
            print(f"{name:22s}" + "".join(f"{sc[scope][m]:13.3f}" for m in order))
    print("\nslope is a diagnostic, not a score: 1.0 = calibrated, >1 = every "
          "prevalence inflated.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")

    if args.emit_test1:
        os.makedirs(os.path.dirname(args.emit_test1) or ".", exist_ok=True)
        for view, suffix in (("spec", "_spec.json"), ("gen", "_gen.json")):
            path = args.emit_test1 + suffix
            with open(path, "w") as fh:
                json.dump({"kind": "test", "test": "prevalence_metrics",
                           "note": "point estimates, no bootstrap; the second "
                                   "value in each pair is a placeholder 0",
                           "runs": per_site[view]}, fh)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
