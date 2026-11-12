"""Does federation let a hospital emit codes its own model never could?

Every result in this project so far compares PREVALENCES -- how closely an
arm's emitted rates track the real ones. That says nothing about which codes
were reachable at all. This asks the coverage question directly, per hospital:

    Of the codes hospital h actually needs but its LOCAL-ONLY generator never
    emits, how many does the FEDAVG global generator emit? How many does
    FEDAVG_FT, that hospital's fine-tuned copy? How many does CENTRALIZED?

Sets, per hospital ``h``, restricted to the scored code pool:

======  ====================================================================
R_h     codes in h's real TEST split -- what h actually sees
T_h     codes in h's real TRAIN split -- what a local model could have learnt
L_h     codes h's local-only generator emits
F       codes the fedavg global generator emits (one set, shared by all)
F_h     codes fedavg_ft's copy for h emits
C       codes centralized emits
======  ====================================================================

    missed_h  = R_h \\ L_h                     needed, local could not say it
    recovered = |missed_h & X| / |missed_h|    X in {F, F_h, C, ...}
    spurious  = |X \\ R_h| / |X|               emitted, h has no use for it

WHY ``spurious`` IS REPORTED BESIDE EVERY ``recovered``. A generator that emits
the whole vocabulary covers everything local missed, trivially, and scores
recovered = 1.0. HALO's sigmoid decoder does exactly this when it hedges -- 460
codes per visit against a real 12.5 -- so recovered ALONE cannot distinguish
"federation supplied the code" from "the model says everything". The pair can.
Read them together or not at all.

Two controls bracket every number:

* ``real_pooled`` substitutes the pooled real TRAIN split for a generator. That
  is the most any federated method could recover, so it converts every figure
  into a fraction of achievable rather than an absolute.
* ``prior`` draws the same number of patients from the pooled marginal, each
  code independently at its pooled rate. This floor is high, and that is the
  finding it protects against: it is what emitting a lot buys on its own.

``--min-carriers`` is the other guard. At k=1 a single hallucinated patient
counts as covering a code; at k=5 it does not. Expect the hedging arms to
disagree between the two, and trust k=5.

THE SAMPLE-SIZE TRAP, which an earlier version of this script walked into.
The number of DISTINCT codes in a set of patients grows with how many patients
there are. Drawing 8,000 synthetic patients and comparing their code set
against a 40-patient real test split does not measure coverage, it measures
8,000 > 40 -- and it produced the nonsense result that a 142-patient
hospital's local generator "emits more codes than its site needs". Every arm is
therefore capped at the size of the real split it is compared against, and
because a 40-patient draw is mostly luck, every set statistic is averaged over
``--n-resamples`` independent subsamples. The controls are drawn the same way.

Runs on CPU in a couple of minutes -- no model is loaded, every arm's
``synthetic.json`` is already on disk.

Example:
    python examples/fedpyhealth/scripts/eda_code_coverage.py \\
        --cohort-cache $FEDCOHORT_CACHE/hilo8_random \\
        --run local=_outputs/local_full_hilo8_random_nes_save \\
        --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save \\
        --out _outputs/results/eda_code_coverage
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.cohort import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    hospitals as cohort_hospitals,
    load_manifest,
    load_shared_generator,
    load_synthetic,
    read_trajectories,
    scored_codes,
)

SEED = 0


def code_carriers(patients: Sequence[dict], key: str = "visits") -> Counter:
    """How many PATIENTS carry each code. Per patient, not per mention.

    eICU charts a diagnosis repeatedly inside one stay, so counting mentions
    would let a single talkative patient look like broad coverage.
    """
    out: Counter = Counter()
    for p in patients:
        seen = {c for visit in p[key] for c in visit}
        out.update(seen)
    return out


def traj_carriers(traj: Dict[str, list]) -> Counter:
    """``code_carriers`` for the ``{patient: visits}`` shape the cache returns."""
    out: Counter = Counter()
    for visits in traj.values():
        out.update({c for visit in visits for c in visit})
    return out


def emitted(counts: Counter, pool: Set[str], k: int) -> Set[str]:
    """Codes in ``pool`` carried by at least ``k`` patients."""
    return {c for c, n in counts.items() if c in pool and n >= k}


def prior_draw(pooled_counts: Counter, n_patients: int, n_draw: int,
               pool: Set[str], k: int, seed: int) -> Set[str]:
    """The floor arm: each code drawn independently at its pooled rate.

    Not a generator -- a model with no structure whatsoever, emitting codes
    independently at the right marginal frequency. Anything that fails to beat
    this has learnt nothing about which codes go together.
    """
    rng = np.random.default_rng(seed)
    out = set()
    for c in pool:
        p = pooled_counts.get(c, 0) / max(1, n_patients)
        if p <= 0:
            continue
        # P(at least k carriers in n_draw draws), evaluated by simulation on the
        # count itself rather than per patient -- same distribution, far cheaper.
        if rng.binomial(n_draw, p) >= k:
            out.add(c)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR)
    p.add_argument("--run", action="append", default=[], metavar="NAME=SAVE_DIR",
                   help="an arm to measure; repeatable")
    p.add_argument("--fold", default="test", choices=["val", "test"],
                   help="which real split defines what a hospital NEEDS")
    p.add_argument("--min-carriers", default="1,5",
                   help="a code counts as emitted at >= this many carrying "
                        "patients. Comma-separated; 1 is the permissive "
                        "reading and 5 the one to trust")
    p.add_argument("--budget", default="match",
                   help="synthetic patients per arm per hospital. DISTINCT-CODE "
                        "COUNT GROWS WITH SAMPLE SIZE, so this is not a detail: "
                        "comparing 8000 synthetic draws against a 40-patient "
                        "real split measures 8000 > 40 and nothing else. "
                        "'match' (default) caps each hospital's synthetic set at "
                        "the size of the real split it is compared against, "
                        "which is the only setting under which |L| and |R| mean "
                        "the same thing. An integer overrides it, and should be "
                        "used only when comparing arms to each other rather "
                        "than to real data.")
    p.add_argument("--n-resamples", type=int, default=20,
                   help="with --budget match the synthetic subsample is random, "
                        "so every set statistic is averaged over this many "
                        "draws; one draw of 40 patients is far too noisy to "
                        "read")
    p.add_argument("--code-pool", default="scored",
                   choices=["scored", "rare", "all"],
                   help="'scored' is the pooled-rare list with enough fold "
                        "positives (what Test 2 scores); 'rare' is the "
                        "manifest's strict global_rare subset; 'all' is every "
                        "code present in the fold")
    p.add_argument("--min-positives", type=int, default=1)
    p.add_argument("--out", default="_outputs/results/eda_code_coverage")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    manifest = load_manifest(args.cohort_cache)
    hosp = [str(h) for h in manifest["hospitals"]]
    real_eval = read_trajectories(args.cohort_cache, args.fold, hosp)
    real_train = read_trajectories(args.cohort_cache, "train", hosp)

    fold_codes = set()
    for pats in real_eval.values():
        for visits in pats.values():
            fold_codes.update(c for v in visits for c in v)
    if args.code_pool == "all":
        pool = set(fold_codes)
    elif args.code_pool == "rare":
        pool = set(manifest.get("global_rare_codes", [])) & fold_codes
    else:
        pool = set(scored_codes(manifest, args.fold, args.min_positives))
    print(f"code pool ({args.code_pool}): {len(pool)} codes; "
          f"{len(fold_codes)} present in the {args.fold} fold")

    n_train = {h: len(real_train[h]) for h in hosp}
    n_eval_pat = {h: len(real_eval[h]) for h in hosp}
    match = str(args.budget) == "match"
    reps = args.n_resamples if match else 1

    def budget_for(h):
        """Synthetic draws for hospital h: the real split's size, or a fixed N."""
        return n_eval_pat[h] if match else int(args.budget)

    def rng_pick(n, k, seed):
        if k >= n:
            return list(range(n))
        return np.random.default_rng(seed).choice(n, size=k, replace=False)

    print("budget: " + ("matched to each site's real "
          + args.fold + " split (" + ", ".join(f"{h}:{n_eval_pat[h]}" for h in hosp)
          + f"), averaged over {reps} subsamples"
          if match else f"{args.budget} synthetic patients per site"))
    order = sorted(hosp, key=lambda h: n_train[h])
    print("hospitals by train size: "
          + ", ".join(f"{h}({n_train[h]})" for h in order))

    # --- per-arm emitted sets, per hospital, per threshold ---------------- #
    ks = [int(v) for v in args.min_carriers.split(",")]
    arms: Dict[str, Dict[int, Dict[str, Set[str]]]] = {}
    for spec in args.run:
        name, save_dir = spec.split("=", 1)
        per = load_synthetic(save_dir)
        shared = load_shared_generator(save_dir)
        arms[name] = {k: {} for k in ks}
        for h in hosp:
            src = per[hosp[0]] if shared else per.get(h, [])
            cap = budget_for(h)
            # Average the emitted set over several equal-sized subsamples: at
            # n=40 a single draw is mostly luck, and the union over draws would
            # just reintroduce the sample-size bias this cap exists to remove.
            sets = {k: [] for k in ks}
            for rep in range(reps):
                idx = rng_pick(len(src), cap, SEED + 7919 * rep)
                counts = code_carriers([src[i] for i in idx])
                for k in ks:
                    sets[k].append(emitted(counts, pool, k))
            for k in ks:
                arms[name][k][h] = sets[k]
        note = "one global generator" if shared else "per-hospital generators"
        print(f"  {name:16s} {note}, capped to each site's real split")

    # --- controls --------------------------------------------------------- #
    pooled_train = {p: v for pats in real_train.values() for p, v in pats.items()}
    pooled_counts = traj_carriers(pooled_train)
    # Both controls are drawn at the SAME per-hospital budget as every arm --
    # a ceiling computed from all 8,509 pooled patients while the arms get 40
    # is not a ceiling, it is the sample-size artefact again.
    pooled_ids = sorted(pooled_train)
    arms["real_pooled"] = {k: {} for k in ks}
    arms["prior"] = {k: {} for k in ks}
    for h in hosp:
        cap = budget_for(h)
        ceil_sets = {k: [] for k in ks}
        floor_sets = {k: [] for k in ks}
        for rep in range(reps):
            idx = rng_pick(len(pooled_ids), cap, SEED + 104729 * rep)
            sub = {pooled_ids[int(i)]: pooled_train[pooled_ids[int(i)]]
                   for i in idx}
            counts = traj_carriers(sub)
            for k in ks:
                ceil_sets[k].append(emitted(counts, pool, k))
                floor_sets[k].append(prior_draw(
                    pooled_counts, len(pooled_train), cap, pool, k,
                    SEED + 7717 * rep))
        for k in ks:
            arms["real_pooled"][k][h] = ceil_sets[k]
            arms["prior"][k][h] = floor_sets[k]
    print(f"  {'real_pooled':16s} ceiling: real patients drawn from the pooled "
          f"train split, same budget")
    print(f"  {'prior':16s} floor: independent codes at pooled rates, "
          f"same budget")

    if "local" not in arms:
        raise SystemExit("this analysis is defined against the local-only arm; "
                         "pass --run local=<its save dir>")

    # --- the table --------------------------------------------------------- #
    strict = set(manifest.get("global_rare_codes", []))
    results = {"cohort": manifest.get("cohort_name"), "fold": args.fold,
               "code_pool": args.code_pool, "n_pool": len(pool),
               "budget": args.budget, "min_carriers": ks,
               "n_train": n_train, "rows": []}

    for k in ks:
        for h in order:
            R = {c for visits in real_eval[h].values()
                 for v in visits for c in v} & pool
            T = {c for visits in real_train[h].values()
                 for v in visits for c in v} & pool
            L_sets = arms["local"][k][h]
            unlearnable = R - T          # absent from h's own training split
            for name in arms:
                if name == "local":
                    continue
                Xs = arms[name][k][h]

                def frac(num, den):
                    return len(num) / len(den) if den else float("nan")

                def avg(fn):
                    vals = [fn(X) for X in Xs]
                    vals = [v for v in vals if v == v]
                    return round(float(np.mean(vals)), 4) if vals else float("nan")

                results["rows"].append({
                    "k": k, "hospital": h, "n_train": n_train[h], "arm": name,
                    "n_R": len(R), "n_L": round(float(np.mean(
                        [len(x) for x in L_sets])), 1),
                    "n_missed": round(float(np.mean(
                        [len(R - x) for x in L_sets])), 1),
                    "n_emitted": round(float(np.mean([len(X) for X in Xs])), 1),
                    # Each resample gets its own `missed`, because `missed`
                    # depends on that draw of the local arm too.
                    "recovered": avg(lambda X: np.mean([
                        frac((R - Lx) & X, R - Lx) for Lx in L_sets])),
                    "recovered_rare": avg(lambda X: np.mean([
                        frac(((R - Lx) & X) & strict, (R - Lx) & strict)
                        for Lx in L_sets])),
                    "recovered_unlearnable": avg(
                        lambda X: frac(unlearnable & X, unlearnable)),
                    "spurious": avg(lambda X: frac(X - R, X)),
                })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}.json")

    for k in ks:
        print(f"\n=== a code counts as emitted at >= {k} carrying patients ===")
        print(f"{'site':>5} {'n_tr':>5} {'|R|':>4} {'|L|':>5} {'miss':>6}  "
              + "".join(f"{a[:13]:>14s}" for a in arms if a != "local"))
        for h in order:
            row = {r["arm"]: r for r in results["rows"]
                   if r["k"] == k and r["hospital"] == h}
            any_r = next(iter(row.values()))
            cells = "".join(
                f"{row[a]['recovered']:>7.2f}/{row[a]['spurious']:<6.2f}"
                for a in arms if a != "local")
            print(f"{h:>5} {any_r['n_train']:5d} {any_r['n_R']:4d} "
                  f"{any_r['n_L']:5.1f} {any_r['n_missed']:6.1f}  {cells}")
        print("   cells are recovered / spurious -- high recovered with high "
              "spurious means the arm emits everything, not that it shared.")

    if not args.no_plot:
        try:
            plot(results, arms, order, ks, args.out + ".png")
        except ImportError:
            print("matplotlib missing -- skipped the figure")


def plot(results, arms, order, ks, path) -> None:
    """Recovered as bars per hospital, with spurious marked on the same axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [a for a in arms if a != "local"]
    fig, axes = plt.subplots(len(ks), 1, figsize=(9.5, 3.6 * len(ks)),
                             squeeze=False)
    width = 0.8 / len(names)
    for ai, k in enumerate(ks):
        ax = axes[ai][0]
        xs = np.arange(len(order))
        for j, name in enumerate(names):
            rec = [next(r["recovered"] for r in results["rows"]
                        if r["k"] == k and r["hospital"] == h and r["arm"] == name)
                   for h in order]
            spu = [next(r["spurious"] for r in results["rows"]
                        if r["k"] == k and r["hospital"] == h and r["arm"] == name)
                   for h in order]
            pos = xs - 0.4 + width * (j + 0.5)
            ax.bar(pos, rec, width * 0.9, label=name)
            ax.scatter(pos, spu, marker="_", s=90, color="#15191E", zorder=3,
                       label="spurious" if j == 0 else None)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{h}\n{results['n_train'][h]}" for h in order],
                           fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("recovered (bar) / spurious (tick)", fontsize=9)
        ax.set_title(f"codes local-only missed, recovered by each arm "
                     f"(>= {k} carriers)", fontsize=10)
        if ai == 0:
            ax.legend(fontsize=7.5, ncol=len(names) + 1, loc="upper right")
    axes[-1][0].set_xlabel("hospital, ordered by training patients", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
