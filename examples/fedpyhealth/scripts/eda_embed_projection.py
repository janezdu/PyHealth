"""Where do a hospital's synthetic patients sit, relative to its real ones?

Every other analysis in this project summarises a generator by a statistic --
a prevalence correlation, a coverage fraction. This one draws the patients.

Each patient, real or synthetic, becomes a binary vector over the scored code
pool: 1 if the patient carries the code anywhere in their trajectory. Every
point is projected to 2D by a SINGLE transform fitted on real data, then drawn
as a grid: one ROW per hospital, one COLUMN per training regime, with that
hospital's real test patients overlaid in every panel of its row.

The question it answers, reading down a column:

    local-only at 429 is trained on 142 patients; at 458 on 1,832. FedAvg
    averages them together. Did that move each site's generated patients TOWARD
    what that site actually looks like, or just toward the cohort average?

WHICH POINTS THE AXES ARE FITTED ON, and why it is not the plotted ones.
An earlier version fitted the PCA on the pooled stack of every plotted point.
That makes arms comparable INSIDE one figure, but the basis is then a function
of the arm list: adding a fourth arm rotated the axes and one site's 40 real
patients went from a single blob to two visibly separated clusters, unchanged
data and all. So the default basis is ``--fit-on real_pooled`` -- every
hospital's real train split, 8,509 patients that no arm or hospital choice can
move. The axes mean "directions of variation among real eICU patients", the
same frame in every panel, every row, and every figure. That shared frame is
also what makes ROWS comparable to each other, not just columns.

WHAT THIS PLOT CAN AND CANNOT SHOW. A 2D projection of a 478-dimensional
binary space keeps only the leading variance; two clouds that overlap here are
not necessarily the same distribution. It is an orientation device. The three
numbers printed per panel -- mean codes per patient, centroid distance, and
nearest-neighbour Jaccard -- are computed in the FULL space and are what should
be quoted. Read the numbers, use the picture to understand them.

THE SAMPLE-SIZE TRAP, and why it does NOT apply here. eda_code_coverage.py
caps every arm at the real split's size because the number of DISTINCT codes in
a patient set grows with the number of patients, so an arm allowed more draws
wins for free. Nothing here is a set statistic: a cloud's shape, its centroid,
and its per-patient code count are all budget-free. So synthetic arms are drawn
at ``--n-synth`` (default 400) for a readable cloud even where the real fold
holds only 40 patients. ``--match-real`` restores the cap.

Runs on CPU in a couple of minutes. No model is loaded -- every arm's
``synthetic.json`` is already on disk.

Example:
    python examples/fedpyhealth/scripts/eda_embed_projection.py \\
        --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \\
        --hospitals 429,458 \\
        --run local=_outputs/local_full_hilo8_random_nes_save \\
        --run fedavg=_outputs/fedavg_full_hilo8_random_nes_save \\
        --out _outputs/results/eda_embed_grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.cohort import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    load_manifest,
    load_shared_generator,
    load_synthetic,
    read_trajectories,
    scored_codes,
)

SEED = 0

#: Ordinal ramp, isolated -> fully pooled, matching viz/README.md's design
#: note. There is no rare/common distinction among PATIENTS, so nothing here
#: earns a categorical hue -- the regimes have a natural order and the ramp is
#: that order. Real data is deliberately outside the ramp: it is the target,
#: not an arm.
COLORS = {
    "real": "#1c1b19",
    "local": "#a3c8d4",
    "fedavg": "#5595aa",
    "fedavg_ft": "#5595aa",
    "fedavg_irm": "#2f7186",
    "fedavg_irm_xm": "#2f7186",
    "centralized": "#123f4e",
}
#: The fedavg variants are not further steps along the isolated->pooled axis,
#: so they keep the ramp's middle tone and are told apart by marker instead of
#: by a hue that would imply more pooling.
MARKERS = {"fedavg_irm": "^", "fedavg_irm_xm": "D", "fedavg_ft": "s"}
FALLBACK = "#5595aa"


def patient_vectors(patients: Sequence, pool: List[str],
                    key: str = "visits") -> np.ndarray:
    """Binary patient x code matrix over ``pool``.

    Args:
        patients: Either a list of ``{key: [[code, ...], ...]}`` dicts (the
            synthetic shape) or a list of visit-lists (the cache shape).
        pool: Ordered code vocabulary; columns follow this order.
        key: Field holding the visit list, for the dict shape.

    Returns:
        ``(n_patients, len(pool))`` float32 array of 0/1.
    """
    index = {c: i for i, c in enumerate(pool)}
    out = np.zeros((len(patients), len(pool)), dtype=np.float32)
    for row, p in enumerate(patients):
        visits = p[key] if isinstance(p, dict) else p
        for visit in visits:
            for code in visit:
                col = index.get(code)
                if col is not None:
                    out[row, col] = 1.0
    return out


def jaccard_to_nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of ``a``, the best Jaccard similarity against any row of ``b``.

    Answers "is there a real patient that looks like this synthetic one?". A
    generator can match the real centroid perfectly while producing patients
    that individually resemble nobody, and the centroid distance alone would
    not notice.
    """
    inter = a @ b.T
    sizes_a = a.sum(axis=1, keepdims=True)
    sizes_b = b.sum(axis=1, keepdims=True).T
    union = sizes_a + sizes_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.max(axis=1)


def fit_basis(stack: np.ndarray, method: str, seed: int,
              basis: np.ndarray = None):
    """Fit the 2D transform once; the caller applies it to every cloud.

    Fitting per arm, or per hospital, would centre each cloud on its own mean,
    which is exactly the difference the plot exists to show.

    Args:
        stack: Every point that will be plotted; used when ``basis`` is None.
        method: pca, tsne or umap.
        seed: Random state.
        basis: Rows to fit on. ``None`` fits on ``stack``.

    Returns:
        A callable mapping an ``(n, d)`` matrix to ``(n, 2)``.
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=2, random_state=seed)
        model.fit(stack if basis is None else basis)
        return model.transform
    # Neither t-SNE nor UMAP has an out-of-sample transform, so both must see
    # every point at fit time and cannot honour a separate basis.
    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        pre = PCA(n_components=min(50, stack.shape[1]),
                  random_state=seed).fit_transform(stack)
        out = TSNE(n_components=2, random_state=seed, init="pca",
                   perplexity=min(30, max(5, len(stack) // 5 - 1))
                   ).fit_transform(pre)
        return _replay(out)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "--method umap needs umap-learn, which is not in this "
                "environment. Use --method pca (no extra dependency) or "
                "--method tsne."
            ) from exc
        model = umap.UMAP(n_components=2, random_state=seed, metric="jaccard")
        return model.fit(stack).transform
    raise ValueError(f"unknown --method {method!r}")


def _replay(out: np.ndarray):
    """Hand back a fitted-once embedding in row order, for t-SNE.

    t-SNE cannot transform unseen points, so the caller MUST request clouds in
    exactly the order they were stacked. Guarded rather than trusted.
    """
    cursor = {"at": 0}

    def transform(mat: np.ndarray) -> np.ndarray:
        start = cursor["at"]
        stop = start + mat.shape[0]
        if stop > out.shape[0]:
            raise RuntimeError(
                "t-SNE replay ran past the fitted points: clouds were "
                "requested in a different order than they were stacked."
            )
        cursor["at"] = stop
        return out[start:stop]

    return transform


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR,
                   help="Frozen cohort cache holding manifest.json and folds.")
    p.add_argument("--hospitals", default="429",
                   help="Comma-separated site ids, one figure ROW each, drawn "
                        "in the order given. e.g. 429,458")
    p.add_argument("--run", action="append", default=[], metavar="NAME=SAVE_DIR",
                   help="Arm name and its save dir -- one figure COLUMN each, "
                        "in the order given. Repeatable.")
    p.add_argument("--fold", default="test",
                   help="Real fold to use as the target cloud.")
    p.add_argument("--code-pool", default="scored",
                   choices=["scored", "rare", "all"],
                   help="scored: the pool Test 2 reports on. rare: the strict "
                        "global_rare subset. all: every code in the fold.")
    p.add_argument("--n-synth", type=int, default=400,
                   help="Synthetic patients drawn per arm per site. Cloud "
                        "shape is budget-free, so this is a readability knob.")
    p.add_argument("--match-real", action="store_true",
                   help="Cap every arm at that site's real split size instead.")
    p.add_argument("--fit-on", default="real_pooled",
                   choices=["real_pooled", "real", "stack"],
                   help="What the 2D transform is FITTED on. real_pooled (the "
                        "default) uses every hospital's real train split, so "
                        "the axes are fixed properties of real eICU patients "
                        "and move for no arm or hospital. real uses the first "
                        "listed site's real fold. stack fits on the plotted "
                        "points -- comparable within one figure, but the basis "
                        "rotates whenever the arm or hospital list changes.")
    p.add_argument("--method", default="pca", choices=["pca", "tsne", "umap"],
                   help="2D transform. pca is linear, needs no extra "
                        "dependency, and is the only one with an out-of-sample "
                        "transform, so --fit-on only bites for pca.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default="_outputs/results/eda_embed",
                   help="Output prefix; writes <out>.json and <out>.png.")
    p.add_argument("--no-plot", action="store_true", help="Write the JSON only.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    sites = [h.strip() for h in args.hospitals.split(",") if h.strip()]
    if not sites:
        raise SystemExit("--hospitals is empty")
    if not args.run:
        raise SystemExit("nothing to plot: pass at least one --run NAME=SAVE_DIR")

    manifest = load_manifest(args.cohort_cache)
    n_train = {h: v["n_train"] for h, v in manifest["per_hospital"].items()}

    if args.code_pool == "scored":
        pool = scored_codes(manifest, args.fold)
    elif args.code_pool == "rare":
        pool = sorted(set(manifest.get("global_rare_codes", [])))
    else:
        traj = read_trajectories(args.cohort_cache, args.fold, only=sites)
        seen: Set[str] = {c for pats in traj.values() for v in pats.values()
                          for visit in v for c in visit}
        pool = sorted(seen)
    if not pool:
        raise SystemExit(f"empty code pool for --code-pool {args.code_pool!r}")

    # Load each arm's synthetic file ONCE, not once per hospital: centralized's
    # is ~195MB and re-reading it per row is the slowest thing here by far.
    arms: List[str] = []
    synth: Dict[str, dict] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run wants NAME=SAVE_DIR, got {spec!r}")
        name, save_dir = spec.split("=", 1)
        per = load_synthetic(save_dir)
        shared = load_shared_generator(save_dir)
        synth[name] = {"per": per, "shared": shared,
                       "key0": sorted(per)[0] if per else None}
        arms.append(name)
        note = "one global generator" if shared else "per-hospital generators"
        print(f"  {name:16s} {note}")

    real = read_trajectories(args.cohort_cache, args.fold, only=sites)
    print(f"\n{len(pool)} codes in the {args.code_pool} pool; "
          f"{len(sites)} rows x {len(arms)} columns")

    clouds: Dict[str, Dict[str, np.ndarray]] = {}
    for hid in sites:
        if hid not in real:
            raise SystemExit(f"hospital {hid} is not in this cohort")
        ids = sorted(real[hid])
        row: Dict[str, np.ndarray] = {
            "real": patient_vectors([real[hid][p] for p in ids], pool)}
        cap = len(ids) if args.match_real else args.n_synth
        for name in arms:
            blob = synth[name]
            src = blob["per"][blob["key0"]] if blob["shared"] \
                else blob["per"].get(hid, [])
            if not src:
                raise SystemExit(
                    f"{name}: no synthetic patients for hospital {hid}")
            take = min(cap, len(src))
            # Seeded from (arm, hospital), not a shared stream: with one rng,
            # inserting an arm silently redraws every later one and its
            # statistics move for no reason.
            rng = np.random.default_rng(
                [args.seed, abs(hash(name)) % (2 ** 31), int(hid)])
            idx = rng.choice(len(src), size=take, replace=False)
            row[name] = patient_vectors([src[i] for i in idx], pool)
        clouds[hid] = row
        print(f"  site {hid:>5}: {len(ids):>4} real {args.fold} patients, "
              f"{n_train.get(hid, '?'):>5} train, {cap} synthetic per arm")

    order = ["real"] + arms
    stack = np.vstack([clouds[h][n] for h in sites for n in order])
    basis = None
    if args.fit_on == "real":
        basis = clouds[sites[0]]["real"]
    elif args.fit_on == "real_pooled":
        pooled = read_trajectories(args.cohort_cache, "train")
        rows = [v for pats in pooled.values() for v in pats.values()]
        basis = patient_vectors(rows, pool)
        print(f"\nprojection fitted on {basis.shape[0]} real pooled train "
              f"patients -- one fixed basis for every row and column")
    if args.method != "pca" and basis is not None:
        print(f"note: --method {args.method} has no out-of-sample transform, "
              f"so it fits on the plotted points regardless of --fit-on")
        basis = None
    transform = fit_basis(stack, args.method, args.seed, basis)

    payload = {
        "cohort": os.path.basename(args.cohort_cache.rstrip("/")),
        "hospitals": sites,
        "arms": arms,
        "n_train": {h: n_train.get(h) for h in sites},
        "fold": args.fold,
        "code_pool": args.code_pool,
        "n_pool": len(pool),
        "method": args.method,
        "fit_on": args.fit_on,
        "n_synth": args.n_synth,
        "match_real": bool(args.match_real),
        "stats": {},
        "coords": {},
    }

    # Same (hospital, arm) order the stack was built in -- t-SNE replays by
    # position and would silently misalign otherwise.
    for hid in sites:
        rowc = clouds[hid]
        real_mat = rowc["real"]
        real_centroid = real_mat.mean(axis=0)
        payload["stats"][hid] = {}
        payload["coords"][hid] = {}
        for name in order:
            mat = rowc[name]
            entry = {
                "n": int(mat.shape[0]),
                "codes_per_patient": float(mat.sum(axis=1).mean()),
                "distinct_codes": int((mat.sum(axis=0) > 0).sum()),
                "centroid_dist": float(
                    np.linalg.norm(mat.mean(axis=0) - real_centroid)),
            }
            if name == "real":
                # Leave-one-out, or every real patient is its own neighbour.
                sims = [jaccard_to_nearest(mat[i:i + 1],
                                           np.delete(mat, i, axis=0))[0]
                        for i in range(mat.shape[0])]
                entry["nn_jaccard_to_real"] = (
                    float(np.mean(sims)) if sims else 0.0)
            else:
                entry["nn_jaccard_to_real"] = float(
                    jaccard_to_nearest(mat, real_mat).mean())
            payload["stats"][hid][name] = entry
            payload["coords"][hid][name] = transform(mat).round(4).tolist()

    report(payload)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}.json")

    if not args.no_plot:
        plot(payload, f"{args.out}.png")
        print(f"wrote {args.out}.png")


def report(payload: dict) -> None:
    """One table per hospital, full-space numbers only."""
    names = ["real"] + list(payload["arms"])
    width = max(len(n) for n in names) + 2
    for hid in payload["hospitals"]:
        print(f"\nhospital {hid}  (n_train {payload['n_train'][hid]})")
        print(f"{'arm':<{width}}{'n':>6}{'codes/pt':>10}{'distinct':>10}"
              f"{'cent.dist':>11}{'NN Jacc':>9}")
        for name in names:
            s = payload["stats"][hid][name]
            print(f"{name:<{width}}{s['n']:>6}{s['codes_per_patient']:>10.2f}"
                  f"{s['distinct_codes']:>10}{s['centroid_dist']:>11.3f}"
                  f"{s['nn_jaccard_to_real']:>9.3f}")
    print("\ncodes/pt against real's is the hedging number; centroid dist and "
          "NN Jaccard\nare distance to the real cloud -- lower and higher are "
          "better respectively.\nEach real row's NN Jaccard is leave-one-out: "
          "that site's ceiling, not a target.")


def plot(payload: dict, path: str) -> None:
    """Grid: one row per hospital, one column per arm, shared axes throughout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites: Sequence[str] = payload["hospitals"]
    arms: Sequence[str] = payload["arms"]
    coords = {h: {n: np.asarray(v) for n, v in payload["coords"][h].items()}
              for h in sites}

    every = np.vstack([c for rowc in coords.values() for c in rowc.values()])
    pad = 0.05 * (every.max(axis=0) - every.min(axis=0) + 1e-9)
    lo, hi = every.min(axis=0) - pad, every.max(axis=0) + pad

    nrow, ncol = len(sites), len(arms)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             sharey=True,
                             figsize=(3.15 * ncol, 3.25 * nrow + 0.55))

    for r, hid in enumerate(sites):
        real = coords[hid]["real"]
        for c, name in enumerate(arms):
            ax = axes[r][c]
            ax.scatter(coords[hid][name][:, 0], coords[hid][name][:, 1], s=12,
                       c=COLORS.get(name, FALLBACK), alpha=0.55,
                       marker=MARKERS.get(name, "o"), edgecolors="none",
                       zorder=2, label=name)
            ax.scatter(real[:, 0], real[:, 1], s=24, c=COLORS["real"],
                       marker="x", linewidths=1.0, zorder=3,
                       label=f"real ({payload['fold']})")
            s = payload["stats"][hid][name]
            # Per-panel stats go inside the axes: with a grid this size,
            # per-panel titles would collide with the row above.
            ax.text(0.03, 0.97,
                    f"{s['codes_per_patient']:.2f} codes/pt\n"
                    f"NN Jacc {s['nn_jaccard_to_real']:.3f}\n"
                    f"cent {s['centroid_dist']:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                    color="#6b6862", linespacing=1.35)
            if r == 0:
                ax.set_title(name, fontsize=11, fontweight="semibold", pad=8)
            if c == 0:
                rs = payload["stats"][hid]["real"]
                ax.set_ylabel(
                    f"hospital {hid}\n{payload['n_train'][hid]} train · "
                    f"real {rs['codes_per_patient']:.2f} codes/pt",
                    fontsize=9.5)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color("#dedbd4")

    axes[0][ncol - 1].legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    fig.suptitle(
        f"Synthetic patients vs real — {payload['method'].upper()} of "
        f"{payload['n_pool']} {payload['code_pool']} codes, "
        f"{payload['cohort']}",
        fontsize=12.5, y=0.995)
    fig.text(0.5, 0.008,
             "One dot = one patient, binary over the code pool. Every panel "
             f"shares one projection fitted on {payload['fit_on']}, so rows "
             "and columns are both comparable. 2D keeps only the leading "
             "variance — quote the numbers, not the overlap.",
             ha="center", fontsize=8, color="#6b6862", wrap=True)
    fig.tight_layout(rect=[0, 0.028, 1, 0.965])
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
