"""Do the eight hospitals actually hold different patients? A UMAP of the cohort.

Every patient in the cohort becomes one point. The representation is the
patient's own record and nothing else -- a multi-hot vector over the cohort
vocabulary, 1 where the patient holds that code in any visit -- so what the
picture shows is heterogeneity in the DATA, not an artefact of a trained
generator. Jaccard is the metric, since these are sparse binary sets.

WHY SMALL MULTIPLES AND NOT EIGHT COLOURS. Eight categorical hues cannot be
told apart in a scatter. On the reference palette the worst pair sits at
normal-vision delta-E 7.1 against a floor of 15, and 3.2 under protanopia --
i.e. two of the eight are indistinguishable to a reader with full colour
vision, before any overplotting. Each hospital therefore gets its own panel,
drawn over the whole cohort in grey, which also keeps the dense centre legible.
``--style overlay`` gives the single-panel version anyway if a talk needs it.

DO NOT PICK THE TWO MOST DIFFERENT HOSPITALS BY EYE. UMAP preserves local
neighbourhoods; distances BETWEEN clusters are not meaningful, and a gap on
screen can be an artefact of the layout. The divergence matrix this script also
writes is the thing to read for that -- it compares per-code prevalence
directly, in the original space:

    jsd  Jensen-Shannon distance between the hospitals' normalised code
         frequency distributions. Composition: which codes, in what mix.
    l1   mean absolute difference in per-code prevalence, unnormalised.
         Rate: how often codes are recorded at all.

They answer different questions and can rank pairs differently. A hospital that
documents everything twice as thoroughly moves `l1` and barely moves `jsd`.

Example:
    .venv/bin/python examples/fedpyhealth/scripts/fig_hospital_umap.py \\
        --out _outputs/results/hospital_umap.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cohort import DEFAULT_CACHE_DIR, hospitals, load_manifest, read_trajectories

#: Slot 1 of the reference categorical palette. One accent is all a small-
#: multiple needs -- identity comes from the panel, not the hue.
ACCENT = "#2a78d6"
GHOST = "#DCDEE0"
INK = "#1A1A19"
MUTED = "#6E7377"

#: Only used by --style overlay, which the validator FAILS at eight slots.
OVERLAY_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def build_matrix(cache_dir, folds):
    """(X, hosp_per_row, vocab) -- one multi-hot row per patient."""
    import numpy as np

    per_hosp = {}
    for fold in folds:
        traj = read_trajectories(cache_dir, fold)
        for hid, patients in traj.items():
            bag = per_hosp.setdefault(hid, [])
            for pid, visits in patients.items():
                bag.append({c for visit in visits for c in visit})

    vocab = sorted({c for bags in per_hosp.values() for b in bags for c in b})
    index = {c: i for i, c in enumerate(vocab)}
    hids = sorted(per_hosp, key=lambda h: -len(per_hosp[h]))

    rows, labels = [], []
    for hid in hids:
        for codes in per_hosp[hid]:
            v = np.zeros(len(vocab), dtype=np.float32)
            for c in codes:
                v[index[c]] = 1.0
            rows.append(v)
            labels.append(hid)
    return np.asarray(rows), np.asarray(labels), vocab, hids


def self_divergence(X, labels, hids, seed=0, reps=20):
    """Noise floor: JSD between two random halves of a hospital's OWN patients.

    A small hospital estimates its prevalence vector from few patients, and the
    estimation noise inflates every JSD it appears in. Without this control,
    "the two most different hospitals" can read out "the two smallest". Any
    between-hospital JSD must clear the two sites' own floors to mean anything.
    """
    import numpy as np
    from scipy.spatial.distance import jensenshannon

    rng = np.random.default_rng(seed)
    out = {}
    for h in hids:
        rows = np.flatnonzero(labels == h)
        vals = []
        for _ in range(reps):
            perm = rng.permutation(rows)
            a = X[perm[: len(perm) // 2]].mean(axis=0)
            b = X[perm[len(perm) // 2:]].mean(axis=0)
            if a.sum() and b.sum():
                vals.append(float(jensenshannon(a / a.sum(), b / b.sum(), base=2)))
        out[h] = float(np.mean(vals)) if vals else float("nan")
    return out


def divergences(X, labels, hids):
    """Pairwise jsd / l1 between hospitals' per-code prevalence vectors."""
    import numpy as np
    from scipy.spatial.distance import jensenshannon

    prev = {h: X[labels == h].mean(axis=0) for h in hids}
    out = []
    for i, a in enumerate(hids):
        for b in hids[i + 1:]:
            pa, pb = prev[a], prev[b]
            sa, sb = pa.sum(), pb.sum()
            jsd = float(jensenshannon(pa / sa, pb / sb, base=2)) if sa and sb else float("nan")
            out.append({"a": a, "b": b, "jsd": jsd,
                        "l1": float(np.abs(pa - pb).mean())})
    return out, {h: prev[h] for h in hids}


def project(X, method, seed, metric, svd):
    """Optionally reduce first, then embed in 2D.

    WHY SVD FIRST. Patients hold ~3.6 distinct codes out of 737, so most pairs
    share none and their Jaccard distance is exactly 1. UMAP then disconnects
    those points from the manifold and returns NaN coordinates for them (171
    vertices, on the first run). Projecting onto the leading singular vectors
    of the patient-by-code matrix first -- latent semantic analysis, the usual
    treatment for sparse count data -- gives every patient a dense coordinate,
    so cosine distance is informative and nothing disconnects. ``--svd 0``
    keeps the raw binary space if you want the jaccard version back.
    """
    import numpy as np

    if svd and svd < X.shape[1]:
        from sklearn.decomposition import TruncatedSVD
        red = TruncatedSVD(n_components=svd, random_state=seed)
        X = red.fit_transform(X)
        print(f"  SVD to {svd} dims, {red.explained_variance_ratio_.sum():.1%} "
              f"of variance retained")

    if method == "umap":
        try:
            import umap
        except ImportError:
            raise SystemExit(
                "umap-learn is not installed in this environment.\n"
                "  .venv/bin/pip install umap-learn\n"
                "or re-run with --method tsne (scikit-learn, already present)."
            )
        red = umap.UMAP(n_components=2, metric=metric, random_state=seed,
                        n_neighbors=30, min_dist=0.1)
        return red.fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric=metric, init="pca",
                    random_state=seed, perplexity=30).fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR)
    p.add_argument("--folds", default="train,val,test",
                   help="comma-separated; all three by default, so the picture "
                        "shows every patient in the cohort")
    p.add_argument("--method", default="umap", choices=["umap", "tsne", "pca"])
    p.add_argument("--metric", default="cosine",
                   help="cosine after the SVD reduction (default). Use "
                        "jaccard with --svd 0 to work in the raw binary space, "
                        "but expect disconnected points at this sparsity.")
    p.add_argument("--svd", type=int, default=50,
                   help="reduce to this many dims before embedding; 0 to skip")
    p.add_argument("--style", default="facet", choices=["facet", "overlay"],
                   help="facet: one panel per hospital (default, and the only "
                        "one that survives a colour-vision check). overlay: "
                        "all eight in one panel, eight hues.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--out", default="_outputs/results/hospital_umap.png")
    a = p.parse_args(argv)

    import numpy as np
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]

    man = load_manifest(a.cohort_cache)
    print(f"cohort {man.get('cohort_name')!r}, folds {folds}")
    X, labels, vocab, hids = build_matrix(a.cohort_cache, folds)
    print(f"{X.shape[0]} patients x {X.shape[1]} codes; "
          f"{X.sum(axis=1).mean():.1f} codes per patient on average")

    floor = self_divergence(X, labels, hids, seed=a.seed)
    div, prev = divergences(X, labels, hids)
    print("\nwithin-hospital noise floor (jsd between two halves of its own "
          "patients):")
    print(f"  {'hospital':<12}{'n':>7}{'self jsd':>11}")
    for h in hids:
        print(f"  {h:<12}{int((labels == h).sum()):>7,}{floor[h]:>11.4f}")

    look = {(d["a"], d["b"]): d for d in div}
    look.update({(d["b"], d["a"]): d for d in div})
    for key, lab in (("jsd", "JSD (composition: which codes, in what mix)"),
                     ("l1", "L1 (rate: mean abs difference in code prevalence)")):
        print(f"\nfull {len(hids)}x{len(hids)} matrix -- {lab}")
        print("        " + "".join(f"{h:>9}" for h in hids))
        for ra in hids:
            cells = []
            for rb in hids:
                if ra == rb:
                    cells.append(f"{'.':>9}")
                else:
                    v = look[(ra, rb)][key]
                    cells.append(f"{v:>9.4f}" if key == "jsd" else f"{v:>9.5f}")
            print(f"  {ra:<6}" + "".join(cells))

    print(f"\nall {len(div)} pairs, ranked by excess JSD over the noise floor")
    print(f"  {'pair':<14}{'jsd':>8}{'floor':>8}{'excess':>9}{'l1':>10}")
    for d in sorted(div, key=lambda d: -(d["jsd"] - max(floor[d["a"]], floor[d["b"]]))):
        fl = max(floor[d["a"]], floor[d["b"]])
        print(f"  {d['a']+' vs '+d['b']:<14}{d['jsd']:>8.4f}{fl:>8.4f}"
              f"{d['jsd'] - fl:>9.4f}{d['l1']:>10.5f}")

    print(f"\nprojecting with {a.method} ({a.metric}) ...")
    Z = project(X, a.method, a.seed, a.metric, a.svd)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(Z).all(axis=1)
    if not ok.all():
        print(f"  dropping {int((~ok).sum())} patients with no finite "
              f"coordinate (disconnected from the manifold)")
        Z, labels = Z[ok], labels[ok]
    lo = Z.min(axis=0) - 0.5
    hi = Z.max(axis=0) + 0.5
    name = {"umap": "UMAP", "tsne": "t-SNE", "pca": "PCA"}[a.method]

    def dress(ax):
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#E6E9EC")

    if a.style == "facet":
        fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0))
        for ax, hid in zip(axes.ravel(), hids):
            m = labels == hid
            ax.scatter(Z[~m, 0], Z[~m, 1], s=a.point_size * 0.7, c=GHOST,
                       linewidths=0, rasterized=True)
            ax.scatter(Z[m, 0], Z[m, 1], s=a.point_size, c=ACCENT,
                       linewidths=0, rasterized=True)
            ax.set_title(f"hospital {hid}", fontsize=10.5, color=INK, pad=4)
            ax.text(0.5, -0.045, f"{int(m.sum()):,} patients", fontsize=8.5,
                    color=MUTED, ha="center", va="top", transform=ax.transAxes)
            dress(ax)
        fig.suptitle(f"{name} of every patient in the cohort, one panel per "
                     f"hospital", fontsize=12.5, color=INK, y=0.975)
        fig.text(0.5, 0.012, "each panel highlights one hospital against the "
                 "whole cohort in grey; axes are UMAP coordinates and carry no "
                 "units", fontsize=8.5, color=MUTED, ha="center")
        fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    else:
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        for hid, hue in zip(hids, OVERLAY_HUES):
            m = labels == hid
            ax.scatter(Z[m, 0], Z[m, 1], s=a.point_size, c=hue, linewidths=0,
                       rasterized=True, label=f"{hid} (n={int(m.sum()):,})")
        dress(ax)
        ax.set_title(f"{name} of every patient in the cohort", fontsize=12,
                     color=INK)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                  fontsize=9, markerscale=3)
        fig.tight_layout()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    coords = a.out.rsplit(".", 1)[0] + "_coords.json"
    with open(coords, "w") as fh:
        # Rounded to 3dp: the artifact plots these at a few hundred pixels, so
        # more precision is bytes on the wire and nothing on screen.
        json.dump({"cohort": man.get("cohort_name"),
                   "hospitals": [{"id": h, "n": int((labels == h).sum())}
                                 for h in hids],
                   "self_jsd_floor": floor,
                   "pairs": sorted(div, key=lambda d: -d["jsd"]),
                   "labels": labels.tolist(),
                   "x": [round(float(v), 3) for v in Z[:, 0]],
                   "y": [round(float(v), 3) for v in Z[:, 1]]}, fh)
    print(f"wrote {coords}")

    side = a.out.rsplit(".", 1)[0] + "_divergence.json"
    with open(side, "w") as fh:
        json.dump({"cohort": man.get("cohort_name"), "folds": folds,
                   "n_patients": int(X.shape[0]), "vocab": len(vocab), "self_jsd_floor": floor,
                   "pairs": sorted(div, key=lambda d: -d["jsd"])}, fh, indent=1)
    print(f"\nwrote {a.out}\nwrote {side}")


if __name__ == "__main__":
    main()
