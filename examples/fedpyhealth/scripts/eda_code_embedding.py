"""Does federation rearrange the model's internal map of the code vocabulary?

The companion to ``eda_embed_projection.py``. That script draws the patients a
generator EMITS; this one draws the geometry it LEARNT. HALO's first layer is
``vis_embed_mat``, an ``nn.Linear(total_vocab_size, n_embd, bias=False)``, so
column ``j`` of its weight is a 256-number vector for code ``j`` -- the model's
sense of what that code is. Codes the model treats as interchangeable end up
near each other.

    local-only at hospital 429 saw 142 patients. fedavg averaged its weights
    with seven other sites. Did that change WHICH CODES THE MODEL THINKS ARE
    ALIKE, or only how often it emits them?

THE ALIGNMENT PROBLEM, which is why this script reports two different things.
Each arm was trained from its own random initialisation. An embedding space is
only defined up to rotation, so the raw coordinates of arm A and arm B are not
comparable -- code 250 sitting at (3, -1) in one and (-2, 2) in the other means
nothing at all. Two honest responses, both produced here:

1. **Rotation-invariant comparison (the number to quote).** Build each arm's
   code-by-code cosine similarity matrix and correlate the matrices across
   arms. Similarity between codes survives rotation, so this measures whether
   the arms agree on which codes are alike WITHOUT any alignment step. It is
   the headline in ``similarity_agreement``.

2. **Per-arm projections (the picture).** Each panel is that arm's own PCA,
   fitted on its own embeddings. Panels are NOT in a shared coordinate system,
   so compare STRUCTURE between panels -- how tight the rare codes cluster,
   whether they separate from common ones -- never position.

``--align procrustes`` adds an overlay panel that rotates every arm onto a
reference and draws per-code drift arrows. It is off by default and should be
read with suspicion: orthogonal Procrustes finds the best rotation whether or
not one exists, so a tidy-looking overlay is not evidence of anything on its
own. Check ``procrustes_residual`` before believing the arrows.

CPU-only, a minute or two. Loads three checkpoints with ``map_location="cpu"``
and touches no GPU.

Example:
    python examples/fedpyhealth/scripts/eda_code_embedding.py \\
        --cohort-cache "$FEDCOHORT_CACHE/hilo8_random" \\
        --hospital 429 \\
        --ckpt local=_outputs/local_full_hilo8_random_nes_save/local_429.pt \\
        --ckpt fedavg=_outputs/fedavg_full_hilo8_random_nes_save/fedavg_state.pt \\
        --ckpt centralized=_outputs/centralized_full_hilo8_random_nes_save/\
            centralized_state.pt \\
        --out _outputs/results/eda_code_embed_429
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.cohort import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    load_manifest,
    load_processor,
    read_trajectories,
    scored_codes,
)

SEED = 0

#: Where HALO keeps its per-code vectors. ``nn.Linear`` stores weight as
#: (out_features, in_features), so this is (n_embd, total_vocab_size) and a
#: CODE is a COLUMN. Transposed on load.
EMBED_KEY = "halo_model.transformer.vis_embed_mat.weight"

COLORS = {
    "local": "#cfe0e6",
    "fedavg": "#4e8fa5",
    "fedavg_irm": "#7b5ea7",
    "fedavg_ft": "#8fbccb",
    "centralized": "#245b6e",
}
FALLBACK = "#b8563e"
#: Code groups, drawn as marker styles inside every panel.
GROUP_STYLE = {
    "needed_rare": ("#2f6f7e", "o", 18, 0.85),
    "needed_common": ("#b8563e", "o", 14, 0.55),
    "other": ("#9a968e", ".", 6, 0.30),
}


def load_embeddings(path: str) -> np.ndarray:
    """Pull the code-embedding matrix out of a run checkpoint.

    Returns:
        ``(total_vocab_size, n_embd)`` float64 -- one row per vocabulary slot.

    Raises:
        SystemExit: If the file is not a checkpoint this project wrote, with
            the keys it does contain, since a silently-wrong tensor here would
            produce a plausible and meaningless figure.
    """
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "global_state" in blob:
        state = blob["global_state"]
    elif isinstance(blob, dict) and EMBED_KEY in blob:
        state = blob
    else:
        keys = list(blob)[:8] if isinstance(blob, dict) else type(blob).__name__
        raise SystemExit(
            f"{path} is not a checkpoint this script understands. Expected a "
            f"dict with 'global_state' or the raw state_dict; found {keys}."
        )
    if EMBED_KEY not in state:
        raise SystemExit(
            f"{path} has no {EMBED_KEY!r}. Either it predates the current HALO "
            f"layout or it is not a HALO run."
        )
    return state[EMBED_KEY].detach().cpu().numpy().astype(np.float64).T


def cosine_matrix(mat: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, the rotation-invariant view of a space."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.maximum(norms, 1e-12)
    return unit @ unit.T


def upper(mat: np.ndarray) -> np.ndarray:
    """The strict upper triangle, flattened -- each code pair counted once."""
    i, j = np.triu_indices(mat.shape[0], k=1)
    return mat[i, j]


def procrustes(source: np.ndarray, target: np.ndarray):
    """Rotate ``source`` onto ``target`` without scaling or reflection-checking.

    Returns:
        ``(rotated, residual)`` where residual is the remaining Frobenius error
        as a fraction of ``target``'s norm. A residual near 1.0 means the
        rotation achieved nothing and the overlay should not be read.
    """
    u, _, vt = np.linalg.svd(source.T @ target)
    rot = u @ vt
    rotated = source @ rot
    denom = np.linalg.norm(target) or 1.0
    return rotated, float(np.linalg.norm(rotated - target) / denom)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort-cache", default=DEFAULT_CACHE_DIR)
    p.add_argument("--hospital", default="429",
                   help="Site whose real codes are highlighted. The embeddings "
                        "themselves are global; this only sets the colouring.")
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH",
                   help="Arm name and its .pt checkpoint. Repeatable.")
    p.add_argument("--fold", default="test")
    p.add_argument("--code-pool", default="scored",
                   choices=["scored", "rare", "all"])
    p.add_argument("--align", default="none", choices=["none", "procrustes"],
                   help="procrustes adds a drift-arrow overlay. Read the "
                        "residual before trusting it.")
    p.add_argument("--n-arrows", type=int, default=40,
                   help="Codes labelled in the Procrustes overlay, chosen as "
                        "the largest drift among the site's needed rare codes.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default="_outputs/results/eda_code_embed")
    p.add_argument("--no-plot", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    hid = str(args.hospital)
    if not args.ckpt:
        raise SystemExit("nothing to plot: pass at least one --ckpt NAME=PATH")

    manifest = load_manifest(args.cohort_cache)
    proc = load_processor(args.cohort_cache)
    code_vocab: Dict[str, int] = proc.code_vocab

    if args.code_pool == "scored":
        pool = scored_codes(manifest, args.fold)
    elif args.code_pool == "rare":
        pool = sorted(set(manifest.get("global_rare_codes", [])))
    else:
        pool = sorted(c for c in code_vocab if not c.startswith("<"))

    real = read_trajectories(args.cohort_cache, args.fold, only=[hid])[hid]
    needed = {c for visits in real.values() for visit in visits for c in visit}
    strict_rare = set(manifest.get("global_rare_codes", []))

    # Drop anything the pinned vocabulary does not carry: a code with no column
    # has no embedding, and silently indexing 0 would plot <pad>.
    missing = [c for c in pool if c not in code_vocab]
    pool = [c for c in pool if c in code_vocab]
    if missing:
        print(f"skipped {len(missing)} pool codes absent from the pinned "
              f"vocabulary")
    if not pool:
        raise SystemExit("no pool code has an embedding column")

    cols = np.array([code_vocab[c] for c in pool])
    groups = [
        "needed_rare" if (c in needed and c in strict_rare)
        else "needed_common" if c in needed
        else "other"
        for c in pool
    ]
    n_needed = sum(g != "other" for g in groups)
    print(f"hospital {hid}: {len(pool)} codes in the {args.code_pool} pool, "
          f"{n_needed} of them present in its real {args.fold} fold "
          f"({groups.count('needed_rare')} of those globally rare)")

    embeds: Dict[str, np.ndarray] = {}
    order: List[str] = []
    for spec in args.ckpt:
        if "=" not in spec:
            raise SystemExit(f"--ckpt wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        full = load_embeddings(path)
        if cols.max() >= full.shape[0]:
            raise SystemExit(
                f"{name}: vocabulary index {cols.max()} exceeds the "
                f"checkpoint's {full.shape[0]} embedding rows. The cache and "
                f"the run disagree on the vocabulary -- do not plot this."
            )
        embeds[name] = full[cols]
        order.append(name)
        print(f"  {name:14s} {full.shape[0]:>5} vocab x {full.shape[1]:>4} dims"
              f"  -> {embeds[name].shape[0]} pool codes")

    # --- 1. rotation-invariant agreement ---------------------------------- #
    sims = {n: upper(cosine_matrix(embeds[n])) for n in order}
    agreement: Dict[str, float] = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            agreement[f"{a}|{b}"] = float(np.corrcoef(sims[a], sims[b])[0, 1])

    print("\ncode-geometry agreement (Pearson r between cosine-similarity")
    print("matrices; rotation-invariant, so no alignment is assumed)")
    for pair, r in agreement.items():
        print(f"  {pair:<28} r = {r:+.3f}")
    print("\n1.0 would mean the two arms agree perfectly on which codes are")
    print("alike. Near 0 means averaging produced a different vocabulary map,")
    print("not merely different emission rates.")

    # --- 2. per-arm spread, an alignment-free shape statistic -------------- #
    spread: Dict[str, dict] = {}
    for n in order:
        mat = embeds[n]
        centred = mat - mat.mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        var = sv ** 2
        rare_idx = [i for i, g in enumerate(groups) if g == "needed_rare"]
        oth_idx = [i for i, g in enumerate(groups) if g == "other"]
        spread[n] = {
            "norm_mean": float(np.linalg.norm(mat, axis=1).mean()),
            "pc1_var_frac": float(var[0] / var.sum()),
            "pc2_var_frac": float(var[1] / var.sum()),
            # Effective rank: how many directions the code cloud really uses.
            "participation_ratio": float(var.sum() ** 2 / (var ** 2).sum()),
            "rare_norm_mean": float(
                np.linalg.norm(mat[rare_idx], axis=1).mean())
            if rare_idx else None,
            "other_norm_mean": float(
                np.linalg.norm(mat[oth_idx], axis=1).mean())
            if oth_idx else None,
        }

    width = max(len(n) for n in order) + 2
    print(f"\n{'arm':<{width}}{'|v| mean':>10}{'PC1 var':>9}{'eff.rank':>10}"
          f"{'|v| rare':>10}{'|v| other':>11}")
    for n in order:
        s = spread[n]
        rn = "n/a" if s["rare_norm_mean"] is None else f"{s['rare_norm_mean']:.3f}"
        on = "n/a" if s["other_norm_mean"] is None else f"{s['other_norm_mean']:.3f}"
        print(f"{n:<{width}}{s['norm_mean']:>10.3f}{s['pc1_var_frac']:>9.3f}"
              f"{s['participation_ratio']:>10.1f}{rn:>10}{on:>11}")
    print("\neff.rank is how many embedding directions the vocabulary actually")
    print("uses. A model that collapses its rare codes onto one another shows")
    print("up here as a low effective rank and a small rare-code norm.")

    # --- 3. per-arm 2D projections ---------------------------------------- #
    from sklearn.decomposition import PCA
    coords = {
        n: PCA(n_components=2, random_state=args.seed)
        .fit_transform(embeds[n]).round(4).tolist()
        for n in order
    }

    overlay = None
    if args.align == "procrustes" and len(order) > 1:
        ref = order[0]
        target = embeds[ref] - embeds[ref].mean(axis=0)
        fitted = PCA(n_components=2, random_state=args.seed).fit(target)
        overlay = {"reference": ref, "residual": {}, "coords": {}}
        overlay["coords"][ref] = fitted.transform(target).round(4).tolist()
        for n in order[1:]:
            src = embeds[n] - embeds[n].mean(axis=0)
            rot, resid = procrustes(src, target)
            overlay["residual"][n] = resid
            overlay["coords"][n] = fitted.transform(rot).round(4).tolist()
            print(f"procrustes {n} -> {ref}: residual {resid:.3f}"
                  + ("   <- rotation explained little; do not read the arrows"
                     if resid > 0.9 else ""))

    payload = {
        "cohort": os.path.basename(args.cohort_cache.rstrip("/")),
        "hospital": hid,
        "fold": args.fold,
        "code_pool": args.code_pool,
        "codes": pool,
        "groups": groups,
        "arms": order,
        "similarity_agreement": agreement,
        "spread": spread,
        "coords": coords,
        "overlay": overlay,
        "n_arrows": args.n_arrows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}.json")

    if not args.no_plot:
        plot(payload, f"{args.out}.png")
        print(f"wrote {args.out}.png")


def plot(payload: dict, path: str) -> None:
    """Per-arm PCA panels, plus the Procrustes overlay when one was computed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order: Sequence[str] = payload["arms"]
    groups = np.asarray(payload["groups"])
    overlay = payload["overlay"]
    ncol = len(order) + (1 if overlay else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 4.0))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, order):
        pts = np.asarray(payload["coords"][name])
        for gname, (color, marker, size, alpha) in GROUP_STYLE.items():
            sel = groups == gname
            if not sel.any():
                continue
            ax.scatter(pts[sel, 0], pts[sel, 1], s=size, c=color,
                       marker=marker, alpha=alpha, edgecolors="none",
                       label=gname.replace("_", " "))
        s = payload["spread"][name]
        ax.set_title(f"{name}\neff.rank {s['participation_ratio']:.1f} · "
                     f"|v| rare {s['rare_norm_mean']:.2f}", fontsize=10)
        ax.set_xlabel("own PC1", fontsize=8)
        ax.set_ylabel("own PC2", fontsize=8)

    if overlay:
        ax = axes[-1]
        ref = overlay["reference"]
        ref_pts = np.asarray(overlay["coords"][ref])
        rare = np.flatnonzero(groups == "needed_rare")
        ax.scatter(ref_pts[:, 0], ref_pts[:, 1], s=5, c="#9a968e", alpha=0.3,
                   edgecolors="none", label=f"{ref} (reference)")
        for name in order:
            if name == ref:
                continue
            pts = np.asarray(overlay["coords"][name])
            drift = np.linalg.norm(pts[rare] - ref_pts[rare], axis=1)
            pick = rare[np.argsort(-drift)[:payload["n_arrows"]]]
            for i in pick:
                ax.annotate("", xy=pts[i], xytext=ref_pts[i],
                            arrowprops=dict(arrowstyle="->", lw=0.6,
                                            color=COLORS.get(name, FALLBACK),
                                            alpha=0.8))
            ax.scatter(pts[pick, 0], pts[pick, 1], s=12,
                       c=COLORS.get(name, FALLBACK), edgecolors="none",
                       label=f"{name} (rotated)")
        resid = ", ".join(f"{k} {v:.2f}" for k, v in overlay["residual"].items())
        ax.set_title(f"Procrustes onto {ref}\nresidual: {resid}", fontsize=9)
        ax.set_xlabel(f"{ref} PC1", fontsize=8)
        ax.set_ylabel(f"{ref} PC2", fontsize=8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        for side in ax.spines.values():
            side.set_color("#dedbd4")

    agree = " · ".join(f"{k} r={v:+.2f}"
                       for k, v in payload["similarity_agreement"].items())
    fig.suptitle(
        f"Hospital {payload['hospital']} — HALO code embeddings, "
        f"{len(payload['codes'])} {payload['code_pool']} codes",
        fontsize=11.5, y=0.99)
    fig.text(0.5, 0.015,
             "Each panel is that arm's OWN PCA — compare structure, never "
             f"position. Rotation-invariant agreement: {agree}",
             ha="center", fontsize=8, color="#6b6862", wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
