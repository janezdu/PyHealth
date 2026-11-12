#!/usr/bin/env python3
"""Paper figure 3: what a downstream model learns from each regime's synthetic records.

Train-on-synthetic, test-on-real. One bar per generator regime, plus the
real-data reference lines that say whether any of it was worth doing.

Nothing here trains or scores anything. It reads what ``test2_rare_efficacy.py``
already wrote and draws. Seeds are aggregated: the bar is the mean over seed
files and the whisker is the min/max, so a bar whose whisker overlaps its
neighbour's is not a difference this evidence can carry.

The reference lines are the argument, not decoration:

``prior``        per-code training prevalence. Below this a model learned nothing.
``real_local``   a hospital's OWN real records. This is the bar federation must
                 clear -- if synthetic loses to it, the pipeline buys nothing.
``real_pooled``  all eight hospitals' real records. The ceiling. No synthetic
                 regime should be expected to reach it, and saying so plainly
                 is what makes the rest of the figure credible.

Hospital identifiers never reach the output; arms are relabelled to the paper's
names and only macro-averaged metrics are read. Safe to commit
(``.llms/rules/03-data-safety.md``).

Usage::

    .venv/bin/python examples/fedpyhealth/scripts/fig_test2_utility.py
    .venv/bin/python examples/fedpyhealth/scripts/fig_test2_utility.py --list

Run from the repo root. CPU-only, about a second.
"""
import argparse
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: ``(family key, paper label, glob)``, in draw order (top bar first).
#:
#: Each arm names its OWN source, because no single results file holds every
#: arm. That is legitimate only because the files below share an evaluation
#: protocol -- same classifier, same 30 drawn codes, same eval seed per seed
#: file, same 8000-record budget, same mask-folds. ``_check_protocol`` enforces
#: it and raises rather than drawing a comparison that is not one.
#:
#: The three real-data references appear in every file; they are read from
#: whichever files are used and must agree, which is itself a check that the
#: protocol claim holds.
RWFIG = "_outputs/results/tests/test2_head_rwfig_seed*.json"
OURO  = "_outputs/results/tests/test2_head_ouroxm_seed*.json"
RLSP  = "_outputs/results/tests/test2_head_rlsp_rnn_seed*.json"

ARMS = [
    ("local",         "local-only",                RWFIG),
    ("centralized",   "centralized",               RWFIG),
    ("fedavg",        "FedAvg",                    RWFIG),
    ("fedavg_irm1e4", r"IRM $\rho=10^4$",          RWFIG),
    ("fed_xm4_do03",  "FedAvg + best-of-K",        RLSP),
    ("st20_xm4_irm",  "IRM + best-of-K",           OURO),
    ("xm4",           "best-of-K, centralized",    RWFIG),
]

#: Fields that must match across every file used, or the bars are not
#: comparable. ``classifier`` is the one that bites: a logreg file and an rnn
#: file will happily average together and produce a plausible-looking lie.
PROTOCOL = ("classifier", "n_scored_codes", "train_budget", "mask_folds",
            "cohort_name", "fold")

#: Which number.
#:
#: ``overall`` is the ``n_scored_codes`` (30) codes DRAWN from the pooled rare
#: pool for this evaluation -- so it is already the rare tail, just the sampled
#: 30 rather than all 533.
#:
#: ``global_rare`` scores the whole pool and its ``roc_auc_macro`` is NaN in
#: every arm of every file: most of those 533 codes have no positive in the
#: validation fold, and ROC-AUC is undefined for a code with one class. AP
#: survives there, ROC-AUC does not. If you want the full pool, switch METRIC
#: to ``ap_macro`` -- do not switch SCOPE alone, you will get a blank figure.
SCOPE = "overall"
METRIC = "roc_auc_macro"

REFS = [
    ("prior",       "prior",       "#6b7480", ":"),
    ("real_local",  "real_local",  "#b8336a", "--"),
    ("real_pooled", "real_pooled", "#16191d", "-."),
]

BAR = "#7b8087"
BAR_METHOD = "#0f6e6e"


def _metric(node):
    """Pull SCOPE/METRIC out of one arm payload, or None."""
    scope = node.get(SCOPE)
    if not isinstance(scope, dict):
        return None
    v = scope.get(METRIC)
    return v if isinstance(v, (int, float)) else None


def _arm_values(path, wanted):
    """-> {family: median over that file's hospital-level values}."""
    d = json.load(open(path))
    per = {}
    for name, node in d.get("arms", {}).items():
        v = _metric(node)
        if v is None:
            continue
        fam = node.get("family") or name
        fam = fam[5:] if fam.startswith("tstr:") else fam
        per.setdefault(fam, []).append(v)
    return {k: st.median(v) for k, v in per.items() if k in wanted}


def _check_protocol(paths):
    """Raise if the files were not produced under one evaluation protocol."""
    seen = {}
    for p in paths:
        d = json.load(open(p))
        key = tuple(d.get(f) for f in PROTOCOL)
        seen.setdefault(key, []).append(os.path.basename(p))
    if len(seen) > 1:
        lines = ["files do not share an evaluation protocol, refusing to draw:"]
        for key, files in seen.items():
            lines.append("  " + ", ".join(f"{f}={v!r}" for f, v in zip(PROTOCOL, key)))
            lines.append("    " + ", ".join(sorted(files)))
        raise SystemExit("\n".join(lines))
    return dict(zip(PROTOCOL, next(iter(seen))))


def collect():
    """-> ({label: [value per seed]}, {ref: [value per seed]}, protocol, n_seeds)

    Read from ``arms``, not ``families``. The family summaries carry average
    precision only; ROC-AUC exists per hospital. A multi-generator regime has
    eight hospital entries and we take their MEDIAN -- the mean is dragged
    somewhere no hospital is by one degenerate classifier, the same reason
    ``fig_specialist_generalist.py`` uses the median.
    """
    globs = sorted({g for _, _, g in ARMS})
    all_paths = []
    for g in globs:
        hit = sorted(glob.glob(g))
        if not hit:
            raise SystemExit(f"no files matched {g!r} (run from the repo root)")
        all_paths += hit
    protocol = _check_protocol(all_paths)

    wanted = {k for k, _, _ in ARMS}
    refkeys = {k for k, _, _, _ in REFS}

    arms, refs, counts = {}, {}, []
    for key, label, g in ARMS:
        paths = sorted(glob.glob(g))
        vals = [v[key] for p in paths for v in [_arm_values(p, wanted)] if key in v]
        if not vals:
            print(f"WARNING: no data for {key!r} in {g}; dropped")
            continue
        arms[label] = vals
        counts.append(len(vals))

    for p in all_paths:
        for k, v in _arm_values(p, refkeys).items():
            refs.setdefault(k, []).append(v)

    if len(set(counts)) > 1:
        print(f"WARNING: arms do not share a seed count ({sorted(set(counts))}); "
              f"whiskers are not comparable across bars")
    return arms, refs, protocol, (counts[0] if counts else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="paper/figs")
    ap.add_argument("--stem", default="test2_utility")
    ap.add_argument("--list", action="store_true",
                    help="print the families available in every test2 file and exit")
    args = ap.parse_args()

    if args.list:
        for p in sorted(glob.glob("_outputs/results/tests/test2_*.json")):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            fams = sorted(k[5:] for k in d.get("families", {}) if k.startswith("tstr:"))
            if fams:
                print(f"{os.path.basename(p)}  clf={d.get('classifier','?')} "
                      f"n={d.get('n_scored_codes')} budget={d.get('train_budget')}")
                print("   ", ", ".join(fams))
        return

    arms, refs, protocol, n_seeds = collect()
    rows = [(lab, arms[lab]) for _, lab, _ in ARMS if lab in arms]

    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(rows) + 2.0))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, vals) in zip(ys, rows):
        m = st.mean(vals)
        colour = BAR_METHOD if ("IRM" in lab or "best-of-K" in lab) else BAR
        ax.barh(y, m, height=0.62, color=colour, zorder=2)
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y, y], color="#16191d",
                    lw=1.1, zorder=3, solid_capstyle="butt")
        ax.text(m + 0.006, y, f"{m:.3f}", va="center", ha="left",
                fontsize=9, color="#3d454e", zorder=3)

    # Reference labels go in headroom ABOVE the top bar, staggered so that
    # prior and real_local -- which are always close together -- do not
    # overlap, and so that none of them lands on a bar's value label.
    ax.set_ylim(-0.62, len(rows) - 1 + 1.55)
    top = len(rows) - 1 + 1.30
    for i, (key, lab, colour, ls) in enumerate(REFS):
        if key not in refs:
            continue
        v = st.mean(refs[key])
        ax.axvline(v, color=colour, ls=ls, lw=1.2, zorder=1)
        ax.annotate(f"{lab} {v:.3f}", xy=(v, top - 0.40 * i),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=colour,
                    bbox=dict(fc="white", ec="none", pad=1.0), zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=10)
    ax.set_xlabel("ROC-AUC, 30 drawn rare codes  ($\\rightarrow$ better)", fontsize=10)
    hi = max([st.mean(v) for _, v in rows] + [st.mean(v) for v in refs.values()])
    ax.set_xlim(0, hi * 1.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8ebee", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{protocol['classifier']} classifier, "
                 f"{protocol['n_scored_codes']} drawn codes, "
                 f"budget {protocol['train_budget']}; mean over {n_seeds} seeds, "
                 f"whisker is min-max", fontsize=8, color="#6b7480", loc="left")
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(args.out_dir, f"{args.stem}.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()
