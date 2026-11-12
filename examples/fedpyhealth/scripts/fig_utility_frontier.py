"""Specialist against generalist, for the DOWNSTREAM model instead of prevalence.

The prevalence frontier asks whether a generator's synthetic data reproduces
its own site's code rates or the cohort's. This asks the same question one
level down: a classifier trained on RLSP data (a site's own real records plus
the federation's pooled synthetic) is scored twice --

    y  own site   that hospital's own test patients
    x  pooled     all eight hospitals' test patients

Both numbers already exist. ``test2_rare_efficacy.py`` computes the own-site
view for every arm that HAS an own site, and writes it under ``own_site``; the
pooled view is the ``overall`` block every arm carries. Nothing here re-scores.

READ THE SMALL SITES WITH CARE, and that is not a formality. The per-site test
folds run 523, 463, 451, 434, 286, 178, 55, 40 patients. At ~3% prevalence over
30 codes the last two see roughly one positive per code, so their own-site
AUROC is close to a coin flip, and they are averaged in with equal weight. The
spread on the y axis is therefore part signal and part small-site noise. The
pooled axis does not have this problem -- it always scores on all 2,430.

Example:
    .venv/bin/python examples/fedpyhealth/scripts/fig_utility_frontier.py \\
        --json _outputs/results/tests/test2_rlspall_natural_rnn_seed0.json \\
        --json _outputs/results/tests/test2_rlspall_natural_rnn_seed1.json \\
        --json _outputs/results/tests/test2_rlspall_natural_rnn_seed2.json \\
        --out _outputs/results/utility_frontier.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from collections import defaultdict

#: arm family -> (label, colour, marker, filled). Mirrors the prevalence
#: frontier's conventions so the two figures can sit side by side: circles for
#: the no-method references, squares/plus for mechanisms, hollow for an arm
#: whose update was constrained to an adapter.
ARMS = [
    ("real_local",       "own real data only",                "#43494F", "o", True),
    ("rlsp:fedavg",      "+ FedAvg",                          "#8A9299", "o", True),
    ("rlsp:fedavg_irm",  "+ IRM",                             "#C44E52", "s", True),
    ("rlsp:lat_xm4",     "+ best-of-K (latent)",              "#5B8C5A", "s", True),
    ("rlsp:lat_fed_irm", "+ IRM + best-of-K (latent)",        "#5B8C5A", "P", True),
    ("rlsp:lat_lora_ft", "+ IRM + best-of-K + LoRA FT",       "#5B8C5A", "P", False),
    ("rlsp:fedavg_ft",   "+ full fine-tune",                  "#8A9299", "s", False),
    ("rlsp:st20",        "+ self-training",                   "#4A6FA5", "s", True),
    ("rlsp:ditto0.1",    "Ditto (lambda=0.1)",                "#2E7D5B", "D", True),
    ("rlsp:perada",      "PerAda",                            "#2E7D5B", "D", False),
]


def load(paths, metric):
    """Mean (own-site, pooled) per arm, over seeds and then over hospitals."""
    own, pool = defaultdict(list), defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        po, pp = defaultdict(list), defaultdict(list)
        for k, x in d["arms"].items():
            if "own_site" not in x:
                continue
            fam = x.get("family") or k
            po[fam].append(x["own_site"]["overall"][metric])
            pp[fam].append(x["overall"][metric])
        # hospital-level mean first, so a seed counts once however many sites
        # it happens to carry.
        for f in po:
            own[f].append(st.fmean(po[f]))
            pool[f].append(st.fmean(pp[f]))
    return ({f: st.fmean(v) for f, v in own.items()},
            {f: st.fmean(v) for f, v in pool.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="append", required=True,
                   help="scored test2 files; repeat once per seed")
    p.add_argument("--metric", default="roc_auc_macro",
                   choices=["roc_auc_macro", "ap_macro", "f1_best_macro"])
    p.add_argument("--out", default="_outputs/results/utility_frontier.png")
    p.add_argument("--title", default=None)
    p.add_argument("--marker-scale", type=float, default=1.0,
                   help="Multiply every marker area. <1 when arms cluster, "
                        "which they do here -- the whole cloud spans 0.03 "
                        "AUROC on the own-site axis.")
    a = p.parse_args(argv)

    own, pool = load(a.json, a.metric)
    pts = [(lab, pool[f], own[f], c, m, fi)
           for f, lab, c, m, fi in ARMS if f in own]
    for f, lab, *_ in ARMS:
        if f not in own:
            print(f"  skipping {lab}: not in the scored files")

    print(f"{'arm':<32}{'pooled':>9}{'own site':>10}{'own-pooled':>12}")
    for lab, x, y, *_ in pts:
        print(f"  {lab:<30}{x:>9.3f}{y:>10.3f}{y - x:>+12.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [x for _, x, _, _, _, _ in pts] + [y for _, _, y, _, _, _ in pts]
    lo, hi = min(xs) - 0.02, max(xs) + 0.02
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # y = x: a classifier that serves its own hospital exactly as well as it
    # serves the cohort. ABOVE it is a specialist, below a generalist.
    ax.plot([lo, hi], [lo, hi], ls=":", lw=1.0, color="#C9CFD4", zorder=1)
    ax.text(hi - 0.004, hi - 0.004, "serves both equally", fontsize=8.5,
            color="#9AA1A8", ha="right", va="bottom", rotation=45,
            rotation_mode="anchor")

    for lab, x, y, c, m, fi in pts:
        ax.scatter(x, y, s=(150 if m == "P" else 110) * a.marker_scale, marker=m,
                   facecolor=c if fi else "white",
                   edgecolor="white" if fi else c,
                   linewidth=1.1 if fi else 1.8, zorder=3, label=lab)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    name = {"roc_auc_macro": "ROC-AUC", "ap_macro": "average precision",
            "f1_best_macro": "F1 at best threshold"}[a.metric]
    ax.set_xlabel(f"scored on the POOLED test fold\n(generalist) — {name}",
                  fontsize=9.5)
    ax.set_ylabel(f"scored on the site's OWN test fold\n(specialist) — {name}",
                  fontsize=9.5)
    ax.set_title(a.title or "Downstream utility: specialist against generalist",
                 fontsize=11.5)
    ax.grid(True, color="#E6E9EC", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#C9CFD4")
    ax.tick_params(labelsize=8.5, color="#C9CFD4")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
