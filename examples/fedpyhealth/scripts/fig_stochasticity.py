#!/usr/bin/env python3
"""Static figure of the stochasticity sweeps: dropout x K and latent width x K.

The companion to viz/mechanisms.html, for when a PNG is more useful than an
interactive page (SSH, a slide, a message). Same data, same reference lines.

    python examples/fedpyhealth/scripts/fig_stochasticity.py

Writes _outputs/figs/stochasticity.png. Macro-averaged across the 8 hospitals;
error bars are the BETWEEN-SITE standard deviation, not a confidence interval
on the mean -- they show how unevenly one global model serves the cohort, which
is a different (and larger) quantity. The standard error on the macro mean is
that divided by sqrt(8), and is drawn as the shaded band so the two are not
confused.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.getcwd(), "_outputs", "results", "runs")
OUTDIR = os.path.join(os.getcwd(), "_outputs", "figs")
METRICS = [
    ("PrevVal_Rare_Prevalence_Pearson", "Rare-code prevalence (Pearson)"),
    ("PrevVal_All_Prevalence_Pearson", "All-code prevalence (Pearson)"),
]
K_COLORS = {1: "#b8563e", 4: "#2f6f7e", 8: "#7a6a9b"}


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        with open(path) as fh:
            d = json.load(fh)
        c = d["config"]
        rows.append({
            "dropout": float(c.get("dropout", 0.0) or 0.0),
            "z": int(c.get("latent_dim", 0) or 0),
            "k": int(c.get("xm_k", 1) or 1),
            "m": d["macro_avg_metrics"],
        })
    return rows


def val(row, key):
    v = row["m"].get(key, {})
    return v.get("mean"), v.get("between_hospital_std"), v.get("n_hospitals", 8)


def panel(ax, rows, x_key, x_label, metric, refs):
    ks = sorted({r["k"] for r in rows})
    for k in ks:
        sub = sorted([r for r in rows if r["k"] == k], key=lambda r: r[x_key])
        xs = [r[x_key] for r in sub]
        ys, es = [], []
        for r in sub:
            m, sd, _ = val(r, metric)
            ys.append(m)
            es.append(sd or 0.0)
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, capsize=3,
                    color=K_COLORS.get(k, "#666"), label=f"K={k}", alpha=.92)
    for i, (label, m, se) in enumerate(refs):
        ax.axhline(m, ls="--" if i == 0 else ":", lw=1.3, color="#8a867e", zorder=0)
        # The band is the standard error of the macro mean -- the honest "is
        # this line different from that point" scale. The error bars are the
        # between-site spread and are much larger; drawing both stops the
        # bars from being read as uncertainty on the mean.
        ax.axhspan(m - se, m + se, color="#8a867e", alpha=.10, zorder=0)
        # control and trunk land within ~0.006 of each other, so their labels
        # collide if both are drawn at the same anchor. Alternate side and
        # baseline rather than nudging by a fixed offset, which would break
        # again on a different metric's scale.
        ax.annotate(label,
                    xy=(0.985 if i == 0 else 0.015, m),
                    xycoords=("axes fraction", "data"),
                    ha="right" if i == 0 else "left",
                    va="bottom" if i == 0 else "top",
                    fontsize=7.5, color="#8a867e")
    ax.set_xlabel(x_label, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.18, lw=.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    dropout, latent = load("dropout_high_v1-*.json"), load("latent_width_v1-*.json")
    if not (dropout and latent):
        raise SystemExit(f"no sweep results under {RESULTS}")

    refs_src = [("control E2/R50", "control_plain_v1-*.json"),
                ("trunk E10/R10", "trunk_e10_v1-*.json")]

    fig, axes = plt.subplots(len(METRICS), 2, figsize=(9.6, 3.5 * len(METRICS)),
                             sharey="row")
    axes = axes.reshape(len(METRICS), 2)
    for row, (metric, nice) in enumerate(METRICS):
        refs = []
        for label, pat in refs_src:
            rs = load(pat)
            if rs:
                m, sd, n = val(rs[0], metric)
                refs.append((label, m, (sd or 0.0) / math.sqrt(max(1, n))))
        panel(axes[row][0], dropout, "dropout", "dropout", metric, refs)
        panel(axes[row][1], latent, "z", "latent width (z)", metric, refs)
        axes[row][0].set_ylabel(nice, fontsize=9)
        axes[row][0].set_title("Dropout x K  (no latent)", fontsize=9.5, loc="left")
        axes[row][1].set_title("Latent width x K  (no dropout)", fontsize=9.5, loc="left")
    axes[0][0].legend(fontsize=8, frameon=False)
    axes[0][1].legend(fontsize=8, frameon=False)
    fig.suptitle("Stochasticity sweeps — what best-of-K has to select over",
                 fontsize=11.5, x=.007, ha="left", y=.995)
    fig.text(.007, .005,
             "Bars = between-site sd (8 hospitals). Shaded band = standard error "
             "of the macro mean, the scale on which differences are readable.",
             fontsize=7.5, color="#8a867e", ha="left")
    fig.tight_layout(rect=(0, .022, 1, .972))
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "stochasticity.png")
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
