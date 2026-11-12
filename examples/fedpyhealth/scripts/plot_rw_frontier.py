"""Specialist against generalist R^2 for the rare-code-reweighted arms.

A publication figure, in the style of the project's earlier specialist/generalist
plots: two square panels -- all codes and rare codes -- a y = x diagonal with the
upper-left half shaded, and a legend underneath.

Reads the scored Test 1 JSONs directly, so it always reflects what was actually
measured rather than a hand-copied table.

Two choices worth knowing about:

* **Square axes, shared between panels.** Specialist and generalist are the same
  quantity measured against different reference folds, so an aspect ratio other
  than 1 would make the diagonal a lie -- the whole point of the diagonal is that
  distance from it is readable by eye.
* **The dashed line is the baseline frontier**, through local-only and
  centralized -- the two no-method regimes at opposite ends of the
  specialist/generalist trade-off. Shading marks what is reachable without
  applying any method, so a point above the line is the only kind of result
  worth claiming. The faint dotted line is still y = x, for reference.
* **Off-scale arms are clamped and annotated, not dropped.** Unweighted
  ``centralized`` reaches -55 / -70 on R^2. Letting it set the limits compresses
  everything else into a few pixels; dropping it silently would hide a real
  failure. It is drawn at the corner with its true value printed.

Example:
    python examples/fedpyhealth/scripts/plot_rw_frontier.py \\
        --spec _outputs/results/tests/test1_rwfig_spec_b200.json \\
        --gen  _outputs/results/tests/test1_rwfig_gen_b200.json \\
        --out  _outputs/results/rw_frontier.png
"""

import argparse
import json
import os
from statistics import fmean

# CIRCLE = a plain training regime, nothing added. SQUARE = the same trunk with
# something bolted on. STAR = the arm stacking both methods.
#
# The frontier is anchored on local-only and FedAvg -- the two regimes at
# opposite ends of the trade-off -- so both are circles and a reader can see
# what the line is drawn through. Fill separates arms that share a shape.
ARMS = [
    # arm             label                       colour     mark fill  base
    ("local",         "local-only",               "#43494F", "o", True,  True),
    ("fedavg",        "FedAvg",                   "#8A9299", "o", True,  True),
    # Two arm names for the same regime, because different scoring runs spell
    # it differently: mix_spec/mix_gen (the unweighted-arms figure that
    # borrows the REWEIGHTED centralized) calls it centralized_rw, while
    # every other file calls it centralized. Only one is present in a given
    # file set, so listing both lets one table serve every figure and the
    # absent one is skipped. --centralized-label sets the legend text.
    ("centralized",    None,                      "#43494F", "o", False, True),
    ("centralized_rw", None,                      "#43494F", "o", False, True),
    # "20 ep" is in the label because the b200 rebuild had to use full_ft20:
    # no 2-epoch full fine-tune was ever scored at 200 bootstraps. Dropping
    # the suffix would silently equate it with the old 2-epoch point.
    # The epoch count is in BOTH labels because the two arms sit at opposite
    # ends of the fine-tuning dial and were confused for each other: the
    # prebuild source's `fedavg_ft` is ft_epochs=2 (98 federated passes + 2
    # local), while `full_ft20` is ft_epochs=20 (80 + 20). Same 100-pass
    # budget, very different split.
    ("full_ft20",     "FedAvg + fine-tune (full 6M finetune)",
                                                "#8A9299", "s", True, True),
    # Same 20 fine-tune epochs as full_ft20, same 100-pass budget, same trunk.
    # The ONLY difference is that 38K adapter parameters move instead of all
    # 6.17M. Triangle so the pair reads as a matched ablation rather than two
    # more points in the fine-tuning family.
    ("lora_head20",   "FedAvg + fine-tune (LoRA head, 38K)",
                                                "#8A9299", "s", False, True),
    ("fedavg_irm1e4", r"FedAvg + IRM $\rho=10^4$", "#C44E52", "s", True,  False),
    ("fed_xm4_do03",  "FedAvg + best-of-K",       "#DD8452", "s", True,  False),
    # The latent twin of the orange square: best-of-K, NO IRM. Same square as
    # fed_xm4_do03 because they are the same recipe; the colour carries the one
    # difference (latent z=16 vs dropout 0.3). Without it the figure has no
    # one-variable IRM ablation at all -- every other pair that looks like one
    # also changes the stochasticity source.
    ("lat_xm4",       "FedAvg + best-of-K (latent z=16)",
                                                  "#5B8C5A", "s", True,  False),
    # The dropout best-of-K family. It lands on top of lat_fed_irm
    # (+0.511/+0.945 vs +0.474/+0.940) -- dropout and latent best-of-K score the
    # same -- so the two are drawn on SEPARATE figures via --drop rather than
    # overplotted.
    ("do01_fed_irm",  "FedAvg + IRM + best-of-K (dropout)",
                                                  "#8172B3", "*", True,  False),
    ("do_lora_ft",    "FedAvg + IRM + best-of-K (dropout)\n+ fine-tune (LoRA head)",
                                                  "#8172B3", "*", False, False),
    # The matched no-fine-tune partner for lat_lora_ft below: same latent z=16,
    # same IRM, same K=4, dropout 0. Filled/hollow of the SAME marker so the
    # pair reads as one ablation. Without it the hollow star's only neighbour
    # was do01_fed_irm, which differs in stochasticity source (dropout, not
    # latent) as well as fine-tuning -- a two-variable gap read as one.
    ("lat_fed_irm",   "FedAvg + IRM + best-of-K (latent z=16)",
                                                  "#8172B3", "*", True,  False),
    # The same recipe plus per-site fine-tuning: hollow star, same purple, so
    # "personalized version of the arm above" reads from the shape pairing.
    # NOTE it is the z16/dropout-0 best-of-K checkpoint, not the do-0.1 one
    # do01_fed_irm names -- the do-0.1 run has never been fine-tuned.
    ("lat_lora_ft",   "FedAvg + IRM + best-of-K (latent z=16)\n+ fine-tune (LoRA head)",
                                                  "#8172B3", "*", False, False),
    # Literature baselines get a shape of their own -- DIAMOND -- so "a method
    # from a paper" is separable from "ours" at a glance, before any colour is
    # read. Ditto is shown at lambda=0.1, the best of the sweep that is
    # actually Ditto: lambda=0 scores higher but is by definition plain
    # FedAvg+FT, already on the figure as its own point.
    # The Ditto lambda sweep, as a colour RAMP on one marker: lambda is a
    # continuous knob, so light-to-dark reads as a trajectory rather than four
    # unrelated arms. lambda=0 removes the proximal term entirely and IS plain
    # FedAvg+FT, which is why it sits so far from the other three.
    ("ditto0",        r"Ditto $\lambda=0$ (= FedAvg+FT)",
                                                  "#9CCFB4", "D", True,  False),
    ("ditto0.001",    r"Ditto $\lambda=0.001$",    "#C4E3D1", "D", True,  False),
    ("ditto0.01",     r"Ditto $\lambda=0.01$",     "#8CC2A5", "D", True,  False),
    ("ditto0.1",      r"Ditto $\lambda=0.1$",      "#5FA37F", "D", True,  False),
    ("ditto1",        r"Ditto $\lambda=1$",        "#2E7D5B", "D", True,  False),
    ("ditto10",       r"Ditto $\lambda=10$",       "#164F38", "D", True,  False),
    ("perada",        "PerAda (LoRA head + KD)",   "#2E7D5B", "D", False, False),
    # These three over-emit by 2.6-3.5x, so R^2 puts them far off the bottom.
    # Their Pearson is normal (0.83-0.88 specialist) -- the code ORDERING is
    # fine and only the scale is wrong. Clamped to the axis with the true value
    # in the annotation rather than dropped.
    ("fedprox",       r"FedProx $\mu=10^{-2}$",    "#8C6D3F", "D", True,  False),
    ("fedrep",        "FedRep",                    "#8C6D3F", "D", False, False),
    ("lgfedavg",      "LG-FedAvg",                 "#4F7A8C", "D", False, False),
    # The two over-emitters rerun with a mild emission-rate penalty
    # (--rate-mu 1e-3). OUR modification, not the published methods, so the
    # label says so; filled to tell them apart from the hollow originals.
    ("fedrep_rate",   r"FedRep + rate penalty",    "#C9822E", "D", True,  False),
    ("lgfedavg_rate", r"LG-FedAvg + rate penalty", "#6FA3BA", "D", True,  False),
]





def arm_value(runs, arm, scope, metric="R2", only=None):
    """Mean over hospitals of one metric, or None if the arm was not scored.

    Args:
        runs: The merged ``{arm: {hospital: {metric: [value, se]}}}`` block.
        arm: Arm id.
        scope: ``"All"`` or ``"Rare"`` -- which code pool.
        metric: Metric suffix, e.g. ``"R2"``.
        only: Restrict the mean to these hospital ids. ``None`` uses every
            site, which is the standard all-sites figure.
    """
    if arm not in runs:
        return None
    key = f"PrevVal_{scope}_Prevalence_{metric}"
    vals = [v[key][0] for h, v in runs[arm].items()
            if only is None or h in only]
    return fmean(vals) if vals else None


def size_groups(manifest_path, threshold):
    """Split the cohort's hospitals into large and small by train size.

    hilo8_random was BUILT as four sites at or above 1,500 train patients and
    four below, so the default threshold reproduces the cohort's own design
    split rather than imposing a new one.

    Returns:
        ``(large, small)``, each a sorted list of hospital ids, descending by
        train size.
    """
    per = json.load(open(manifest_path))["per_hospital"]
    sizes = {h: v["n_train"] for h, v in per.items()}
    order = sorted(sizes, key=lambda h: -sizes[h])
    large = [h for h in order if sizes[h] >= threshold]
    small = [h for h in order if sizes[h] < threshold]
    return large, small, sizes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", action="append", default=[],
                   help="scored specialist JSON; repeatable. The unweighted "
                        "arms live across several files, so they are merged "
                        "here rather than being re-scored into one. First file "
                        "to define an arm wins, which makes the order "
                        "meaningful when two files share a name.")
    p.add_argument("--gen", action="append", default=[],
                   help="scored generalist JSON; repeatable, same rules")
    p.add_argument("--metric", default="R2",
                   choices=["R2", "Pearson", "RMSE", "CCC", "spearman",
                            "r2_log", "slope"],
                   help="CCC and below come from compare_prevalence_metrics.py "
                        "rather than test1_prevalence, so point --spec/--gen "
                        "at the files it emits")
    p.add_argument("--centralized-label", default="centralized",
                   help="Legend text for the centralized arm. The reweighted "
                        "figure's centralized IS the reweighted run, so that "
                        "is the default; the unweighted runner overrides it "
                        "with plain 'centralized'.")
    p.add_argument("--split", default="scope", choices=["scope", "size"],
                   help="What the two panels contrast. scope (the default) is "
                        "all codes vs rare codes, every site averaged into "
                        "both. size splits the SITES instead -- large vs small "
                        "-- at one fixed code pool set by --scope, which is "
                        "the view that shows whether a method helps the sites "
                        "federation is supposed to help.")
    p.add_argument("--scope", default="All", choices=["All", "Rare"],
                   help="Which code pool to use when --split size. Ignored "
                        "for --split scope, which shows both.")
    p.add_argument("--manifest",
                   default=os.path.join(
                       os.environ.get("FEDCOHORT_CACHE", ""),
                       "hilo8_random", "manifest.json"),
                   help="Cohort manifest, read for per-hospital train sizes "
                        "when --split size.")
    p.add_argument("--size-threshold", type=int, default=1500,
                   help="Train-patient count at or above which a site counts "
                        "as large. Default 1500 is hilo8_random's own design "
                        "boundary: 4 sites above, 4 below.")
    p.add_argument("--lim", default="",
                   help="axis limits as LO,HI; default is fitted to the data "
                        "with a margin, rounded outward to a tick")
    p.add_argument("--only", default="",
                   help="Comma-separated arm ids to keep, in place of --drop. "
                        "Easier than dropping a dozen when the figure wants "
                        "four arms.")
    p.add_argument("--logscale", action="store_true",
                   help="SYMLOG both axes. Plain log cannot render this metric "
                        "-- R^2 vs identity is unbounded below and several arms "
                        "are negative (LG-FedAvg reaches -17.7), so a log axis "
                        "would drop them silently. Symlog is linear inside "
                        "--linthresh and logarithmic outside, which keeps the "
                        "sign and shows three orders of magnitude at once.")
    p.add_argument("--linthresh", type=float, default=0.1,
                   help="Half-width of the LINEAR region of the symlog axes. "
                        "Below this, values are drawn linearly (log would "
                        "explode toward zero); above, logarithmically.")
    p.add_argument("--marker-scale", type=float, default=1.0,
                   help="Multiply every marker area. <1 for crowded figures.")
    p.add_argument("--drop", default="",
                   help="Comma-separated arm ids to omit. The dropout and "
                        "latent best-of-K families land on the same point, so "
                        "each figure drops the other's family rather than "
                        "stacking two markers on one spot.")
    p.add_argument("--frontier", default="local,fedavg",
                   help="the two arms the frontier line is drawn through, as "
                        "NAME,NAME. It is a TWO-POINT line, so an anchor far "
                        "outside the plotted region determines it almost "
                        "alone: unweighted centralized sits at (-44, -53), "
                        "which gives a slope of +1.23 and a frontier of +2.4 "
                        "where the methods actually are. Use an on-scale pair "
                        "-- local,fedavg -- whenever an anchor is off the "
                        "chart. Empty draws no frontier.")
    p.add_argument("--title", default="")
    # Presentation overrides, so one figure can be relabelled for the paper
    # without renaming the arm in every other figure that shares ARMS.
    p.add_argument("--rename", action="append", default=[],
                   help="ARM=LABEL, repeatable: legend text for one arm")
    p.add_argument("--fill", default="",
                   help="comma-separated arm ids to draw FILLED regardless of "
                        "ARMS (e.g. to promote a hollow arm to the headline)")
    p.add_argument("--panel-titles", default="",
                   help="comma-separated titles, one per panel, replacing "
                        "'Prevalence <metric>, <scope>'")
    p.add_argument("--xlabel", default="scored against the pooled cohort "
                   "fold\n(generalized)")
    p.add_argument("--ylabel", default="scored against the site's own "
                   "fold\n(personalized)")
    p.add_argument("--color", action="append", default=[],
                   help="ARM=#HEX, repeatable: marker colour for one arm")
    p.add_argument("--bottom", type=float, default=0.17,
                   help="figure fraction reserved under the axes for the "
                        "legend; lower pulls the legend up")
    p.add_argument("--legend-y", type=float, default=-0.09,
                   help="vertical anchor of the legend in figure coords")
    p.add_argument("--region-above", default="beyond the frontier",
                   help="label for the unshaded region above the frontier")
    p.add_argument("--region-below", default="within the trade-off",
                   help="label for the shaded region below the frontier")
    p.add_argument("--no-region-text", action="store_true",
                   help="omit the 'above the baseline frontier' / 'reachable "
                        "without a method' annotations")
    p.add_argument("--out", default="_outputs/results/rw_frontier.png")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not a.spec:
        a.spec = ["_outputs/results/tests/test1_rwfig_spec_b200.json"]
    if not a.gen:
        a.gen = ["_outputs/results/tests/test1_rwfig_gen_b200.json"]

    def merge(paths):
        out = {}
        for path in paths:
            for arm, v in json.load(open(path))["runs"].items():
                out.setdefault(arm, v)
        return out

    spec, gen = merge(a.spec), merge(a.gen)
    print(f"merged {len(a.spec)} specialist and {len(a.gen)} generalist files: "
          f"{len(spec)} arms available")

    # panels: (key, axis subtitle, code scope, hospital subset or None)
    if a.split == "size":
        if not os.path.exists(a.manifest):
            raise SystemExit(
                f"--split size needs the cohort manifest for train sizes; "
                f"{a.manifest!r} does not exist. Pass --manifest."
            )
        large, small, sizes = size_groups(a.manifest, a.size_threshold)
        if not large or not small:
            raise SystemExit(
                f"--size-threshold {a.size_threshold} puts every site on one "
                f"side: {sizes}"
            )
        panels = [
            ("large", f"large sites (n_train >= {a.size_threshold})",
             a.scope, set(large)),
            ("small", f"small sites (n_train < {a.size_threshold})",
             a.scope, set(small)),
        ]
        print("large: " + ", ".join(f"{h}({sizes[h]})" for h in large))
        print("small: " + ", ".join(f"{h}({sizes[h]})" for h in small))
    else:
        panels = [("All", "all codes", "All", None),
                  ("Rare", "rare codes", "Rare", None)]

    # --drop is applied BEFORE loading, so a dropped arm never prints a
    # "not present in the scored files" line it would share with a genuinely
    # missing one.
    dropped = {x for x in a.drop.split(",") if x}
    keep = {x for x in a.only.split(",") if x}
    unknown = (dropped | keep) - {arm for arm, *_ in ARMS}
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(sorted(unknown))}")
    if keep:
        dropped |= {arm for arm, *_ in ARMS} - keep

    renames = {}
    for spec_ in a.rename:
        if "=" not in spec_:
            raise SystemExit(f"--rename wants ARM=LABEL, got {spec_!r}")
        k, v = spec_.split("=", 1)
        renames[k] = v
    force_fill = {x for x in a.fill.split(",") if x}
    recolour = {}
    for spec_ in a.color:
        if "=" not in spec_:
            raise SystemExit(f"--color wants ARM=#HEX, got {spec_!r}")
        k, v = spec_.split("=", 1)
        recolour[k] = v
    unknown = (set(renames) | force_fill | set(recolour)) - {arm for arm, *_ in ARMS}
    if unknown:
        raise SystemExit(f"unknown arm(s) in --rename/--fill: "
                         f"{', '.join(sorted(unknown))}")

    pts = []
    for arm, label, colour, marker, filled, is_base in ARMS:
        if arm in dropped:
            continue
        if label is None:
            label = a.centralized_label
        label = renames.get(arm, label)
        filled = filled or arm in force_fill
        colour = recolour.get(arm, colour)
        row = {"label": label, "colour": colour, "marker": marker,
               "filled": filled, "arm": arm, "base": is_base}
        ok = True
        for key, _sub, scope, only in panels:
            sv = arm_value(spec, arm, scope, a.metric, only)
            gv = arm_value(gen, arm, scope, a.metric, only)
            if sv is None or gv is None:
                ok = False
                break
            row[key] = (gv, sv)
        if ok:
            pts.append(row)
        else:
            print(f"  skipping {arm}: not present in the scored files")

    if a.lim:
        lo, hi = (float(v) for v in a.lim.split(","))
    else:
        vals = [v for r in pts for key, _s, _sc, _o in panels for v in r[key]]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.12 or 0.1
        lo, hi = lo - pad, hi + pad
    print(f"axis limits: {lo:.3f} .. {hi:.3f}")

    label = {"R2": "$R^2$ vs identity", "Pearson": "Pearson $r$",
             "RMSE": "RMSE", "CCC": "concordance (CCC)",
             "spearman": "Spearman $\\rho$", "r2_log": "$R^2$ on $\\log_{10}$ p",
             "slope": "calibration slope"}[a.metric]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.0))
    titles = [t.strip() for t in a.panel_titles.split(",") if t.strip()]
    if titles and len(titles) != len(panels):
        raise SystemExit(f"--panel-titles has {len(titles)} entries for "
                         f"{len(panels)} panels")
    for pi, (ax, (scope, sub, _pool, _only)) in enumerate(zip(axes, panels)):
        # THE FRONTIER: the line through the two no-method regimes that sit at
        # opposite ends of the trade-off -- local-only, which serves its own
        # site, and centralized, which serves the cohort. Everything below it
        # is reachable without a method; the shaded half is what a method has
        # to clear to be worth anything.
        anchors = {r["arm"]: r[scope] for r in pts}
        names = [v for v in a.frontier.split(",") if v]
        fx, fy = None, None
        if len(names) == 2 and all(v in anchors for v in names):
            (gx1, sy1), (gx2, sy2) = anchors[names[0]], anchors[names[1]]
            if gx2 != gx1:
                slope = (sy2 - sy1) / (gx2 - gx1)
                icept = sy1 - slope * gx1
                # SAMPLED, not two endpoints. The frontier is straight in DATA
                # space, but matplotlib joins the points it is given with
                # segments in DISPLAY space. On a symlog axis those differ, and
                # a two-point line visibly misses its own anchors. Sampling
                # densely makes the polyline follow the true line on any scale.
                fx = [lo + (hi - lo) * i / 512 for i in range(513)]
                fy = [icept + slope * x for x in fx]
        if fx:
            ax.fill_between(fx, [lo] * len(fx), fy, color="#4C72B0", alpha=0.085,
                            linewidth=0, zorder=0)
            ax.plot(fx, fy, ls="--", lw=1.3, color="#6E757C", zorder=1)

        # Same reasoning for y = x.
        _d = [lo + (hi - lo) * i / 512 for i in range(513)]
        ax.plot(_d, _d, ls=":", lw=0.9, color="#C9CFD4", zorder=1)
        # Off-scale arms are clamped to the axis and listed in a stacked
        # caption instead of annotated in place: three arms clamp to the same
        # corner, so in-place labels overprint each other into mush.
        offscale = []
        for r in pts:
            gv, sv = r[scope]
            clipped = not (lo <= gv <= hi and lo <= sv <= hi)
            ax.scatter(min(max(gv, lo), hi), min(max(sv, lo), hi),
                       s=(118 if r["marker"] != "*" else 230) * a.marker_scale,
                       marker=r["marker"],
                       facecolor=r["colour"] if r["filled"] else "white",
                       edgecolor="white" if r["filled"] else r["colour"],
                       linewidth=1.1 if r["filled"] else 1.8, zorder=3,
                       alpha=0.4 if clipped else 1.0)
            if clipped:
                offscale.append(r)
        if offscale:
            ax.text(0.028, 0.028 + 0.052 * len(offscale), "off scale:",
                    transform=ax.transAxes, fontsize=7.5, color="#7A828B",
                    style="italic", zorder=4)
            for i, r in enumerate(reversed(offscale)):
                gv, sv = r[scope]
                ax.text(0.028, 0.028 + 0.052 * i,
                        f"{r['label']}  ({gv:.1f}, {sv:.1f})",
                        transform=ax.transAxes, fontsize=7.5,
                        color=r["colour"], zorder=4)
        if a.logscale:
            # Symlog, not log: several arms are negative and a log axis would
            # drop them without a word.
            ax.set_xscale("symlog", linthresh=a.linthresh)
            ax.set_yscale("symlog", linthresh=a.linthresh)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles[pi] if titles else f"Prevalence {label}, {sub}",
                     fontsize=10.5)
        ax.set_xlabel(a.xlabel.replace("\\n", "\n"), fontsize=9)
        ax.grid(True, color="#E6E9EC", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("#C9CFD4")
        ax.tick_params(labelsize=8.5, color="#C9CFD4")
    axes[0].set_ylabel(a.ylabel.replace("\\n", "\n"), fontsize=9)
    if not a.no_region_text:
        # The frontier runs from local-only (top left) down to centralized
        # (bottom right), so the region above it is the top-right corner.
        for ax in axes:
            ax.text(0.965, 0.965, a.region_above, transform=ax.transAxes,
                    fontsize=8, color="#7A828B", va="top", ha="right")
            ax.text(0.965, 0.035, a.region_below, transform=ax.transAxes,
                    fontsize=8, color="#7A828B", ha="right")

    handles = [Line2D([], [], marker=r["marker"], color="none",
                      markerfacecolor=r["colour"] if r["filled"] else "white",
                      markeredgecolor="white" if r["filled"] else r["colour"],
                      markeredgewidth=1.1 if r["filled"] else 1.8,
                      markersize=10 if r["marker"] != "*" else 14,
                      label=r["label"]) for r in pts]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, a.legend_y), handletextpad=0.4,
               columnspacing=1.4)
    if a.title:
        fig.suptitle(a.title, fontsize=12)
    # Room for a two-line x label plus two or three legend rows; without it the
    # legend prints on top of the axis label.
    fig.tight_layout(rect=(0, a.bottom, 1, 1 if not a.title else 0.95))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)
    for r in pts:
        cells = "   ".join(
            f"{key} {r[key][1]:+.3f}/{r[key][0]:+.3f}"
            for key, _sub, _pool, _only in panels)
        print(f"  {r['label']:30s} {cells}")


if __name__ == "__main__":
    main()
