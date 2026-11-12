"""Does adding synthetic data to a site's own real records help it?

A recut of the utility panel in viz/specialist_generalist_b200.html. That panel
pairs each arm's TSTR score (train on synthetic ALONE) against its RLSP score
(the site's own real records PLUS that arm's synthetic). Useful for asking "how
good is the synthetic data on its own", which is not the question a hospital
actually has.

This figure asks the hospital's question instead:

    You have your own real records. Does bolting synthetic data on top make
    your downstream classifier better?

So the baseline bar is REAL ONLY -- one site's own real records, nothing added
-- and every other bar is that same real data plus one arm's synthetic. The
comparison is like-for-like: every bar trains on the site's real records, and
they differ only in what was added.

WHY THE TSTR BARS ARE GONE. Synthetic-alone is a different experiment with a
different training set, so putting it in the same bar group invites reading the
grey/red gap as "synthetic helps" when it is really "real data helps". Drop it
and each bar answers one question.

WHAT IS DELIBERATELY NOT DRAWN. The real_pooled reference (all eight sites'
real records pooled) is not an option available to any single hospital -- it is
the thing federation exists because you cannot do. Showing it as a target
implies a comparison that is not on the table. The chance line at 0.5 IS kept:
this downstream task has a standing pathology where arms score BELOW chance,
and a figure that hides the line hides that.

Reads the payload straight out of the committed HTML, so it needs no scored
files and no model. CPU, seconds.

Example:
    .venv/bin/python examples/fedpyhealth/scripts/fig_rlsp_bars.py \\
        --variant up_rnn --metric auc \\
        --out _outputs/results/rlsp_real_plus_synthetic.png
"""
from __future__ import annotations

import argparse
import json
import os

#: Arm id in the payload -> how it reads as "+ <this>" on the chart.
#: Order is the drawing order, top to bottom.
ARMS = [
    # Base, then one mechanism, then combinations -- ordered so the bars read
    # as "what does adding X do", not as a ranking.
    ("fedavg",        "FedAvg"),
    ("fedavg_irm",    "IRM"),
    ("fed_xm4_do03",  "best-of-K (dropout)"),
    ("fedavg_irm_xm", "IRM + best-of-K (dropout)"),
    ("lat_xm4",       "best-of-K (latent z=16)"),
    ("lat_fed_irm",   "IRM + best-of-K (latent z=16)"),
    ("lat_lora_ft",   "IRM + best-of-K (latent) + LoRA-head fine-tune (38K, 20 ep)"),
    ("lora_head20",   "LoRA-head fine-tune only (38K, 20 ep)"),
    ("fedavg_ft",     "full fine-tune only (6.2M, 2 ep)"),
    ("st20",          "ouroboros"),
    ("st20_xm4_irm",  "ouroboros + IRM + best-of-K"),
    # kept for reference, off by default via --arms
    ("xm4",           "best-of-K K=4 - centralized"),
    ("ditto0",        "Ditto (lambda=0, = FedAvg+FT)"),
    ("ditto0.1",      "Ditto (lambda=0.1)"),
    ("ditto1",        "Ditto (lambda=1)"),
    ("ditto10",       "Ditto (lambda=10)"),
    ("fedprox",       "FedProx (mu=1e-2)"),
    ("fedrep",        "FedRep"),
    ("lgfedavg",      "LG-FedAvg"),
    ("perada",        "PerAda"),
    ("fedavg_irm1e4", "IRM rho=1e4"),
]

#: Short labels for --paper. The screen labels spell out parameter counts,
#: which a caption should carry instead: on a column-width figure they squeeze
#: the plot until the value and delta annotations collide.
PAPER_LABEL = {
    "fedavg":        "FedAvg",
    "fedavg_irm":    "+ IRM",
    "lat_xm4":       "+ best-of-K",
    "lat_fed_irm":   "+ IRM + best-of-K",
    "lat_lora_ft":   "+ IRM + best-of-K + LoRA FT",
    "lora_head20":   "+ LoRA FT only",
    "fedavg_ft":     "+ full FT only",
    "fed_xm4_do03":  "+ best-of-K (dropout)",
    "fedavg_irm_xm": "+ IRM + best-of-K (dropout)",
    "st20":          "+ self-training",
}
#: Arms drawn in the accent rather than the neutral: the ones that fine-tune.
PAPER_ACCENT = {"lat_lora_ft", "lora_head20", "fedavg_ft"}

SERIF = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif", "serif"]

BASE = "#8A9299"   # real only
ADDED = "#C44E52"  # real + synthetic
NEUTRAL_P = "#77818B"  # print: an arm that does not fine-tune
ACCENT_P = "#B8434A"   # print: an arm that does
INK = "#43494F"
RULE = "#C9CFD4"

DEFAULT_HTML = "examples/fedpyhealth/viz/specialist_generalist_b200.html"


def load_payload(path: str) -> dict:
    """Pull ``const RLSP = {...}`` out of the viz page.

    The page is self-contained by design (see viz/README.md), so the numbers
    live inline rather than in a sidecar JSON. Parsing the one line is more
    honest than duplicating the values into this script, where they would go
    stale the next time the page is regenerated.
    """
    marker = "const RLSP = "
    html = open(path).read()
    i = html.find(marker)
    if i < 0:
        raise SystemExit(
            f"{path} has no 'const RLSP = ' block. Either it is not the page "
            f"holding the utility panel, or the page was regenerated with a "
            f"different variable name."
        )
    line = html[i + len(marker):html.find("\n", i)].rstrip().rstrip(";")
    return json.loads(line)


def load_scored(paths):
    """Average scored test2 RLSP files into one payload-shaped cell.

    The page's payload stores one number per arm; the scored files store one
    number per (arm, hospital, seed). Collapsing is a plain mean over both, in
    that order, which is how the page's numbers were produced -- checked by
    reproducing fedavg 0.678 and real_local 0.421 from the IRM 4-arm file.
    """
    import statistics as st
    from collections import defaultdict

    bucket = defaultdict(list)
    clf = mix = None
    for path in paths:
        d = json.load(open(path))
        clf = d.get("classifier", "rnn")
        w, mode = d.get("rlsp_weight"), d.get("rlsp_weight_mode")
        mix = f"{int(w * 100)}/{100 - int(w * 100)} {mode}" if w else "natural"
        per_arm = defaultdict(list)
        for key, v in d["arms"].items():
            fam = v.get("family") or key
            per_arm[fam].append(v["overall"])
        for fam, hosp in per_arm.items():
            # hospital-level mean first, so a seed counts once regardless of
            # how many sites it happens to carry.
            bucket[fam].append({
                m: st.mean(h[m] for h in hosp)
                for m in ("roc_auc_macro", "ap_macro")
            })
    cell = {
        fam: {"auc": st.mean(s["roc_auc_macro"] for s in seeds),
              "ap": st.mean(s["ap_macro"] for s in seeds)}
        for fam, seeds in bucket.items()
    }
    return cell, {"clf": clf, "mix": mix}


def build_rows(cell, variant, a):
    if "real_local" not in cell:
        raise SystemExit(
            "no real_local baseline in the input, so there is nothing to add "
            "the synthetic data TO."
        )
    arms = ARMS
    if a.arms:
        want = [x for x in a.arms.split(",") if x]
        known = dict(ARMS)
        bad = [x for x in want if x not in known]
        if bad:
            raise SystemExit(
                f"unknown --arms {', '.join(bad)}; available: "
                f"{', '.join(k for k, _ in ARMS)}"
            )
        arms = [(x, known[x]) for x in want]
    rows = [("own real data only", cell["real_local"][a.metric], BASE, "real_local")]
    missing = []
    for arm, label in arms:
        key = f"rlsp:{arm}"
        if key not in cell:
            missing.append(arm)
            continue
        lab = PAPER_LABEL.get(arm, f"+ {label}") if a.paper else f"+ {label}"
        rows.append((lab, cell[key][a.metric], ADDED, arm))
    if missing:
        print(f"  no real+synthetic score for: {', '.join(missing)}")
    return rows


def report(rows, variant, a):
    metric_label = ("ROC-AUC" if a.metric == "auc"
                    else "average precision (macro)")
    print(f"{variant['clf']}, {variant['mix']} — {metric_label}")
    for label, value, *_ in rows:
        delta = value - rows[0][1]
        note = "" if label == rows[0][0] else f"   ({delta:+.3f} vs real only)"
        print(f"  {label:<32} {value:.3f}{note}")
    return metric_label, a.metric, a.out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--html", default=DEFAULT_HTML)
    p.add_argument("--json", action="append", default=None,
                   help="Read scored test2 RLSP files instead of the HTML "
                        "payload. Repeat once per seed; scores are averaged "
                        "over seeds and over the eight hospitals, exactly as "
                        "the HTML payload is built. Use this for arms that "
                        "were scored but never folded into the page.")
    p.add_argument("--variant", default="up_rnn",
                   help="Which classifier x mix cell. --list shows them all.")
    p.add_argument("--metric", default="auc", choices=["auc", "ap"],
                   help="auc is ROC-AUC; ap is macro average precision, whose "
                        "floor is the prevalence rather than 0.5.")
    p.add_argument("--paper", action="store_true",
                   help="Print styling: serif type, short labels, no delta "
                        "column, muted neutrals with only the fine-tuned arms "
                        "in the accent. The screen default keeps the long "
                        "labels and the delta annotations.")
    p.add_argument("--arms", default=None,
                   help="Comma-separated arm ids to draw, in order. Default "
                        "is all of ARMS. Use this to cut the figure down to "
                        "one comparison, e.g. --arms fedavg.")
    p.add_argument("--list", action="store_true",
                   help="Print the available variants and exit.")
    p.add_argument("--out", default="_outputs/results/rlsp_real_plus_synthetic.png")
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    if a.json:
        cell, variant = load_scored(a.json)
        rows = build_rows(cell, variant, a)
        plot(rows, variant, *report(rows, variant, a), paper=a.paper)
        print(f"\nwrote {a.out}")
        return

    blob = load_payload(a.html)
    variants = {v["key"]: v for v in blob["variants"]}

    if a.list:
        for k, v in variants.items():
            print(f"  {k:<10} {v['clf']:<8} {v['mix']}")
        return
    if a.variant not in variants:
        raise SystemExit(
            f"unknown --variant {a.variant!r}; available: "
            f"{', '.join(variants)}"
        )

    cell = blob["data"][a.variant]
    if "real_local" not in cell:
        raise SystemExit(
            f"variant {a.variant!r} has no real_local baseline, so there is "
            f"nothing to add the synthetic data TO."
        )

    arms = ARMS
    if a.arms:
        want = [x for x in a.arms.split(",") if x]
        known = dict(ARMS)
        bad = [x for x in want if x not in known]
        if bad:
            raise SystemExit(
                f"unknown --arms {', '.join(bad)}; available: "
                f"{', '.join(k for k, _ in ARMS)}"
            )
        arms = [(x, known[x]) for x in want]

    rows = [("own real data only", cell["real_local"][a.metric], BASE, "real_local")]
    missing = []
    for arm, label in arms:
        key = f"rlsp:{arm}"
        if key not in cell:
            missing.append(arm)
            continue
        lab = PAPER_LABEL.get(arm, f"+ {label}") if a.paper else f"+ {label}"
        rows.append((lab, cell[key][a.metric], ADDED, arm))
    if missing:
        print(f"  no real+synthetic score for: {', '.join(missing)}")

    v = variants[a.variant]
    metric_label = ("ROC-AUC" if a.metric == "auc"
                    else "average precision (macro)")
    print(f"{v['clf']}, {v['mix']} — {metric_label}")
    for label, value, *_ in rows:
        delta = value - rows[0][1]
        note = "" if label == rows[0][0] else f"   ({delta:+.3f} vs real only)"
        print(f"  {label:<32} {value:.3f}{note}")

    plot(rows, v, metric_label, a.metric, a.out, paper=a.paper)
    print(f"\nwrote {a.out}")


def plot(rows, variant, metric_label, metric, path, paper=False) -> None:
    """Draw the bars.

    Two styles. The screen default keeps long labels and a delta column. The
    print style drops both: at column width the value and delta annotations
    collide once a bar passes ~0.65, and a caption carries the parameter counts
    better than a y-tick can.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    arms = [r[3] if len(r) > 3 else "" for r in rows]
    base = values[0]
    ys = list(range(len(rows)))[::-1]
    span = max(values) - min(min(values), 0.5 if metric == "auc" else min(values))

    if paper:
        plt.rcParams["font.family"] = SERIF
        fig, ax = plt.subplots(figsize=(6.9, 0.40 * len(rows) + 1.15))
        bh, fs_lab, fs_val = 0.52, 10.5, 10.5
        # Neutral for everything, accent ONLY for the arms that fine-tune --
        # that contrast is the finding, so it is the one thing colour encodes.
        cols = [BASE if i == 0 else (ACCENT_P if arms[i] in PAPER_ACCENT else NEUTRAL_P)
                for i in range(len(rows))]
    else:
        fig, ax = plt.subplots(figsize=(8.6, 0.40 * len(rows) + 1.9))
        bh, fs_lab, fs_val = 0.42, 11, 10.5
        cols = [r[2] for r in rows]

    ax.barh(ys, values, height=bh, color=cols, zorder=3)

    for i, (y, val) in enumerate(zip(ys, values)):
        ax.text(val + span * 0.018, y, f"{val:.3f}", va="center", ha="left",
                fontsize=fs_val, color=INK)
        if not paper and labels[i] != labels[0]:
            ax.text(1.0, y, f"{val - base:+.3f}", va="center", ha="right",
                    fontsize=9.5, color="#A6ACB2",
                    transform=ax.get_yaxis_transform(which="grid"))

    if metric == "auc":
        ax.axvline(0.5, ls=(0, (3, 3)), lw=0.9, color="#B0B6BC", zorder=1)
        ax.text(0.5, 0.985, "chance", fontsize=8.5 if paper else 9,
                color="#A6ACB2", va="top", ha="center",
                transform=ax.get_xaxis_transform())

    lo = min(min(values), 0.5 if metric == "auc" else min(values)) - span * 0.10
    # Print style needs less right margin: no delta column to make room for.
    ax.set_xlim(max(0.0, lo), max(values) + span * (0.13 if paper else 0.30))
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=fs_lab)
    ax.get_yticklabels()[-1].set_color("#5A6168")

    ax.set_xlabel(f"{metric_label}, 30 drawn codes   \u2192 better",
                  fontsize=10 if paper else 10, color="#3A4046" if paper else "#5A6168",
                  labelpad=8)
    if not paper:
        ax.set_title("Adding synthetic data to a site's own real records",
                     fontsize=13, loc="left", pad=30, color=INK)
        ax.text(0, 1.035, f"{variant['clf']} classifier  \u00b7  {variant['mix']} mix",
                transform=ax.transAxes, fontsize=10, color="#9AA1A8", va="bottom")

    ax.grid(True, axis="x", color="#EDEFF1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=9.5, color=RULE, labelcolor="#5A6168")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300 if paper else 200, bbox_inches="tight",
                facecolor="white")
    if paper:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
