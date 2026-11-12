"""DP-FedAvg: what the privacy noise costs, measured.

Three panels, because the headline number hides the actual finding:

  LEFT   Does the code ORDERING survive? Pearson against epsilon.
         This is the panel that matters. The control
         sits at 0.86/0.98; every DP arm is at ~0, INCLUDING sigma=0.5 whose
         epsilon of 50 is not a guarantee anyone would report. Pearson near
         zero means the synthetic prevalences have no relationship to the real
         ones -- the model learned nothing, as opposed to learning the wrong
         scale.

  MIDDLE Are the RATES right? Prevalence R^2 on a symlog axis, the scale the
         rest of the project reports: -1,400 to -3,000 against a control near 0. Symlog
         because a linear axis cannot hold a control at +0.1 and an arm at
         -3,057 at once.

  RIGHT  WHY it fails, worked out before the runs. Noise is added to each of d=6.2M
         parameters at std sigma*C/n, while the update spreads a length of at
         most C across all of them, so a typical parameter moves C/sqrt(d).
         The ratio n/(sigma*sqrt(d)) has no C in it: 8 hospitals against
         sqrt(6.2M)=2490. The measured collapse follows this curve's shape.

Example:
    .venv/bin/python examples/fedpyhealth/scripts/fig_dp.py \\
        --out _outputs/results/dp_cost.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dpaccount import epsilon          # noqa: E402
from utils.dpfed import effective_sigma      # noqa: E402

#: (arm, label, sigma, adaptive) -- sigma None marks the no-DP control.
ARMS = [
    ("dp_ctrl_fedavg",   "no DP",            None, False),
    ("dp_s0.5",          r"$\sigma$=0.5",    0.5,  False),
    ("dp_s1.0",          r"$\sigma$=1.0",    1.0,  False),
    ("dp_s1.0_adaptive", r"$\sigma$=1.0 adaptive clip", 1.0, True),
    ("dp_s2.0",          r"$\sigma$=2.0",    2.0,  False),
]
D, N_CLIENTS, ROUNDS = 6_198_104, 8, 10
ACCENT, NEUTRAL, CTRL = "#B8434A", "#77818B", "#2F6F5E"


def load(stem):
    return (json.load(open(f"_outputs/results/tests/{stem}_spec_b200.json"))["runs"],
            json.load(open(f"_outputs/results/tests/{stem}_gen_b200.json"))["runs"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", default="test1_dp")
    p.add_argument("--out", default="_outputs/results/dp_cost.png")
    a = p.parse_args(argv)

    S, G = load(a.stem)
    m = lambda d, arm, met: st.fmean(
        v[f"PrevVal_All_Prevalence_{met}"][0] for v in d[arm].values())

    rows = []
    for arm, lab, sig, adapt in ARMS:
        if arm not in S:
            print(f"  skipping {lab}: not scored")
            continue
        if sig is None:
            eps = None
        else:
            # The adaptive quantile bit is a second release, so its RDP adds.
            se = effective_sigma(sig, 4.0) if adapt else sig
            eps = epsilon(se, ROUNDS)[0]
        rows.append(dict(lab=lab, sig=sig, adapt=adapt, eps=eps,
                         ps=m(S, arm, "Pearson"), pg=m(G, arm, "Pearson"),
                         rs=m(S, arm, "R2"), rg=m(G, arm, "R2")))
        print(f"  {lab:<28} eps={'-' if eps is None else f'{eps:6.1f}'}  "
              f"Pearson {rows[-1]['ps']:+.3f}/{rows[-1]['pg']:+.3f}  "
              f"R2 {rows[-1]['rs']:>9.1f}/{rows[-1]['rg']:>9.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    dp = [r for r in rows if r["eps"] is not None]
    ctrl = next((r for r in rows if r["eps"] is None), None)

    # ---- left: Pearson vs epsilon -------------------------------------------
    ax = axes[0]
    ax.axhline(0, lw=1.0, color="#C9CFD4", zorder=1)
    for key, mk, lab in (("pg", "o", "generalist"), ("ps", "s", "specialist")):
        ax.plot([r["eps"] for r in dp], [r[key] for r in dp], mk + "-",
                color=ACCENT if key == "pg" else NEUTRAL, lw=1.4, ms=7,
                label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"privacy budget $\varepsilon$  ($\delta$=1e-5, 10 rounds)",
                  fontsize=9.5)
    ax.set_ylabel("Prevalence Pearson $r$", fontsize=9.5)
    ax.set_title("Does the code ordering survive?  (Pearson $r$)", fontsize=10.5)
    ax.set_ylim(-0.45, 1.05)
    # Control lines drawn AFTER set_xscale, and labelled in axes fractions --
    # data coords here depend on a log xlim that does not exist until the
    # scale is set, which is how the first version put the text off-canvas.
    if ctrl:
        for v, ls, lab in ((ctrl["pg"], "--", "no DP, generalist"),
                           (ctrl["ps"], ":", "no DP, specialist")):
            ax.axhline(v, ls=ls, lw=1.4, color=CTRL, zorder=1)
            ax.text(0.98, v, lab, transform=ax.get_yaxis_transform(),
                    va="bottom", ha="right", fontsize=8.5, color=CTRL)
    ax.legend(fontsize=8.5, frameon=False, loc="center left")

    # ---- middle: R^2 --------------------------------------------------------
    ax = axes[1]
    xs = range(len(rows))
    ax.bar([x - .2 for x in xs], [r["rs"] for r in rows], .38,
           color=[CTRL if r["eps"] is None else NEUTRAL for r in rows])
    ax.bar([x + .2 for x in xs], [r["rg"] for r in rows], .38,
           color=[CTRL if r["eps"] is None else ACCENT for r in rows])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["lab"] for r in rows], rotation=28, ha="right",
                       fontsize=8.5)
    ax.axhline(0, lw=1.0, color="#C9CFD4")
    ax.set_ylabel("Prevalence $R^2$ (symlog)", fontsize=9.5)
    ax.set_title("Are the rates right?  (prevalence $R^2$)", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NEUTRAL, label="specialist"),
                       Patch(color=ACCENT, label="generalist"),
                       Patch(color=CTRL, label="no DP")],
              fontsize=8.5, frameon=False, loc="lower right", ncol=3)

    # ---- right: the prediction ---------------------------------------------
    ax = axes[2]
    sig = [0.1 * 1.06 ** i for i in range(90)]
    ax.plot(sig, [N_CLIENTS / (s * math.sqrt(D)) for s in sig], lw=1.8,
            color=NEUTRAL, label=f"n={N_CLIENTS}, d=6.2M (this cohort)")
    ax.plot(sig, [N_CLIENTS / (s * math.sqrt(37_792)) for s in sig], lw=1.4,
            ls="--", color=ACCENT, label="if only a 38K adapter were noised")
    ax.axhline(1.0, lw=1.2, color="#2F6F5E", ls=":")
    ax.text(0.11, 1.15, "signal = noise", fontsize=8.5, color="#2F6F5E")
    for r in dp:
        ax.plot(r["sig"], N_CLIENTS / (r["sig"] * math.sqrt(D)), "o",
                ms=7, color=ACCENT, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"noise multiplier $\sigma$", fontsize=9.5)
    ax.set_ylabel(r"signal / noise  $= n\,/\,(\sigma\sqrt{d})$", fontsize=9.5)
    ax.set_title("Why: noise vs update, per parameter", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    for ax in axes:
        ax.grid(True, color="#EDEFF1", lw=.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("#C9CFD4")
        ax.tick_params(labelsize=8.5, color="#C9CFD4")

    fig.suptitle("Client-level DP-FedAvg on 8 hospitals: privacy cost to "
                 "the generator", fontsize=12.5, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
