"""One figure for the whole weight-averaging experiment.

Reruns the pieces of ``toy_fedavg_distribution`` that matter and lays them out
as a single image instead of ten, because the claim is one claim:

    Averaging two generators' WEIGHTS does not average their OUTPUT
    DISTRIBUTIONS. It interpolates them.

Layout, three rows on a shared frame:

  row 1  the two clients and what several ways of combining them produce,
         on the DISJOINT setup -- the cleanest test, no mode in common
  row 2  the same for OVERLAP, where one mode is shared and one is unique to
         each client, which is the hospital case
  row 3  FedAvg round by round: what each client had just before aggregation,
         and what survived the mean

Everything runs at sigma=0.25. At the sigma=1 originally proposed the two
Gaussians overlap so heavily that their mixture is a single blob, and "union"
and "midpoint" predict the same picture -- that panel cannot falsify anything,
so it is not drawn.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_fedavg_distribution import (  # noqa: E402
    SEED, SETUPS, Fitter, Gen, average_weights, mode_mass, sample_mixture,
    sliced_w1, train,
)

TEAL, CLAY, INK = "#14707F", "#A8522C", "#15191E"


def panel(ax, x, centres, title=None, ylabel=None, colour=TEAL, note=None):
    ax.scatter(x[:, 0], x[:, 1], s=1.5, alpha=0.20, linewidths=0, color=colour)
    for c in centres:
        ax.plot(*c, marker="+", ms=9, mew=1.6, color=INK)
    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-1.2, 2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#C9CFD4")
    if title:
        ax.set_title(title, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if note:
        ax.text(0.5, 0.015, note, transform=ax.transAxes, fontsize=6.6,
                ha="center", va="bottom", color="#42484F",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.78))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--n-eval", type=int, default=4000)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out", default="_outputs/results/toy_fedavg_summary.png")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    g = lambda s: torch.Generator().manual_seed(s)
    draw = lambda mdl: mdl.draw(a.n_eval, g(999)).detach().numpy()

    fig = plt.figure(figsize=(11.6, 9.4))
    gs = fig.add_gridspec(3, 6, hspace=0.22, wspace=0.08,
                          height_ratios=[1, 1, 1])
    per_round = max(1, a.steps // a.rounds)

    for row, setup in enumerate(("disjoint", "overlap")):
        cen = SETUPS[setup]
        rng = np.random.default_rng(SEED)
        d1 = sample_mixture(cen["c1"], a.n, a.sigma, rng)
        d2 = sample_mixture(cen["c2"], a.n, a.sigma, rng)
        allc = sorted(set(cen["c1"]) | set(cen["c2"]))
        truth = sample_mixture(allc, a.n_eval, a.sigma, rng)

        def fresh(seed):
            torch.manual_seed(seed)
            return Gen(a.hidden)

        m1, m2 = fresh(SEED + 1), fresh(SEED + 2)
        train(m1, d1, a.steps, a.lr, a.batch, g(11))
        train(m2, d2, a.steps, a.lr, a.batch, g(12))
        indep = average_weights([m1, m2], [1, 1])

        base = fresh(SEED + 3)
        s1, s2 = copy.deepcopy(base), copy.deepcopy(base)
        train(s1, d1, a.steps, a.lr, a.batch, g(21))
        train(s2, d2, a.steps, a.lr, a.batch, g(22))
        shared = average_weights([s1, s2], [1, 1])

        glob = fresh(SEED + 4)
        for r in range(a.rounds):
            loc = []
            for i, d in enumerate((d1, d2)):
                lm = copy.deepcopy(glob)
                train(lm, d, per_round, a.lr, a.batch, g(100 + 10 * r + i))
                loc.append(lm)
            glob = average_weights(loc, [1, 1])

        pooled = fresh(SEED + 5)
        train(pooled, np.concatenate([d1, d2]), a.steps, a.lr, a.batch, g(31))

        cells = [("client 1", draw(m1), TEAL), ("client 2", draw(m2), TEAL),
                 ("avg, shared init", draw(shared), CLAY),
                 ("avg, independent inits", draw(indep), CLAY),
                 (f"FedAvg, {a.rounds} rounds", draw(glob), CLAY),
                 ("pooled data (ceiling)", draw(pooled), TEAL)]
        for col, (name, x, colour) in enumerate(cells):
            ax = fig.add_subplot(gs[row, col])
            mm = mode_mass(x, allc, max(0.45 * a.sigma * 2.2, 0.35))
            w1 = sliced_w1(x, truth, 128, SEED)
            panel(ax, x, allc, title=name if row == 0 else None,
                  ylabel=f"{setup}\n(sigma={a.sigma:g})" if col == 0 else None,
                  colour=colour,
                  note="mass " + "/".join(f"{v:.2f}" for v in mm)
                       + f"\nW1 {w1:.3f}")
        if setup == "disjoint":
            keep = (d1, d2, allc)

    # ---- row 3: FedAvg round by round, on the disjoint setup ------------- #
    d1, d2, allc = keep
    torch.manual_seed(SEED + 4)
    glob = Gen(a.hidden)
    want = [1, 2, 5, 10, 20]
    shots = {}
    for r in range(1, max(want) + 1):
        loc = []
        for i, d in enumerate((d1, d2)):
            lm = copy.deepcopy(glob)
            train(lm, d, per_round, a.lr, a.batch, g(100 + 10 * r + i))
            loc.append(lm)
        glob = average_weights(loc, [1, 1])
        if r in want:
            shots[r] = draw(glob)
    for col, r in enumerate(want):
        ax = fig.add_subplot(gs[2, col])
        panel(ax, shots[r], allc, title=f"after round {r}" if True else None,
              ylabel="FedAvg global\n(disjoint)" if col == 0 else None,
              colour=CLAY)
    ax = fig.add_subplot(gs[2, 5])
    rng = np.random.default_rng(SEED)
    panel(ax, sample_mixture(allc, a.n_eval, a.sigma, rng), allc,
          title="the true mixture", colour=INK)

    fig.suptitle("Averaging generator weights interpolates the modes; it does "
                 "not mix the distributions", fontsize=12.5, y=0.975)
    fig.text(0.5, 0.945,
             "Crosses mark the true component means. 'mode mass' is the "
             "fraction of samples within 0.35 of each cross; W1 is sliced "
             "Wasserstein to the true mixture.",
             fontsize=8.5, ha="center", color="#5C646D")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
