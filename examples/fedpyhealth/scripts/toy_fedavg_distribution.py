"""Does averaging two generators' WEIGHTS average their OUTPUT DISTRIBUTIONS?

The question behind the federated arms in this project: when FedAvg averages two
clients that learned different modes, does the aggregate learn the union of those
modes, or something else?

For a linear model, averaging weights averages functions. A generator is not
linear, so ``G(z; (t1+t2)/2)`` is neither ``G(z; t1)`` nor a mixture of the two --
to first order it is the POINTWISE MIDPOINT of the two generators' outputs. The
prediction that follows: average a generator that learned a mode at (0,0) with one
that learned (0,1) and you get a single mode near (0, 0.5) with SHRUNK variance,
not a bimodal union. Interpolating between two distributions is not mixing them.

Two setups, matching the question as asked:

* ``disjoint``  C1 ~ N((0,0), s^2 I), C2 ~ N((0,1), s^2 I). No shared mode.
* ``overlap``   C1 ~ {(0,0), (0,1)}, C2 ~ {(1,1), (0,1)}. One mode in common,
  one unique to each -- the case that matters for hospitals, which share common
  codes and differ in the tail.

Arms, all sampled and scored identically:

===============  ==========================================================
arm              what it is
===============  ==========================================================
c1 / c2          trained on one client only
pooled           trained on both clients' data -- the ceiling
avg_shared       ONE average of two models from a SHARED init
avg_indep        ONE average of two models from INDEPENDENT inits
fedavg_R         R rounds of (local steps -> average -> broadcast)
mixture          the true 50/50 mixture -- the "union" answer, for reference
===============  ==========================================================

``avg_shared`` vs ``avg_indep`` is the permutation-alignment question (Git
Re-Basin, Ainsworth et al. 2022): two nets can compute the same function through
permuted hidden units, and averaging those gives nonsense. FedAvg broadcasts a
common starting point every round, which is why it works at all; the independent
arm shows what is being avoided.

WHY THE SPEC'S OWN NUMBERS NEED A SECOND SIGMA. At sigma=1 with means one unit
apart, the two Gaussians overlap almost completely -- their mixture IS a single
blob, so "union" and "midpoint" predict nearly the same picture and the test
cannot separate them. The script therefore runs the requested sigma=1.0 and a
well-separated sigma=0.25, and only the latter can falsify anything.

The generator is a small MLP trained against an RBF-kernel MMD, so there is no
discriminator to destabilise and no density to normalise. Runs on CPU in about a
minute.

Example:
    python examples/fedpyhealth/scripts/toy_fedavg_distribution.py \\
        --out _outputs/results/toy_fedavg
"""

import argparse
import copy
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SEED = 0


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
SETUPS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    # No mode in common: the cleanest test of union-vs-midpoint.
    "disjoint": {"c1": [(0.0, 0.0)], "c2": [(0.0, 1.0)]},
    # One mode shared, one unique per client -- the hospital case.
    "overlap": {"c1": [(0.0, 0.0), (0.0, 1.0)],
                "c2": [(1.0, 1.0), (0.0, 1.0)]},
}


def sample_mixture(centres: Sequence[Tuple[float, float]], n: int, sigma: float,
                   rng: np.random.Generator) -> np.ndarray:
    """``n`` points from an equal-weight Gaussian mixture over ``centres``."""
    which = rng.integers(0, len(centres), size=n)
    mu = np.asarray(centres, dtype=np.float64)[which]
    return mu + sigma * rng.standard_normal((n, 2))


# --------------------------------------------------------------------------- #
# Generator                                                                    #
# --------------------------------------------------------------------------- #
class Gen(nn.Module):
    """z ~ N(0, I_2) -> 2-D sample. Small enough that averaging is meaningful."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def draw(self, n: int, gen: torch.Generator) -> torch.Tensor:
        return self.forward(torch.randn(n, 2, generator=gen))


def mmd2(x: torch.Tensor, y: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
    """Unbiased-ish squared MMD with a sum of RBF kernels.

    Several bandwidths rather than one: a single bandwidth tuned to the data can
    be blind to exactly the structure under test here -- two modes one unit
    apart look like one mode to a wide kernel.
    """
    def k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * s * s)) for s in sigmas)

    return k(x, x).mean() - 2 * k(x, y).mean() + k(y, y).mean()


BANDWIDTHS = (0.1, 0.25, 0.5, 1.0, 2.0)


class Fitter:
    """A model plus its optimiser, advanced in segments.

    Training in segments rather than one call is what makes the staged plots
    possible: two clients can be stepped in lockstep and averaged at each
    checkpoint, which is the only way to see the average DEGRADE as the clients
    drift. Holding the optimiser here keeps Adam's moments across segments, so
    a segmented run matches an unsegmented one.
    """

    def __init__(self, model: Gen, data: np.ndarray, lr: float, batch: int,
                 gen: torch.Generator):
        self.model = model
        self.real = torch.as_tensor(data, dtype=torch.float32)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.batch = batch
        self.gen = gen
        self.steps = 0

    def advance(self, steps: int) -> None:
        for _ in range(steps):
            idx = torch.randint(0, len(self.real), (self.batch,),
                                generator=self.gen)
            loss = mmd2(self.model.draw(self.batch, self.gen),
                        self.real[idx], BANDWIDTHS)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.steps += steps


def train(model: Gen, data: np.ndarray, steps: int, lr: float, batch: int,
          gen: torch.Generator) -> None:
    """Fit ``model`` to ``data`` in place by minimising MMD."""
    Fitter(model, data, lr, batch, gen).advance(steps)


def average_weights(models: Sequence[Gen], weights: Sequence[float]) -> Gen:
    """Sample-weighted mean of several models' parameters -- FedAvg's step."""
    total = float(sum(weights))
    out = copy.deepcopy(models[0])
    with torch.no_grad():
        for name, p in out.named_parameters():
            p.zero_()
            for m, w in zip(models, weights):
                p.add_(dict(m.named_parameters())[name] * (w / total))
        for name, b in out.named_buffers():
            b.zero_()
            for m, w in zip(models, weights):
                b.add_(dict(m.named_buffers())[name] * (w / total))
    return out


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def sliced_w1(a: np.ndarray, b: np.ndarray, n_proj: int, seed: int) -> float:
    """Sliced 1-Wasserstein: mean over random 1-D projections."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)
    n = min(len(pa), len(pb))
    qa = pa[np.linspace(0, len(pa) - 1, n).astype(int)]
    qb = pb[np.linspace(0, len(pb) - 1, n).astype(int)]
    return float(np.abs(qa - qb).mean())


def mode_mass(x: np.ndarray, centres: Sequence[Tuple[float, float]],
              radius: float) -> List[float]:
    """Fraction of samples within ``radius`` of each true centre.

    The union-versus-midpoint discriminator. A model that learned the union puts
    comparable mass on every centre; one that interpolated puts little on any of
    them, because the midpoint of two modes is not near either.
    """
    out = []
    for c in centres:
        d = np.linalg.norm(x - np.asarray(c), axis=1)
        out.append(float((d < radius).mean()))
    return out


# --------------------------------------------------------------------------- #
# Staged snapshots                                                             #
# --------------------------------------------------------------------------- #
def drift_stages(d1: np.ndarray, d2: np.ndarray, stages: Sequence[int],
                 per: int, args) -> List[dict]:
    """Two clients trained in lockstep, averaged and sampled at each stage.

    The question this answers: does the average get better or worse as the
    clients train longer? Both start from a SHARED init and are never
    re-broadcast, so local drift accumulates -- one long FedAvg round. If the
    average degrades while each client improves, local work and aggregation are
    pulling against each other, which is the mechanism the federated arms of
    this project keep running into.
    """
    torch.manual_seed(SEED + 3)
    base = Gen(args.hidden)
    f1 = Fitter(copy.deepcopy(base), d1, args.lr, args.batch,
                torch.Generator().manual_seed(21))
    f2 = Fitter(copy.deepcopy(base), d2, args.lr, args.batch,
                torch.Generator().manual_seed(22))

    out, prev = [], 0
    for st in stages:
        target = st * per
        f1.advance(target - prev)
        f2.advance(target - prev)
        prev = target
        avg = average_weights([f1.model, f2.model], [1, 1])
        g = torch.Generator().manual_seed(999)
        with torch.no_grad():
            out.append({
                "stage": st,
                "c1": f1.model.draw(args.n_eval, g).numpy(),
                "c2": f2.model.draw(args.n_eval, g).numpy(),
                "avg": avg.draw(args.n_eval, g).numpy(),
            })
    return out


def fedavg_stages(d1: np.ndarray, d2: np.ndarray, stages: Sequence[int],
                  args) -> List[dict]:
    """The global model after each of several FedAvg rounds.

    Same loop as the real thing: broadcast, local steps, average, repeat. The
    locals are captured JUST BEFORE aggregation, so each row shows both what the
    two sites had learned and what survived the mean.
    """
    torch.manual_seed(SEED + 4)
    glob = Gen(args.hidden)
    per_round = max(1, args.steps // args.rounds)
    want = set(stages)
    out = []
    g = torch.Generator().manual_seed(999)

    if 0 in want:
        with torch.no_grad():
            out.append({"stage": 0,
                        "c1": glob.draw(args.n_eval, g).numpy(),
                        "c2": glob.draw(args.n_eval, g).numpy(),
                        "avg": glob.draw(args.n_eval, g).numpy()})
    for r in range(1, max(stages) + 1):
        locals_ = []
        for i, d in enumerate((d1, d2)):
            lm = copy.deepcopy(glob)
            train(lm, d, per_round, args.lr, args.batch,
                  torch.Generator().manual_seed(100 + 10 * r + i))
            locals_.append(lm)
        glob = average_weights(locals_, [1, 1])
        if r in want:
            gg = torch.Generator().manual_seed(999)
            with torch.no_grad():
                out.append({
                    "stage": r,
                    "c1": locals_[0].draw(args.n_eval, gg).numpy(),
                    "c2": locals_[1].draw(args.n_eval, gg).numpy(),
                    "avg": glob.draw(args.n_eval, gg).numpy(),
                })
    return out


def plot_stages(rows: List[dict], centres, title: str, xlabel: str,
                path: str) -> None:
    """One row per stage: client 1, client 2, and their average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["c1", "c2", "avg"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6.3, 2.05 * len(rows)),
                             squeeze=False)
    for i, row in enumerate(rows):
        for j, key in enumerate(cols):
            ax = axes[i][j]
            x = row[key]
            ax.scatter(x[:, 0], x[:, 1], s=1.6, alpha=0.22, linewidths=0,
                       color="#A8522C" if key == "avg" else "#14707F")
            for c in centres:
                ax.plot(*c, marker="+", ms=9, mew=1.5, color="#15191E")
            ax.set_xlim(-2, 3)
            ax.set_ylim(-2, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title({"c1": "client 1", "c2": "client 2",
                              "avg": "averaged weights"}[key], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{xlabel} {row['stage']}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# One experiment                                                               #
# --------------------------------------------------------------------------- #
def run_setup(setup: str, sigma: float, args) -> dict:
    """Train every arm for one (setup, sigma) and score them all the same way."""
    centres = SETUPS[setup]
    rng = np.random.default_rng(SEED)
    d1 = sample_mixture(centres["c1"], args.n, sigma, rng)
    d2 = sample_mixture(centres["c2"], args.n, sigma, rng)
    pooled = np.concatenate([d1, d2])

    all_centres = sorted(set(centres["c1"]) | set(centres["c2"]))
    truth = sample_mixture(all_centres, args.n_eval, sigma, rng)

    def g(seed: int) -> torch.Generator:
        return torch.Generator().manual_seed(seed)

    def fresh(seed: int) -> Gen:
        torch.manual_seed(seed)
        return Gen(args.hidden)

    models: Dict[str, Gen] = {}

    # Independent inits -- the arm that should break.
    m1, m2 = fresh(SEED + 1), fresh(SEED + 2)
    train(m1, d1, args.steps, args.lr, args.batch, g(11))
    train(m2, d2, args.steps, args.lr, args.batch, g(12))
    models["c1"], models["c2"] = m1, m2
    models["avg_indep"] = average_weights([m1, m2], [1, 1])

    # Shared init -- one broadcast, then one average. FedAvg with R = 1.
    base = fresh(SEED + 3)
    s1, s2 = copy.deepcopy(base), copy.deepcopy(base)
    train(s1, d1, args.steps, args.lr, args.batch, g(21))
    train(s2, d2, args.steps, args.lr, args.batch, g(22))
    models["avg_shared"] = average_weights([s1, s2], [1, 1])

    # Real FedAvg: re-broadcast every round, so the clients never drift far.
    glob = fresh(SEED + 4)
    per_round = max(1, args.steps // args.rounds)
    for r in range(args.rounds):
        locals_ = []
        for i, d in enumerate((d1, d2)):
            lm = copy.deepcopy(glob)
            train(lm, d, per_round, args.lr, args.batch, g(100 + 10 * r + i))
            locals_.append(lm)
        glob = average_weights(locals_, [1, 1])
    models[f"fedavg_R{args.rounds}"] = glob

    # Ceiling: one model, all the data.
    pm = fresh(SEED + 5)
    train(pm, pooled, args.steps, args.lr, args.batch, g(31))
    models["pooled"] = pm

    radius = max(0.45 * sigma * 2.2, 0.35)
    rows = []
    samples = {}
    for name, m in models.items():
        with torch.no_grad():
            x = m.draw(args.n_eval, g(999)).numpy()
        samples[name] = x
        rows.append({
            "arm": name,
            "mean": [round(float(v), 3) for v in x.mean(0)],
            "std": [round(float(v), 3) for v in x.std(0)],
            "mode_mass": [round(v, 3) for v in mode_mass(x, all_centres, radius)],
            "sw1_to_truth": round(sliced_w1(x, truth, args.n_proj, SEED), 4),
        })
    rows.append({
        "arm": "mixture (truth)",
        "mean": [round(float(v), 3) for v in truth.mean(0)],
        "std": [round(float(v), 3) for v in truth.std(0)],
        "mode_mass": [round(v, 3) for v in mode_mass(truth, all_centres, radius)],
        "sw1_to_truth": 0.0,
    })
    samples["mixture"] = truth
    res = {"setup": setup, "sigma": sigma, "centres": all_centres,
           "radius": round(radius, 3), "rows": rows, "_samples": samples,
           "_d1": d1, "_d2": d2}
    if args.stages:
        st = [int(v) for v in args.stages.split(",")]
        res["_drift"] = drift_stages(
            d1, d2, [v for v in st if v > 0],
            max(1, args.steps // args.rounds), args)
        res["_fed"] = fedavg_stages(
            d1, d2, [v for v in st if v <= args.rounds], args)
    return res


def report(res: dict) -> None:
    """Print one experiment as a table."""
    cs = " ".join(f"({a:g},{b:g})" for a, b in res["centres"])
    print(f"\n=== {res['setup']}  sigma={res['sigma']}  modes {cs} ===")
    print(f"{'arm':18s} {'mean':>16s} {'std':>16s} "
          f"{'mass within ' + str(res['radius']):>26s} {'SW1':>8s}")
    for r in res["rows"]:
        mean = f"({r['mean'][0]:+.2f},{r['mean'][1]:+.2f})"
        std = f"({r['std'][0]:.2f},{r['std'][1]:.2f})"
        mass = " ".join(f"{v:.2f}" for v in r["mode_mass"])
        print(f"{r['arm']:18s} {mean:>16s} {std:>16s} {mass:>26s} "
              f"{r['sw1_to_truth']:8.4f}")


def plot(results: List[dict], path: str) -> None:
    """Scatter every arm against the truth. Optional; needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [r["arm"] for r in results[0]["rows"] if r["arm"] != "mixture (truth)"]
    order = ["c1", "c2", "pooled"] + [a for a in arms
                                      if a not in ("c1", "c2", "pooled")]
    fig, axes = plt.subplots(len(results), len(order) + 1,
                             figsize=(2.05 * (len(order) + 1), 2.2 * len(results)),
                             squeeze=False)
    for i, res in enumerate(results):
        for j, name in enumerate(order + ["mixture"]):
            ax = axes[i][j]
            x = res["_samples"][name]
            ax.scatter(x[:, 0], x[:, 1], s=1.5, alpha=0.25, linewidths=0,
                       color="#14707F" if name != "mixture" else "#A8522C")
            for c in res["centres"]:
                ax.plot(*c, marker="+", ms=8, mew=1.4, color="#15191E")
            ax.set_xlim(-2, 3)
            ax.set_ylim(-2, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{res['setup']}\nsigma={res['sigma']}", fontsize=7)
    fig.suptitle("Weight averaging interpolates modes; it does not mix them",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000,
                   help="samples per client (the spec's 1000)")
    p.add_argument("--n-eval", type=int, default=4000)
    p.add_argument("--sigmas", default="1.0,0.25",
                   help="1.0 is the spec; 0.25 separates the modes enough that "
                        "union and midpoint predict different pictures")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--rounds", type=int, default=20,
                   help="FedAvg rounds; local steps per round is --steps/--rounds")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-proj", type=int, default=128)
    p.add_argument("--out", default="_outputs/results/toy_fedavg",
                   help="prefix for the .json and .png")
    p.add_argument("--stages", default="0,1,4,10,20",
                   help="checkpoints for the staged figures. Read as FedAVG "
                        "ROUNDS for the fedavg grid, and as multiples of "
                        "--steps/--rounds local steps for the drift grid, so "
                        "the two grids line up on the x axis. Empty disables "
                        "both.")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    results = []
    for setup in SETUPS:
        for sigma in (float(s) for s in args.sigmas.split(",")):
            res = run_setup(setup, sigma, args)
            report(res)
            results.append(res)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as fh:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in results], fh, indent=1)
    print(f"\nwrote {args.out}.json")
    if not args.no_plot:
        try:
            plot(results, args.out + ".png")
            for r in results:
                tag = f"{r['setup']}_s{r['sigma']:g}"
                if r.get("_fed"):
                    plot_stages(
                        r["_fed"], r["centres"],
                        f"FedAvg rounds -- {r['setup']}, sigma={r['sigma']:g}",
                        "round", f"{args.out}_fed_{tag}.png")
                if r.get("_drift"):
                    per = max(1, args.steps // args.rounds)
                    plot_stages(
                        r["_drift"], r["centres"],
                        f"One long round: drift without re-broadcast -- "
                        f"{r['setup']}, sigma={r['sigma']:g}",
                        f"round-equivalent", f"{args.out}_drift_{tag}.png")
        except ImportError:
            print("matplotlib missing -- skipped the figures")

    print("""
HOW TO READ IT
  mode_mass is the discriminator. A model that learned the UNION puts comparable
  mass on every listed mode. A model that INTERPOLATED puts little on any of
  them, because the midpoint of two modes is not near either one -- look for
  avg_* rows with low mass everywhere but a mean sitting between the centres,
  and a std SMALLER than the mixture's.

  avg_indep vs avg_shared isolates permutation alignment: same averaging, but
  one starts the two clients from a common point. If avg_indep is far worse,
  that is the Git Re-Basin effect, and it is why FedAvg re-broadcasts.

  Compare fedavg_R against avg_shared. Frequent re-broadcasting keeps the
  clients close enough that averaging is not destructive -- but keeping them
  close is exactly what stops either from committing to its own mode.""")


if __name__ == "__main__":
    main()
