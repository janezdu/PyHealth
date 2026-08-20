"""How far did fine-tuning move each hospital away from the federated global?

``fedavg_ft`` scores *below* ``local`` on rare-code TSTR, which is backwards:
warm-starting from 49 rounds of federated training ought to beat training a site
in isolation. Two explanations point in opposite directions, and the fix for one
makes the other worse -- so measure before choosing.

**Not enough personalization.** Fine-tuning runs ``ft_epochs`` (2) epochs at
batch size 64. Hospital 438 holds 79 train records, so it takes **4 gradient
steps** -- against 66 for hospital 420. If drift is negligible, ``fedavg_ft`` is
barely distinguishable from ``fedavg``, and the answer is more fine-tuning, not
less.

**Too much drift.** If the 8 sites move far and in mutually incoherent
directions, local training is destroying federated structure rather than
specializing it. That is the client-drift picture FedProx exists to fix -- but
note FedProx belongs in the *federated rounds*, constraining each client toward
the server model every round. Adding a proximal term to the final fine-tuning
would just undo the personalization on purpose.

What is measured
----------------
``||theta_i - theta_global||``        absolute drift, per hospital
relative drift                        the same over ``||theta_global||``, which
                                      is what makes sites comparable
cosine between UPDATE VECTORS         ``delta_i = theta_i - theta_global``,
                                      pairwise. This is the client-drift number.
per-layer drift                       where the movement concentrates

Cosine between the raw weight vectors is deliberately *not* the headline: eight
models warm-started from the same global sit at ~0.9999 to each other no matter
what happened, so it cannot distinguish the two explanations. It is reported
only as a sanity check. The deltas are where the signal is.

CPU only, no GPU. Loads 9 checkpoints of ~36 MB, about a minute.
"""

import os
import statistics
from typing import Dict, List

import torch

from utils.cohort import load_manifest
from utils.eda.common import table, write_json

NAME = "drift"
SUMMARY = ("weight-space drift of each fine-tuned hospital model from the "
           "federated global (loads checkpoints; ~1 min)")

DEFAULTS = {
    # run directory holding fedavg_state.pt and ft_<hid>.pt
    "save_dir": "_outputs/fedavg_ft_full_strat8_random_save",
    # how many layers to show in the per-layer table
    "top_layers": 12,
}


def load_state(path: str) -> Dict[str, torch.Tensor]:
    """Read a checkpoint's weights.

    Both file kinds store their weights under ``global_state`` -- ``_save_ckpt``
    reuses the key for the per-hospital fine-tuned models too, so the name is
    misleading for ``ft_<hid>.pt`` but the layout is identical.
    """
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("global_state", ckpt)
    return {k: v.float() for k, v in state.items()
            if isinstance(v, torch.Tensor) and v.is_floating_point()}


def flatten(state: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    """One long vector, in a fixed key order so every model lines up."""
    return torch.cat([state[k].reshape(-1) for k in keys])


def run(cfg: dict) -> dict:
    """Compare each ``ft_<hid>.pt`` against the ``fedavg_state.pt`` it started from."""
    save_dir = cfg["save_dir"]
    gpath = os.path.join(save_dir, "fedavg_state.pt")
    if not os.path.exists(gpath):
        raise SystemExit(f"no fedavg_state.pt in {save_dir}")
    gckpt = torch.load(gpath, map_location="cpu")
    rounds = gckpt.get("completed_rounds")
    glob = load_state(gpath)
    keys = sorted(glob)

    hids = sorted(f[3:-3] for f in os.listdir(save_dir)
                  if f.startswith("ft_") and f.endswith(".pt"))
    if not hids:
        raise SystemExit(
            f"no ft_<hid>.pt in {save_dir}. Only fedavg_ft runs write "
            "them, and only since per-hospital checkpointing was added -- an "
            "older run has its fine-tuned models only in RAM, and they are gone."
        )

    sizes: Dict[str, int] = {}
    try:
        man = load_manifest(cfg["cohort_cache"])
        sizes = {h: man["per_hospital"][h]["n_train"]
                 for h in man["per_hospital"]}
    except Exception:                              # noqa: BLE001
        pass

    gvec = flatten(glob, keys)
    gnorm = float(torch.linalg.vector_norm(gvec))
    print(f"global: {gpath}")
    print(f"  completed_rounds={rounds}   tensors={len(keys)}   "
          f"params={gvec.numel():,}   ||theta_global||={gnorm:.4f}")

    deltas: Dict[str, torch.Tensor] = {}
    rows, per_layer = [], {}
    for hid in hids:
        st = load_state(os.path.join(save_dir, f"ft_{hid}.pt"))
        vec = flatten(st, keys)
        d = vec - gvec
        deltas[hid] = d
        l2 = float(torch.linalg.vector_norm(d))
        raw_cos = float(torch.nn.functional.cosine_similarity(
            vec, gvec, dim=0))
        rows.append([hid, sizes.get(hid, "?"),
                     f"{l2:.4f}", f"{l2 / gnorm:.5f}",
                     f"{float(torch.linalg.vector_norm(vec)):.4f}",
                     f"{raw_cos:.6f}"])
        per_layer[hid] = {
            k: float(torch.linalg.vector_norm(st[k] - glob[k]))
            / max(1e-12, float(torch.linalg.vector_norm(glob[k])))
            for k in keys
        }

    print(f"\nDrift from the round-{rounds} global "
          f"(the state every site warm-started from)")
    table(["hospital", "n_train", "L2", "relative", "||theta_i||",
           "cos(theta_i, theta_g)"], rows)
    print("  relative = ||theta_i - theta_g|| / ||theta_g||. The last column is "
          "near 1.0 by\n  construction -- shared warm start -- and is a sanity "
          "check, not a finding.")

    # The client-drift number. Two sites specializing in compatible directions
    # give positive cosine; sites pulling apart give ~0 or negative, and an
    # average of such updates cancels rather than combines.
    print("\nCosine between UPDATE VECTORS (theta_i - theta_global)")
    head = [""] + hids
    cos_rows, offdiag = [], []
    pairwise: Dict[str, Dict[str, float]] = {}
    for a in hids:
        row = [a]
        pairwise[a] = {}
        for b in hids:
            c = float(torch.nn.functional.cosine_similarity(
                deltas[a], deltas[b], dim=0))
            row.append("—" if a == b else f"{c:+.3f}")
            if a != b:
                pairwise[a][b] = c
            if a < b:
                offdiag.append(c)
        cos_rows.append(row)
    table(head, cos_rows)

    mean_cos = sum(offdiag) / max(1, len(offdiag))
    print(f"\n  mean pairwise cosine: {mean_cos:+.4f}   "
          f"min {min(offdiag):+.4f}   max {max(offdiag):+.4f}")

    rels = [float(x[3]) for x in rows]
    print("\nReading it")
    if max(rels) < 0.01:
        print(f"  Drift is TINY (max {max(rels):.4%} of the global norm). "
              "fedavg_ft is barely\n  distinguishable from fedavg -- 2 epochs "
              "moved almost nothing. A proximal\n  term would constrain "
              "something that is already not moving; raise --ft-epochs\n  "
              "instead.")
    elif mean_cos < 0.05:
        print(f"  Drift is substantial and INCOHERENT (mean pairwise cosine "
              f"{mean_cos:+.3f}).\n  Sites are moving in unrelated directions -- "
              "the client-drift picture FedProx\n  addresses. Note it belongs "
              "in the federated rounds, not the fine-tuning.")
    else:
        print(f"  Drift is substantial and largely COHERENT (mean pairwise "
              f"cosine {mean_cos:+.3f}).\n  Sites agree on direction, so "
              "averaging is not cancelling them; look elsewhere\n  for why "
              "fedavg_ft underperforms local.")

    if sizes and len(rows) > 2:
        # 2 epochs at batch 64 means hospital 438 takes 4 optimizer steps and
        # hospital 420 takes 66. If drift tracks size, the small sites are
        # simply undertrained rather than badly served by federation.
        xs = [sizes.get(r[0], 0) for r in rows]
        ys = rels
        if len(set(xs)) > 1:
            mx, my = statistics.mean(xs), statistics.mean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = ((sum((a - mx) ** 2 for a in xs) ** 0.5)
                   * (sum((b - my) ** 2 for b in ys) ** 0.5))
            print(f"\n  corr(train size, relative drift) = {num / den:+.3f}   "
                  "(positive => small sites\n  barely moved, because 2 epochs "
                  "is only a handful of steps for them)")

    print(f"\nLargest relative per-layer drift "
          f"(mean over the {len(hids)} hospitals)")
    means = {k: sum(per_layer[h][k] for h in hids) / len(hids) for k in keys}
    table(["tensor", "mean rel drift"],
          [[k, f"{v:.5f}"] for k, v in
           sorted(means.items(), key=lambda kv: -kv[1])[:cfg["top_layers"]]])

    payload = {
        "save_dir": save_dir,
        "completed_rounds": rounds,
        "global_norm": gnorm,
        "hospitals": {
            r[0]: {"n_train": sizes.get(r[0]), "l2": float(r[2]),
                   "relative": float(r[3])} for r in rows},
        "update_cosine": pairwise,
        "mean_pairwise_update_cosine": mean_cos,
        "per_layer_mean_relative_drift": means,
    }
    print(f"\nwrote {write_json(os.path.join(cfg['out'], 'drift.json'), payload)}")
    return payload
