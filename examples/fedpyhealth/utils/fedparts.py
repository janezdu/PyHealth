"""Splitting a model's parameters into shared and per-client halves.

FedAvg averages every key. Several personalized-FL methods differ from it in
exactly one way -- they hold some parameters back from the average and keep them
at the client -- and differ from each other only in WHICH keys those are:

======================  =========================================================
method                  personal keys
======================  =========================================================
FedRep                  the output head; the body is shared
  (Collins et al. 2021)
LG-FedAvg               the body; the head is shared
  (Liang et al. 2020)
======================  =========================================================

So the mechanism is one key predicate plus somewhere to keep the held-back
slice between rounds, and the two methods are two entries in a table.

WHY THE STORE IS THE HARD PART. ``run_fedavg`` reloads the broadcast global
state into the model at the top of every client's turn. That is correct for
FedAvg -- there is nothing to preserve -- but it silently destroys personal
parameters, and the failure is invisible: the run completes, the numbers look
plausible, and the method has quietly been FedAvg all along. The caller must
therefore merge its stored slice back in at that point, and must persist the
store in the checkpoint so a preempted run does not resume as FedAvg.

Keys are HALO-wrapper keys, i.e. ``halo_model.transformer.*`` and
``halo_model.ehr_head.*`` -- the wrapper holds the inner ``HALOModel`` as
``halo_model``, so every state-dict key carries that prefix.
"""

import re
from typing import Callable, Dict, Set, Tuple

import torch

# Predicate per mode, over state-dict keys. Add a method by adding a row.
PERSONAL_RULES: Dict[str, Callable[[str], bool]] = {
    # Plain FedAvg: nothing is held back.
    "none": lambda k: False,
    # FedRep: the autoregressive code head stays local, the trunk is shared.
    "head": lambda k: ".ehr_head." in k or k.startswith("ehr_head."),
    # LG-FedAvg: the trunk stays local, the head is shared.
    "body": lambda k: ".transformer." in k or k.startswith("transformer."),
    # Kept for completeness; on HALO this selects only LayerNorm affine
    # parameters, a fraction of a percent of the model, so a method built on it
    # has almost nothing to act on here. Not wired to a regime.
    "norm": lambda k: bool(re.search(r"(^|\.)(ln_\d+|ln_f)\.", k)),
}

MODES = tuple(PERSONAL_RULES)


def personal_keys(state: Dict[str, torch.Tensor], mode: str) -> Set[str]:
    """Which keys of ``state`` this mode keeps at the client.

    Args:
        state: A model ``state_dict``.
        mode: One of :data:`MODES`.

    Returns:
        The subset of keys that must not be averaged.

    Raises:
        ValueError: If ``mode`` is unknown, or if a mode that should select
            something selects nothing -- which means the key naming has drifted
            and the run would silently degrade to FedAvg.
    """
    if mode not in PERSONAL_RULES:
        raise ValueError(
            f"unknown personal_mode {mode!r}; expected one of {MODES}")
    keep = {k for k in state if PERSONAL_RULES[mode](k)}
    if mode != "none" and not keep:
        raise ValueError(
            f"personal_mode={mode!r} matched no keys out of {len(state)}. The "
            "state-dict naming has changed; fix the predicate in "
            "utils/fedparts.py rather than letting the run become FedAvg."
        )
    return keep


def split_state(
    state: Dict[str, torch.Tensor], keys: Set[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Partition a state dict into ``(shared, personal)`` by key."""
    shared = {k: v for k, v in state.items() if k not in keys}
    personal = {k: v for k, v in state.items() if k in keys}
    return shared, personal


def merge_state(
    shared: Dict[str, torch.Tensor], personal: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Recombine the two halves into a dict ``load_state_dict`` will accept.

    ``personal`` wins on any overlap. An empty ``personal`` gives back
    ``shared`` unchanged, which is what makes round 0 -- where no client has a
    stored slice yet -- start every client from the same broadcast weights.
    """
    merged = dict(shared)
    merged.update(personal)
    return merged


def describe(state: Dict[str, torch.Tensor], keys: Set[str]) -> str:
    """``'head: 2.85M/6.17M params (46.1%)'`` -- log this, it is a finding.

    A mode that holds back a negligible share of the model cannot do much, and
    knowing that before reading the metrics saves mistaking "the method did
    nothing" for "the method is broken".
    """
    total = sum(v.numel() for v in state.values())
    held = sum(v.numel() for k, v in state.items() if k in keys)
    pct = 100.0 * held / total if total else 0.0
    return f"{held / 1e6:.2f}M/{total / 1e6:.2f}M params ({pct:.1f}%)"
