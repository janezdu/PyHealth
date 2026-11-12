"""Regularizers that pull a client's model toward a federated reference.

Two mechanisms, shared by four methods:

``ProxRef``
    ``||theta - theta_ref||^2`` against frozen weights. FedProx (Li et al.,
    MLSys 2020) applies it during federated rounds against the round's broadcast
    weights; Ditto (Li et al., ICML 2021) applies it during per-client
    fine-tuning against the final global model. Same penalty, different
    reference and phase.

``binary_kl``
    Per-code distillation against a frozen teacher's output distribution. PerAda
    (Xie et al., CVPR 2024) uses it to hold a personalized adapter near the
    global model.

WHY THE REFERENCE IS NOT A BUFFER. The obvious implementation registers the
reference weights with ``register_buffer``, which puts them in ``state_dict()``
-- where FedAvg then averages them, ``load_state_dict`` on a fresh model rejects
them, and every existing checkpoint becomes incompatible. ``_DeltaRef`` in
``adapters.py`` has exactly that shape and documents the resulting care needed.
Here the reference is a plain Python object assigned to ``_fed_prox``:
``nn.Module.__setattr__`` only diverts ``Parameter`` and ``Module`` values (and
tensors whose name is already a registered buffer), so an ordinary object lands
in ``__dict__`` and never appears in the state dict at all.

The cost of that choice is that ``.to(device)`` is not automatic -- the caller
moves the reference once, after the model is on its device.
"""

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn


class ProxRef:
    """Frozen reference weights for a proximal penalty.

    Pairs by parameter NAME against the live module, so a mismatch is an error
    rather than a silently skipped term -- a proximal penalty that quietly
    covers half the model is worse than one that fails.

    Args:
        module: The live module whose parameters will be penalized.
        ref_state: Reference weights. Keys may carry a ``halo_model.`` prefix
            (a wrapper-level ``state_dict``) or not (module-level); both are
            accepted, since callers hold whichever is to hand.
        trainable_only: Penalize only parameters with ``requires_grad``. True is
            right for an adapter run, where the frozen trunk contributes a
            constant; False penalizes everything.

    Raises:
        KeyError: If a parameter to be penalized has no reference weight.
    """

    def __init__(self, module: nn.Module, ref_state: Dict[str, torch.Tensor],
                 trainable_only: bool = True):
        self.trainable_only = trainable_only
        # Accept either naming: strip a leading "halo_model." when present.
        flat: Dict[str, torch.Tensor] = {}
        for k, v in ref_state.items():
            flat[k[len("halo_model."):] if k.startswith("halo_model.") else k] = v

        self._ref: Dict[str, torch.Tensor] = {}
        missing = []
        for name, p in module.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if name not in flat:
                missing.append(name)
                continue
            self._ref[name] = flat[name].detach().clone()
        if missing:
            raise KeyError(
                f"no reference weights for {len(missing)} parameter(s), e.g. "
                f"{missing[:3]}. The reference state dict does not match this "
                f"module; a proximal term over only part of the model is not "
                f"the penalty the method specifies."
            )
        self._module = module

    def to(self, device) -> "ProxRef":
        """Move the reference. Must be called after the module is on ``device``."""
        self._ref = {k: v.to(device) for k, v in self._ref.items()}
        return self

    def pairs(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for name, p in self._module.named_parameters():
            ref = self._ref.get(name)
            if ref is not None:
                yield p, ref

    def penalty(self) -> torch.Tensor:
        """``sum_k ||theta_k - theta_ref_k||^2`` -- no 1/2, the caller scales."""
        terms = [((p - ref) ** 2).sum() for p, ref in self.pairs()]
        if not terms:
            raise RuntimeError("ProxRef covers no parameters")
        return torch.stack(terms).sum()

    def distance(self) -> float:
        """``||theta - theta_ref||`` as a float, for logging how far a client drifted."""
        with torch.no_grad():
            return float(torch.sqrt(self.penalty()).item())

    def __len__(self) -> int:
        return len(self._ref)


def set_prox_ref(module: nn.Module, ref_state: Dict[str, torch.Tensor],
                 trainable_only: bool = True) -> ProxRef:
    """Attach a reference to ``module``. Plain attribute, not a buffer."""
    ref = ProxRef(module, ref_state, trainable_only=trainable_only)
    module._fed_prox = ref
    return ref


def clear_prox_ref(module: nn.Module) -> None:
    """Drop the reference. Safe when none is set."""
    if hasattr(module, "_fed_prox"):
        del module._fed_prox


def get_prox_ref(module: nn.Module) -> ProxRef:
    return getattr(module, "_fed_prox", None)


def prox_penalty(module: nn.Module) -> torch.Tensor:
    """The penalty, or a differentiable zero when no reference is set."""
    ref = get_prox_ref(module)
    if ref is None:
        p = next(module.parameters())
        return torch.zeros((), device=p.device, dtype=p.dtype)
    return ref.penalty()


def binary_kl(p_student: torch.Tensor, p_teacher: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Mean per-code binary KL, ``KL(teacher || student)``.

    HALO emits every code as an independent Bernoulli through a sigmoid, so the
    distribution over a visit is a product of per-code Bernoullis and the KL
    decomposes into a sum of binary KLs. Averaging over codes and positions
    matches how the BCE loss it joins is reduced, which keeps ``kd_lambda`` on a
    comparable scale to the risk.

    Positions outside a patient's real visits are already zeroed by
    ``HALOModel.forward`` before this is called, so both sides are 0 there and
    the term contributes exactly 0 -- no separate mask is threaded through.

    Args:
        p_student: Probabilities from the model being trained.
        p_teacher: Probabilities from the frozen teacher. Same shape.
        eps: Clamp keeping the logs finite at the 0/1 ends.

    Returns:
        Scalar.

    Raises:
        ValueError: On a shape mismatch -- which would otherwise broadcast into
            a number that looks like a loss.
    """
    if p_student.shape != p_teacher.shape:
        raise ValueError(
            f"KD shape mismatch: student {tuple(p_student.shape)} vs teacher "
            f"{tuple(p_teacher.shape)}")
    q = p_teacher.clamp(eps, 1.0 - eps)
    p = p_student.clamp(eps, 1.0 - eps)
    kl = q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()
    return kl.mean()
