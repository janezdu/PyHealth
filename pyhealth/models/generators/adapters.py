"""Low-rank and single-layer adapters for HALO, for parameter-efficient
per-client personalisation after federated training.

Full local fine-tuning retrains all 6.17M HALO parameters per hospital. At a site
holding 142 patients that is ~43,000 parameters per patient, and the shared
model's cross-site signal is free to be overwritten. These adapters freeze the
trunk and train a small delta instead:

===========  ==============================================  =========  ========
variant      trainable                                       params     % of FT
===========  ==============================================  =========  ========
lora_attn    LoRA on q,v of every block's attention           32,768      0.53%
last_mlp     the final block's feedforward, in full          524,288      8.49%
lora_head    LoRA on the autoregressive code head             37,792      0.61%
===========  ==============================================  =========  ========

(counts at rank 8, n_embd 256, 4 layers, total_vocab_size 925)

``lora_attn`` and ``lora_head`` adapt different things and are not
interchangeable: the first changes how the model attends over visit history, the
second changes how codes are emitted given that history. Prevalence metrics score
the emitted codes, so ``lora_head`` is aimed at the metric while ``lora_attn`` is
the more conservative edit. ``last_mlp`` is neither low-rank nor small; it is the
reference point the other two are trying to beat on parameter count.

A LoRA delta starts at exactly zero (``B`` is zero-initialised), so an adapted
model reproduces its trunk bit-for-bit before the first optimiser step. That is
what makes "adapter = 0" mean "be the shared model", and it is why an L2 penalty
on the adapter parameters is a proximal term pulling each site back toward the
federated global.
"""

from typing import Iterable, List

import torch
import torch.nn as nn


class LoRAConv1DSlices(nn.Module):
    """LoRA on selected OUTPUT SLICES of a fused :class:`Conv1D`.

    HALO packs query, key and value into one ``Conv1D(3*n_state, nx)`` whose
    weight is ``(nx, 3*n_state)``; q, k and v are consecutive column blocks of
    the output. Adapting "q and v only" therefore means adding a low-rank delta
    to two column blocks and leaving k alone -- which is why this wraps the fused
    layer rather than adapting a standalone projection.

    Args:
        base: The wrapped ``Conv1D``. Its parameters are frozen here.
        rank: LoRA rank.
        slices: ``(start, stop)`` output-column ranges to adapt.
        alpha: Scaling numerator; the delta is scaled by ``alpha / rank`` so
            changing rank does not silently change the effective step size.

    Raises:
        ValueError: If a slice falls outside the layer's output width.
    """

    def __init__(self, base: nn.Module, rank: int,
                 slices: Iterable[tuple], alpha: float = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = (alpha if alpha is not None else float(rank)) / float(rank)
        n_in = base.weight.shape[0]
        n_out = base.weight.shape[1]
        self.slices = [tuple(s) for s in slices]
        for lo, hi in self.slices:
            if lo < 0 or hi > n_out or lo >= hi:
                raise ValueError(
                    f"slice ({lo}, {hi}) does not fit an output width of "
                    f"{n_out}")
        # One (A, B) pair per adapted slice. A is random, B is ZERO, so the
        # delta starts at exactly zero and the adapted model reproduces the
        # trunk until the first step.
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(n_in, self.rank) * 0.02)
            for _ in self.slices])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.rank, hi - lo))
            for lo, hi in self.slices])

    def forward(self, x):
        out = self.base(x)
        flat = x.reshape(-1, x.shape[-1])
        # Accumulate into a FRESH zero tensor and add once, rather than writing
        # in-place into `out`. `out` is an autograd intermediate; slice-assigning
        # into it makes the graph depend on a mutated buffer, which either errors
        # or silently gives wrong gradients depending on what runs after.
        delta_full = torch.zeros_like(out)
        for (lo, hi), a, b in zip(self.slices, self.A, self.B):
            d = ((flat @ a) @ b) * self.scale
            delta_full[..., lo:hi] = d.view(*out.shape[:-1], hi - lo)
        return out + delta_full


class LoRAMaskedLinear(nn.Module):
    """LoRA on an :class:`AutoregressiveLinear`, preserving its causal mask.

    ``AutoregressiveLinear`` applies ``mask * weight`` with a lower-triangular
    mask. That mask is what makes code emission autoregressive WITHIN a visit:
    when the head produces code *i* it cannot see codes *i+1...n* of the same
    visit.

    A naive LoRA computes ``mask * W + B @ A``. ``B @ A`` is dense, so it writes
    into the masked-out upper triangle and the head can read the codes it is
    about to predict. Training loss drops, generation quality collapses, and
    nothing raises. This masks the SUM instead::

        F.linear(x, mask * (W + B @ A), bias)

    so the adapter lives strictly inside the causal triangle.
    ``tests/core/test_halo_adapters.py`` asserts the upper triangle stays zero.
    """

    def __init__(self, base: nn.Module, rank: int, alpha: float = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        if not hasattr(base, "mask"):
            raise ValueError(
                "LoRAMaskedLinear expects a masked layer (AutoregressiveLinear); "
                f"{type(base).__name__} has no 'mask' buffer. Use a plain LoRA "
                "for unmasked layers.")
        self.rank = int(rank)
        self.scale = (alpha if alpha is not None else float(rank)) / float(rank)
        out_f, in_f = base.weight.shape
        self.A = nn.Parameter(torch.randn(self.rank, in_f) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_f, self.rank))

    def effective_weight(self) -> torch.Tensor:
        """``mask * (W + BA)`` -- the tensor the forward pass actually uses.

        Exposed so a test can assert the causal mask survives adaptation.
        """
        delta = (self.B @ self.A) * self.scale
        return self.base.mask * (self.base.weight + delta)

    def forward(self, x):
        return nn.functional.linear(x, self.effective_weight(), self.base.bias)


#: Adapter variants. Keys are what ``--adapter`` accepts.
VARIANTS = ("none", "lora_attn", "last_mlp", "lora_head")


def apply_adapter(halo_model: nn.Module, variant: str, rank: int = 8
                  ) -> List[nn.Parameter]:
    """Freeze the trunk and install ``variant``'s trainable parameters.

    Args:
        halo_model: A ``HALOModel`` (the inner transformer, not the wrapper).
        variant: One of :data:`VARIANTS`. ``"none"`` trains everything, which is
            ordinary full fine-tuning and the baseline the others are measured
            against.
        rank: LoRA rank; ignored by ``last_mlp``.

    Returns:
        The parameters that should be optimised. Everything else has
        ``requires_grad = False``.

    Raises:
        ValueError: On an unknown variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown adapter {variant!r}; choose from "
                         f"{list(VARIANTS)}")
    if variant == "none":
        for p in halo_model.parameters():
            p.requires_grad_(True)
        return [p for p in halo_model.parameters()]

    for p in halo_model.parameters():
        p.requires_grad_(False)

    if variant == "lora_attn":
        blocks = halo_model.transformer.h
        for blk in blocks:
            attn = blk.attn
            n_state = attn.split_size
            # q occupies output columns [0, n_state), k [n_state, 2*n_state),
            # v [2*n_state, 3*n_state). Adapt q and v, leave k untouched --
            # the standard LoRA recipe.
            attn.c_attn = LoRAConv1DSlices(
                attn.c_attn, rank,
                slices=[(0, n_state), (2 * n_state, 3 * n_state)])
    elif variant == "last_mlp":
        # No wrapping: just unfreeze the final block's feedforward in place.
        for p in halo_model.transformer.h[-1].mlp.parameters():
            p.requires_grad_(True)
    elif variant == "lora_head":
        head = halo_model.ehr_head
        head.auto1 = LoRAMaskedLinear(head.auto1, rank)
        head.auto2 = LoRAMaskedLinear(head.auto2, rank)

    trainable = [p for p in halo_model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"adapter {variant!r} left nothing trainable")
    return trainable


def adapter_l2(halo_model: nn.Module) -> torch.Tensor:
    """Sum of squares over the trainable adapter parameters -- the FedProx term.

    With a frozen trunk the trunk cannot drift, so the only thing a proximal
    term has to restrain is the adapter, and pulling the adapter toward zero
    pulls the site back toward the federated global.

    This sums ``||A||^2 + ||B||^2`` over the trainable parameters, which is a
    DELIBERATE choice rather than a convenient approximation of
    ``||theta - theta_global||^2``. Penalising the factors is the variational
    form of the nuclear norm of the update::

        ||D||_*  =  min over D = BA of  0.5 * (||A||_F^2 + ||B||_F^2)

    so mu shrinks the update toward LOW RANK -- the site personalises along a few
    directions or not at all. The literal proximal term ``||B @ A||_F^2`` shrinks
    the update's magnitude uniformly instead. At a 142-patient site "move along
    few directions" is the better prior, which is why this form was chosen.

    Two consequences follow, and neither is a bug:

    * At init ``B`` is zero but ``A`` is random, so the DELTA is zero while this
      penalty is positive. mu begins by shrinking ``A`` before the site has moved
      from the global at all.
    * ``A -> cA, B -> B/c`` leaves the delta unchanged but changes the penalty.
      That is the mechanism: among all factorisations of one delta, this prefers
      the balanced one, which is what makes the minimum equal the nuclear norm.

    ``last_mlp`` is a further exception: its parameters do not start at zero at
    all, so there this is plain weight decay and not proximal in any sense.
    Prefer mu = 0 for that variant unless weight decay is what you want.

    Returns:
        A scalar tensor; zero (with grad) when nothing is trainable.
    """
    terms = [(p ** 2).sum() for p in halo_model.parameters() if p.requires_grad]
    if not terms:
        return torch.zeros((), device=next(halo_model.parameters()).device)
    return torch.stack(terms).sum()
