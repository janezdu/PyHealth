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
last_mlp_l1  last block's feedforward, L1-sparse delta       524,288      8.49%
last_mlp_iht last block's feedforward, top-k sparse delta    524,288      8.49%
===========  ==============================================  =========  ========

(counts at rank 8, n_embd 256, 4 layers, total_vocab_size 925)

``lora_attn`` and ``lora_head`` adapt different things and are not
interchangeable: the first changes how the model attends over visit history, the
second changes how codes are emitted given that history. Prevalence metrics score
the emitted codes, so ``lora_head`` is aimed at the metric while ``lora_attn`` is
the more conservative edit. ``last_mlp`` is neither low-rank nor small; it is the
reference point the other two are trying to beat on parameter count.

``last_mlp_l1`` and ``last_mlp_iht`` train the same weights as ``last_mlp`` but
constrain the UPDATE ``D = W - W_global`` to be sparse -- by a proximal L1 step
and by periodic top-k projection respectively. Their column above is the count
they OPTIMISE; the count they finally use is that times the achieved density,
which :func:`delta_stats` reports and which nothing else in the pipeline can
infer. LoRA's prior is "personalise along few directions"; theirs is "along few
coordinates".

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
VARIANTS = ("none", "lora_attn", "last_mlp", "lora_head",
            "last_mlp_l1", "last_mlp_iht")

#: Variants that constrain the update to be SPARSE rather than low-rank. They
#: train exactly the parameters ``last_mlp`` trains, and differ only in what is
#: allowed to move: a sparse delta instead of a dense one. LoRA asks "can this
#: site personalise along a few DIRECTIONS"; these ask "along a few
#: COORDINATES". Both are complexity priors on the same fine-tuning step, which
#: is what makes them comparable.
SPARSE_VARIANTS = ("last_mlp_l1", "last_mlp_iht")


class _DeltaRef(nn.Module):
    """Frozen copy of the federated global's weights, for delta-sparse variants.

    ``last_mlp`` differs from the LoRA variants in a way that matters here: its
    parameters start at the TRUNK's values, not at zero. So "sparsify" is
    ambiguous, and the two readings are different experiments:

    * sparsify ``W`` -- force the last MLP itself to be mostly zeros. A trained
      MLP is not sparse, so this destroys what the trunk knows. It is a pruning
      experiment, not a personalisation one.
    * sparsify ``D = W - W_global`` -- the site changes a few coordinates and is
      the federated global everywhere else.

    The second is what these variants do. It restores the property plain
    ``last_mlp`` lacks -- "adapter = 0 means be the federated global" -- which is
    what makes the FedProx term meaningful for it, and it lines the variant up
    against LoRA as sparse-vs-low-rank on one axis.

    The reference is stored as BUFFERS so it round-trips through the checkpoint:
    ``generate.py`` applies the adapter before loading, so both sides carry the
    same keys, and a saved run alone is enough to recompute its final sparsity.

    Note that plain ``last_mlp`` deliberately does NOT get one of these. Adding
    buffers to it would change its state dict, and the ``last_mlp`` checkpoints
    already on disk -- saved without them -- would stop loading.
    """

    def __init__(self, params: Iterable[nn.Parameter]):
        super().__init__()
        self._names = []
        for i, p in enumerate(params):
            self.register_buffer(f"w0_{i}", p.detach().clone())
            self._names.append(f"w0_{i}")
        # A plain list attribute, not a ParameterList: these Parameters are
        # already owned by the MLP, and registering them twice would double
        # them in the state dict and in the optimiser.
        self._live = list(params)

    def pairs(self):
        """``[(live_param, w_global), ...]`` in a stable order."""
        return [(p, getattr(self, n)) for p, n in zip(self._live, self._names)]


def _ref(halo_model: nn.Module):
    """The model's :class:`_DeltaRef`, or None if this is not a sparse run."""
    return getattr(halo_model, "sparse_ft", None)


def delta_stats(halo_model: nn.Module, tol: float = 0.0) -> dict:
    """Size and sparsity of ``D = W - W_global`` over the fine-tuned weights.

    This is the run's actual evidence. Sparsity is the thing being claimed, and
    it is not observable from the loss: a subgradient L1 shrinks coordinates
    without ever landing on zero, so a run can look regularised and be fully
    dense. Logging ``nnz_frac`` per epoch is what separates the two.

    Args:
        halo_model: A ``HALOModel`` adapted with a sparse variant.
        tol: Count a coordinate as non-zero when ``|d| > tol``. The default
            ``0.0`` counts EXACT non-zeros, which is the honest measure for the
            proximal and hard-threshold paths, since both produce true zeros.
            Raise it only to ask a different question ("how many coordinates
            moved appreciably"), and say which you used.

    Returns:
        ``{"l1", "l2", "linf", "nnz", "n", "nnz_frac"}``, or ``{}`` when the
        model carries no reference (i.e. not a sparse variant).
    """
    ref = _ref(halo_model)
    if ref is None:
        return {}
    with torch.no_grad():
        flat = torch.cat([(p.detach() - w0).reshape(-1) for p, w0 in ref.pairs()])
        absd = flat.abs()
        nnz = int((absd > tol).sum().item())
        n = flat.numel()
        return {"l1": absd.sum().item(), "l2": flat.norm().item(),
                "linf": absd.max().item() if n else 0.0,
                "nnz": nnz, "n": n, "nnz_frac": nnz / n if n else 0.0}


@torch.no_grad()
def prox_l1_(halo_model: nn.Module, thresh: float) -> None:
    """Soft-threshold the delta in place -- the step that actually makes zeros.

    Adding ``lam * ||D||_1`` to the loss and letting the optimiser handle it does
    NOT sparsify. The subgradient of ``|x|`` is ``sign(x)`` away from zero, so
    the update chatters in a band of width ``lr * lam`` around zero and
    essentially never lands on it. The run then reports a sparsity of 0% and
    looks like a failed idea when it is really a failed representation. The
    proximal step is what closes it::

        D <- sign(D) * max(|D| - thresh, 0)

    Args:
        halo_model: A ``HALOModel`` adapted with a sparse variant. A model with
            no reference is left untouched.
        thresh: Soft-threshold size, normally ``lr * lam``.

    Note:
        ``thresh = lr * lam`` is the EXACT proximal operator only for plain SGD
        (this is ISTA). Under Adam the correct per-coordinate threshold carries
        a ``1/sqrt(v_hat)`` factor, so prox-Adam is a heuristic -- effective in
        practice, but not the operator its name suggests. That is the reason
        ``--adapter-optim sgd`` exists: it is the arm where the maths is exact,
        and a cross-check that the Adam arm's sparsity is not an artefact.
    """
    ref = _ref(halo_model)
    if ref is None or thresh <= 0.0:
        return
    for p, w0 in ref.pairs():
        d = p - w0
        d = torch.sign(d) * torch.clamp(d.abs() - thresh, min=0.0)
        p.copy_(w0 + d)


@torch.no_grad()
def hard_threshold_(halo_model: nn.Module, keep_frac: float,
                    optimizer=None) -> dict:
    """Project the delta onto its ``keep_frac`` largest coordinates.

    Ranked GLOBALLY across every fine-tuned tensor rather than per tensor, so
    the layer allocates its own budget instead of each weight matrix being
    forced to spend the same fraction.

    Called every few epochs rather than once at the end, which is what makes it
    ITERATIVE hard thresholding: training between projections is dense, so a
    coordinate zeroed at one projection can re-grow and survive the next. A
    single projection at the end would be ordinary one-shot pruning.

    Args:
        halo_model: A ``HALOModel`` adapted with a sparse variant.
        keep_frac: Fraction of coordinates to keep, in ``(0, 1]``.
        optimizer: The optimiser being stepped. Passing it is not optional in
            practice for a stateful optimiser -- see the warning below.

    Returns:
        :func:`delta_stats` measured immediately after the projection.

    Raises:
        ValueError: If ``keep_frac`` is outside ``(0, 1]``.

    Warning:
        Zeroing a weight but leaving Adam's ``exp_avg`` for that coordinate
        intact means the very next step pushes it straight back off zero, and
        the run ends dense while every projection looked like it worked. This
        masks the optimiser state alongside the weight. Plain SGD has no state
        and so cannot hit this at all.
    """
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    ref = _ref(halo_model)
    if ref is None:
        return {}
    pairs = ref.pairs()
    absd = torch.cat([(p - w0).abs().reshape(-1) for p, w0 in pairs])
    k = max(1, int(round(keep_frac * absd.numel())))
    # Ties at the cutoff keep slightly more than k, which is the safe direction:
    # it never prunes below the requested budget.
    cutoff = torch.topk(absd, k, largest=True, sorted=True).values[-1]
    for p, w0 in pairs:
        d = p - w0
        mask = (d.abs() >= cutoff).to(d.dtype)
        p.copy_(w0 + d * mask)
        if optimizer is not None:
            state = optimizer.state.get(p)
            if state:
                for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                    buf = state.get(key)
                    if torch.is_tensor(buf):
                        buf.mul_(mask)
    return delta_stats(halo_model)


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
    elif variant in SPARSE_VARIANTS:
        # Same weights as last_mlp. What differs is the CONSTRAINT on how they
        # may move, which lives in the training loop (a proximal step or a
        # periodic projection) and needs the pre-fine-tuning weights to measure
        # a delta against -- hence the reference below.
        for p in halo_model.transformer.h[-1].mlp.parameters():
            p.requires_grad_(True)

    trainable = [p for p in halo_model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"adapter {variant!r} left nothing trainable")
    if variant in SPARSE_VARIANTS:
        # Snapshot AFTER the unfreeze and after the caller has loaded the
        # federated global, which is why finetune_local adapts only once the
        # warm start is in place. Snapshot a randomly initialised model instead
        # and every "delta" is measured from the wrong origin.
        halo_model.sparse_ft = _DeltaRef(trainable)
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
    ref = _ref(halo_model)
    if ref is not None:
        # A delta-sparse variant knows where the federated global is, so mu can
        # be the LITERAL proximal term ||theta - theta_global||^2 rather than
        # weight decay toward zero. Penalising the raw weights here would pull
        # the layer toward the origin, which for a trunk-initialised MLP is
        # damage, not regularisation.
        terms = [((p - w0) ** 2).sum() for p, w0 in ref.pairs()]
    else:
        terms = [(p ** 2).sum()
                 for p in halo_model.parameters() if p.requires_grad]
    if not terms:
        return torch.zeros((), device=next(halo_model.parameters()).device)
    return torch.stack(terms).sum()
