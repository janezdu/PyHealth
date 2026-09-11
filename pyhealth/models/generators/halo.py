"""HALO: Hierarchical Autoregressive Language mOdel for synthetic EHR generation.

This is a faithful port of the reference implementation
(https://github.com/Brandon-Theodorou/HALO_Inpatient) wrapped as a PyHealth
``BaseModel`` so it consumes the standard
``dataset -> set_task -> SampleDataset -> model`` pipeline.

HALO is a two-level model:

* a GPT-2-style **coarse** transformer operates over visit-level multi-hot
  vectors, and
* a **fine** autoregressive head predicts the (multi-label) set of codes within
  each visit.

The transformer/head classes below (``LayerNorm``, ``Conv1D``, ``Attention``,
``MLP``, ``Block``, ``CoarseTransformerModel``, ``AutoregressiveLinear``,
``FineAutoregressiveHead``, ``HALOModel``) are ported verbatim from the
reference ``model.py``. The only behavioural change is that PyHealth's HALO is
**unconditional** (``label_vocab_size = 0``): it generates visit-code sequences
without conditioning on CCS labels.
"""

import copy
import math
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel


# ----------------------------------------------------------------------------
# Configuration (plain class, not a dataclass; mirrors reference config.py)
# ----------------------------------------------------------------------------
class HALOConfig:
    """Hyperparameter container for the HALO transformer.

    Kept as a plain class with explicit ``__init__`` assignments (matching the
    reference ``config.py``) so the low-level modules can read attributes such
    as ``config.n_embd``.
    """

    def __init__(
        self,
        total_vocab_size: int,
        code_vocab_size: int,
        label_vocab_size: int = 0,
        special_vocab_size: int = 3,
        n_positions: int = 56,
        n_ctx: int = 48,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        layer_norm_epsilon: float = 1e-5,
        dropout: float = 0.0,
        latent_dim: int = 0,
        initializer_range: float = 0.02,
        batch_size: int = 48,
        epoch: int = 50,
        pos_loss_weight: Optional[float] = None,
        lr: float = 1e-4,
    ) -> None:
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        # 0.0 reproduces the model as it was before dropout existed, exactly.
        # Anything above 0 is a DIFFERENT model: it changes training for every
        # regime, so every result already on the board was produced at 0.0 and
        # is not comparable to a dropout run.
        self.dropout = dropout
        # 0 disables the latent entirely and the model is unchanged. Above 0,
        # z ~ N(0, I) of this width is projected into the embedding at POSITION
        # 1 -- the slot HALO reserves for a conditioning label, which this port
        # leaves empty. Not an arbitrary position: the fine head pairs
        # history[t] with input_visits[t+1], so the hidden state at position 1
        # is exactly what predicts visit 1, and 79% of this cohort's patients
        # have only visit 1.
        self.latent_dim = latent_dim
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.epoch = epoch
        self.pos_loss_weight = pos_loss_weight
        self.lr = lr


# ----------------------------------------------------------------------------
# Transformer building blocks (ported verbatim from reference model.py)
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside sqrt)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=n_embd (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        # getattr, not config.dropout: a HALOConfig pickled before this field
        # existed still loads, and the tiny test configs do not define it.
        p_drop = getattr(config, "dropout", 0.0)
        self.attn_dropout = nn.Dropout(p_drop)
        self.resid_dropout = nn.Dropout(p_drop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=4 * n_embd
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, x):
        # tanh-approximate GELU, matching the reference HALO implementation.
        h = F.gelu(self.c_fc(x), approximate="tanh")
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class CoarseTransformerModel(nn.Module):
    def __init__(self, config):
        super(CoarseTransformerModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(
            config.total_vocab_size, config.n_embd, bias=False
        )
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.embed_dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        d = getattr(config, "latent_dim", 0)
        self.z_proj = nn.Linear(d, config.n_embd) if d > 0 else None

    def forward(self, input_visits, position_ids=None, past=None, z=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_visits.size(1) + past_length,
                dtype=torch.long,
                device=input_visits.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(
                input_visits.size(0), input_visits.size(1)
            )

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        if self.z_proj is not None and z is not None:
            # OUT OF PLACE. inputs_embeds is an autograd intermediate;
            # slice-assigning into it makes the graph depend on a mutated
            # buffer, which either errors or silently gives wrong gradients.
            # A one-hot over positions, broadcast, adds z at position 1 only.
            slot = torch.zeros(inputs_embeds.shape[1], 1,
                               device=inputs_embeds.device,
                               dtype=inputs_embeds.dtype)
            slot[1] = 1.0
            inputs_embeds = inputs_embeds + slot.unsqueeze(0) * \
                self.z_proj(z).unsqueeze(1)
        hidden_states = self.embed_dropout(inputs_embeds + position_embeds)
        for block, layer_past in zip(self.h, past):
            hidden_states, _ = block(hidden_states, layer_past)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class AutoregressiveLinear(nn.Linear):
    """Same as Linear except it has a configurable mask on the weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer(
            "mask", torch.tril(torch.ones(in_features, out_features)).int()
        )

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class FineAutoregressiveHead(nn.Module):
    def __init__(self, config):
        super(FineAutoregressiveHead, self).__init__()
        self.auto1 = AutoregressiveLinear(
            config.n_embd + config.total_vocab_size,
            config.n_embd + config.total_vocab_size,
        )
        self.auto2 = AutoregressiveLinear(
            config.n_embd + config.total_vocab_size,
            config.n_embd + config.total_vocab_size,
        )
        self.n_embd = config.n_embd
        self.tot_vocab = config.total_vocab_size

    def forward(self, history, input_visits):
        history = history[:, :-1, :]
        input_visits = input_visits[:, 1:, :]
        code_logits = self.auto2(
            torch.relu(self.auto1(torch.cat((history, input_visits), dim=2)))
        )[:, :, self.n_embd - 1:-1]
        return code_logits

    def sample(self, history, input_visits):
        history = history[:, :-1, :]
        input_visits = input_visits[:, 1:, :]
        currVisit = torch.cat((history, input_visits), dim=2)[:, -1, :].unsqueeze(1)
        code_logits = self.auto2(torch.relu(self.auto1(currVisit)))[
            :, :, self.n_embd - 1:-1
        ]
        return code_logits


class HALOModel(nn.Module):
    """Low-level HALO transformer + autoregressive head (ported verbatim)."""

    def __init__(self, config):
        super(HALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)

    def forward(
        self,
        input_visits,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        past=None,
        pos_loss_weight=None,
        sample_weights=None,
        reduce: bool = True,
        irm_scale=None,
        z=None,
    ):
        """Args:
            irm_scale: Optional scalar tensor multiplying the output logits --
                the dummy classifier ``w`` of IRMv1. Left at ``None`` (i.e. 1.0)
                this changes nothing. Pass a ``requires_grad=True`` scalar and
                differentiate the returned loss with respect to it to get
                ``grad_{w|w=1} R``, whose square is the IRM penalty. It scales
                the LOGITS, not the parameters: IRM's ``w`` is a classifier
                stacked on the representation, and scaling ``theta`` instead
                would be a different (and much less meaningful) quantity.
            reduce: ``True`` (default) returns the scalar batch loss, exactly as
                before. ``False`` returns a ``(batch,)`` vector of per-patient
                losses -- needed by best-of-K objectives, which have to pick a
                winner per patient rather than per batch. Averaging the vector
                reproduces the scalar.
            sample_weights: Optional ``(batch,)`` tensor scaling each patient's
                contribution to the loss. Multiplies whatever ``pos_loss_weight``
                already does, so the two compose: ``pos_loss_weight`` says which
                CODES matter more, ``sample_weights`` says which PATIENTS do.
                NOT self-normalising: ``BCELoss`` with ``reduction="mean"``
                divides by the ELEMENT COUNT, not by the sum of weights, so
                scaling every weight by c scales the loss and its gradient by c
                too. Normalise the weights to mean 1 unless you intend to change
                the effective learning rate along with the weighting -- otherwise
                a weighted run is not comparable to an unweighted one and the
                difference cannot be attributed to the weighting.
        """
        hidden_states = self.transformer(input_visits, position_ids, past, z=z)
        code_logits = self.ehr_head(hidden_states, input_visits)
        if irm_scale is not None:
            code_logits = code_logits * irm_scale
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            loss_weights = None
            if pos_loss_weight is not None:
                loss_weights = torch.ones(
                    code_probs.shape, device=code_probs.device
                )
                loss_weights = loss_weights + (pos_loss_weight - 1) * shift_labels
            if sample_weights is not None:
                # (batch,) -> (batch, 1, 1) so it scales every position and code
                # of that patient equally. Built here rather than by the caller
                # so the caller never has to know the label tensor's shape.
                w = sample_weights.to(code_probs.device).view(-1, 1, 1)
                loss_weights = w.expand_as(code_probs).clone() \
                    if loss_weights is None else loss_weights * w
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks
                if pos_loss_weight is not None:
                    loss_weights = loss_weights * ehr_masks

            if reduce:
                bce = nn.BCELoss(weight=loss_weights)
                loss = bce(code_probs, shift_labels)
            else:
                # Per-PATIENT loss, for objectives that need to choose among
                # candidates one patient at a time. Averaging the elementwise
                # losses per row reproduces the reduction="mean" scalar exactly
                # when every element is kept, so the two paths cannot drift.
                bce = nn.BCELoss(weight=loss_weights, reduction="none")
                loss = bce(code_probs, shift_labels).flatten(1).mean(1)
            return loss, code_probs, shift_labels

        return code_probs

    def sample(self, input_visits, random=True, temperature: float = 1.0,
               z=None):
        """Extend ``input_visits`` by one visit, one code at a time.

        Args:
            input_visits: The sequence so far; the LAST position is filled in.
            random: ``True`` draws each code from its Bernoulli, ``False``
                rounds at 0.5. Rounding cannot emit rare codes at all -- a
                0.3%-prevalence code never reaches p = 0.5 -- so it silently
                deletes the tail this cohort exists to measure.
            temperature: Divides the logits before the sigmoid. Because the
                codes are conditionally independent Bernoullis, the expected
                number emitted per visit is ``sum_c p_c`` -- so temperature is a
                direct dial on codes/visit at GENERATION time, with no
                retraining. ``tau < 1`` sharpens (probabilities move toward 0
                and 1, and the sum falls); ``tau > 1`` flattens and the sum
                rises. ``1.0`` is the trained model, unchanged.

                Use it to ask how much of a prevalence gap is CALIBRATION
                rather than representation. Tuning it to hit a prevalence
                target and reporting that as a result would be fitting the
                sampler to the metric.
        """
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits, z=z)
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, input_visits)
            if temperature != 1.0:
                next_logits = next_logits / temperature
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)

            remaining_visit = visit[:, 0, i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:, -1, i + first_nonzero] = visit[:, 0, i + first_nonzero]
            i = i + first_nonzero + 1

        return input_visits


# ----------------------------------------------------------------------------
# PyHealth BaseModel wrapper
# ----------------------------------------------------------------------------
class HALO(BaseModel):
    """HALO synthetic-EHR generator, wrapped as a PyHealth ``BaseModel``.

    Trains a GPT-2-style transformer on patient visit-code sequences and
    generates synthetic patients by autoregressive sampling. Generation is
    **unconditional** (no label conditioning).

    The model infers its code vocabulary from the fitted ``SampleDataset``:
    ``code_vocab_size = dataset.input_processors["visits"].vocab_size()``
    (the ``NestedSequenceProcessor`` vocab, which already reserves index 0 for
    ``<pad>`` and index 1 for ``<unk>``). Three special tokens are appended for
    start-of-sequence, end-of-sequence, and pad-visit.

    Args:
        dataset: A fitted ``SampleDataset`` whose ``input_schema`` contains
            ``{"visits": NestedSequenceProcessor}`` and whose ``output_schema``
            is empty.
        embed_dim: Transformer embedding dimension (``n_embd``). Default: 768.
        n_heads: Number of attention heads. Must divide ``embed_dim``.
            Default: 12.
        n_layers: Number of transformer layers. Default: 12.
        n_ctx: Maximum number of visit positions (context length). Default: 48.
        batch_size: Training batch size. Default: 48.
        epochs: Number of training epochs. Default: 50.
        pos_loss_weight: Positive-class weight for the BCE loss. ``None`` means
            no weighting. Default: None.
        lr: Learning rate for the Adam optimizer. Default: 1e-4.
        save_dir: Directory for checkpoints written by ``train_model``.
            Default: ``"./save/"``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p1", "visits": [["A", "B"], ["C"]]},
        ...     {"patient_id": "p2", "visits": [["A"], ["B", "C"]]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"visits": "nested_sequence"},
        ...     output_schema={},
        ... )
        >>> model = HALO(dataset, embed_dim=16, n_heads=2, n_layers=2, n_ctx=8)
        >>> isinstance(model, HALO)
        True
    """

    def __init__(
        self,
        dataset,
        embed_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        n_ctx: int = 48,
        batch_size: int = 48,
        epochs: int = 50,
        pos_loss_weight: Optional[float] = None,
        lr: float = 1e-4,
        dropout: float = 0.0,
        latent_dim: int = 0,
        save_dir: str = "./save/",
    ) -> None:
        super(HALO, self).__init__(dataset)

        if "visits" not in dataset.input_processors:
            raise ValueError(
                "HALO expects an input feature named 'visits' backed by a "
                "NestedSequenceProcessor."
            )

        self.save_dir = save_dir
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr

        # Code vocab from the NestedSequenceProcessor (includes <pad>, <unk>).
        self.visits_processor = dataset.input_processors["visits"]
        code_vocab_size = self.visits_processor.vocab_size()
        label_vocab_size = 0  # unconditional generation -- no output labels
        # +3 special tokens: start-of-sequence, end-of-sequence, pad-visit.
        total_vocab_size = code_vocab_size + label_vocab_size + 3

        self.config = HALOConfig(
            total_vocab_size=total_vocab_size,
            code_vocab_size=code_vocab_size,
            label_vocab_size=label_vocab_size,
            special_vocab_size=3,
            n_positions=n_ctx + 8,  # position table needs a little slack
            n_ctx=n_ctx,
            n_embd=embed_dim,
            n_layer=n_layers,
            n_head=n_heads,
            batch_size=batch_size,
            epoch=epochs,
            pos_loss_weight=pos_loss_weight,
            lr=lr,
            dropout=dropout,
            latent_dim=latent_dim,
        )

        # Registered as a sub-module so .parameters()/.to() work.
        self.halo_model = HALOModel(self.config)

    # ------------------------------------------------------------------
    # Multi-hot encoding helper
    # ------------------------------------------------------------------
    def _encode_visits(self, visits: torch.Tensor):
        """Place per-visit multi-hot vectors into HALO's context window.

        Takes what :class:`~pyhealth.processors.NestedMultiHotProcessor` emits
        -- one multi-hot row per visit -- and lays it out the way the
        transformer expects: position 0 is the start token, visits occupy
        positions 2+, the end token sits just past the last real visit, and the
        pad token fills the rest.

        Fully vectorised, deliberately. This ran as a triple-nested Python loop
        over (patient, visit, code slot) and cost ~108 ms per patient on an
        A100 -- 99.8% of a training step, against ~0.03 s for the transformer's
        own forward and backward. Two things made it that expensive: an
        ``.item()`` per patient, each forcing a CUDA sync, and a separate
        single-element kernel launch per code. Neither survives here.

        Args:
            visits: FloatTensor ``(batch, max_visits, code_vocab_size)``,
                1.0 where a code is present in that visit. A visit with no
                codes is an all-zero row.

        Returns:
            batch_ehr: FloatTensor ``(batch, n_ctx, total_vocab_size)``.
            batch_mask: FloatTensor ``(batch, n_ctx - 1, 1)``, shifted to align
                with the autoregressive prediction targets.
        """
        cfg = self.config
        visits = visits.to(self.device)
        batch_size, max_visits = visits.shape[0], visits.shape[1]

        batch_ehr = torch.zeros(
            batch_size, cfg.n_ctx, cfg.total_vocab_size, device=self.device
        )
        batch_mask = torch.zeros(batch_size, cfg.n_ctx, 1, device=self.device)

        start_idx = cfg.code_vocab_size + cfg.label_vocab_size
        end_idx = start_idx + 1
        pad_idx = start_idx + 2

        # Real visits per patient, for the whole batch at once. An all-zero row
        # is an empty visit, exactly as a row of <pad> indices was before.
        n_visits = (visits.sum(dim=-1) > 0).sum(dim=1)
        n_visits = n_visits.clamp(max=cfg.n_ctx - 2)             # (batch,)

        # Two positions are reserved (start, end), so only this many visits fit.
        keep = min(max_visits, cfg.n_ctx - 2)
        if keep > 0:
            pos = torch.arange(keep, device=self.device)          # (keep,)
            valid = (pos.unsqueeze(0) < n_visits.unsqueeze(1))    # (batch, keep)
            # Codes occupy the first code_vocab_size columns; the three special
            # tokens live above them and are set separately below.
            batch_ehr[:, 2:2 + keep, :cfg.code_vocab_size] = (
                visits[:, :keep, :] * valid.unsqueeze(-1)
            )
            batch_mask[:, 2:2 + keep, 0] = valid.to(batch_mask.dtype)

        batch_ehr[:, 0, start_idx] = 1                            # start token
        rows = torch.arange(batch_size, device=self.device)
        batch_ehr[rows, n_visits + 1, end_idx] = 1                # end token

        # Everything past the end token is padding.
        ctx = torch.arange(cfg.n_ctx, device=self.device)
        is_pad = ctx.unsqueeze(0) >= (n_visits + 2).unsqueeze(1)  # (batch, n_ctx)
        batch_ehr[:, :, pad_idx] = is_pad.to(batch_ehr.dtype)

        batch_mask = batch_mask[:, 1:, :]  # shift to align with shifted targets
        return batch_ehr, batch_mask

    # ------------------------------------------------------------------
    # forward -- required by BaseModel
    # ------------------------------------------------------------------
    def forward(self, visits: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            visits: LongTensor ``(batch, max_visits, max_codes_per_visit)`` from
                the ``NestedSequenceProcessor``.
            **kwargs: Any other batch keys are ignored.

        Returns:
            Dict with ``loss`` (scalar BCE) and ``y_prob`` (code probabilities,
            shape ``(batch, n_ctx - 1, total_vocab_size)``).
        """
        visits = visits.to(self.device)
        batch_ehr, batch_mask = self._encode_visits(visits)

        loss, code_probs, _ = self.halo_model(
            batch_ehr,
            position_ids=None,
            ehr_labels=batch_ehr,
            ehr_masks=batch_mask,
            pos_loss_weight=self.config.pos_loss_weight,
        )
        return {"loss": loss, "y_prob": code_probs}

    # ------------------------------------------------------------------
    # Custom training loop
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device=None) -> torch.device:
        """Resolve a user-supplied device, defaulting to CUDA when available.

        Args:
            device: ``None``, a device string (e.g. ``"cuda"``, ``"cuda:1"``,
                ``"cpu"``), or a ``torch.device``. When ``None``, CUDA is used
                if available, otherwise CPU.
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _irm_terms(self, batch_ehr, batch_mask, sample_weights=None,
                   xm_k: int = 1):
        """Risk and unbiased IRMv1 penalty for one batch.

        IRMv1 penalises how far the dummy classifier ``w=1`` is from optimal for
        this environment: ``P = (d/dw R(w * logits))^2`` at ``w = 1``. Zero
        penalty in every environment at once means one classifier is
        simultaneously optimal everywhere, which is the invariance IRM wants.

        **Why the batch is split in half.** The naive estimator squares a
        minibatch gradient, and ``E[g^2] != (E[g])^2`` -- the gap is the
        gradient's variance, so the naive penalty is biased UPWARD by exactly
        the minibatch noise, and the model can minimise it by making its
        gradients noisy rather than by becoming invariant. Two disjoint halves
        give independent estimates ``g1``, ``g2`` whose product is unbiased for
        ``(E[g])^2``. This is what the IRM reference implementation does, and
        skipping it is the usual reason an IRM run does nothing.

        The estimator stays unbiased but gets noisier as the batch shrinks, and
        it is a PRODUCT of two estimates, so it can come out negative. That is
        expected, not a bug -- it means the two halves disagreed on the sign,
        i.e. there is no reliable gradient to penalise. Do not clamp it; the
        noise averages out across batches and clamping would reintroduce the
        upward bias the split exists to remove.

        Args:
            batch_ehr: Encoded multi-hot visits, ``(batch, n_ctx, vocab)``.
            batch_mask: Visit mask from ``_encode_visits``.
            sample_weights: Optional ``(batch,)`` per-patient weights.

        Returns:
            ``(risk, penalty)`` -- both scalar tensors carrying grad. ``risk``
            is the mean of the two halves' losses, which equals the full-batch
            loss over the samples used (both halves are padded to the same
            shape, so ``BCELoss``'s element-count mean matches).

        Raises:
            ValueError: If the batch holds fewer than 2 patients, which cannot
                be split.
        """
        n = int(batch_ehr.shape[0])
        if n < 2:
            raise ValueError(
                f"IRM needs at least 2 patients per batch to split, got {n}. "
                "Lower --batch-size so the smallest hospital still fills a "
                "batch, or drop --irm for this run."
            )
        half = n // 2   # an odd batch drops one patient; halves must match
        risks, grads = [], []
        for sl in (slice(0, half), slice(half, 2 * half)):
            w = torch.ones(1, device=batch_ehr.device, requires_grad=True)
            sw = None if sample_weights is None else sample_weights[sl]
            if xm_k > 1:
                # Best-of-K INSIDE the IRM risk. The penalty is then taken on
                # the winning candidate only -- autograd routes the gradient
                # through whichever branch the min selected, which is exactly
                # "the penalty of the candidate we actually trained on".
                #
                # The alternative, averaging the penalty over all K, is a
                # variance reduction on a noisy estimator but costs K
                # second-order backwards instead of one. Not done here.
                #
                # Caveat worth keeping in view: w scales the logits, so in
                # principle a different candidate could win at a different w.
                # The gradient is evaluated at w = 1, where the selection is
                # fixed, so this is the correct local derivative -- but R(w)
                # has kinks and the penalty is a local quantity.
                cands = torch.stack([
                    self.halo_model(
                        batch_ehr[sl], position_ids=None,
                        ehr_labels=batch_ehr[sl], ehr_masks=batch_mask[sl],
                        pos_loss_weight=self.config.pos_loss_weight,
                        sample_weights=sw, irm_scale=w, reduce=False,
                        z=self._draw_z(batch_ehr[sl].shape[0]),
                    )[0] for _ in range(xm_k)
                ])
                loss = cands.min(dim=0).values.mean()
            else:
                loss, _, _ = self.halo_model(
                    batch_ehr[sl],
                    position_ids=None,
                    ehr_labels=batch_ehr[sl],
                    ehr_masks=batch_mask[sl],
                    pos_loss_weight=self.config.pos_loss_weight,
                    sample_weights=sw,
                    irm_scale=w,
                    z=self._draw_z(batch_ehr[sl].shape[0]),
                )
            # create_graph=True keeps the penalty differentiable w.r.t. theta --
            # without it the penalty is a constant and contributes no gradient,
            # which fails silently. It makes the backward second-order and is
            # the main cost of turning IRM on.
            grads.append(torch.autograd.grad(loss, [w], create_graph=True)[0])
            risks.append(loss)
        risk = (risks[0] + risks[1]) / 2
        penalty = (grads[0] * grads[1]).sum()
        return risk, penalty


    def _draw_z(self, batch_size: int):
        """One z ~ N(0, I) per PATIENT, or None when the latent is disabled.

        Per patient, not per position: the point of the latent is a coherent
        alternative phenotype for a whole person, so it has to be constant
        across that patient's visits. A fresh draw per position would be noise
        again, which is what the dropout probe already showed does nothing.

        Never inferred from the data -- there is no encoder and no KL. The model
        only ever sees z drawn from the prior, so training and generation match
        by construction and there is no posterior gap to police. What makes the
        model USE z rather than average over it is best-of-K: if z spreads the
        predictions coherently, one of the K candidates lands close, and that is
        the only branch that gets a gradient.
        """
        d = getattr(self.config, "latent_dim", 0)
        if not d:
            return None
        return torch.randn(batch_size, d, device=self.device)

    def train_model(self, train_dataset, val_dataset=None, device=None,
                    on_epoch_end: Callable[[int, float], None] = None,
                    sample_weight_fn: Callable[[dict], "torch.Tensor"] = None,
                    irm_rho=0.0, adapter_mu: float = 0.0,
                    adapter_l1: float = 0.0,
                    adapter_sparsity: float = 0.0,
                    adapter_iht_every: int = 1,
                    adapter_optim: str = "adam",
                    adapter_lr: float = 0.0,
                    xm_k: int = 1) -> None:
        """Train the HALO model with a custom loop.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``. Uses the standard ``get_dataloader`` (which pads
        the variable visit dimension for us), an Adam optimizer, and BCE loss.
        When ``val_dataset`` is given, validation loss is computed after each
        epoch and the best checkpoint is saved to ``self.save_dir``.

        Args:
            train_dataset: ``SampleDataset`` for training.
            val_dataset: Optional ``SampleDataset`` for validation.
            irm_rho: IRMv1 penalty weight -- a float, or a callable
                ``epoch -> float`` when it varies across the epochs of ONE call.
                Both forms exist because the regimes call this differently:
                FedAvg calls it once per client per round and resolves the round
                schedule itself (a float), while centralized/local call it once
                for the whole run and need the warmup resolved per epoch here
                (a callable). ``0.0`` (default) trains the plain
                risk and costs nothing. Above zero each step optimises
                ``risk + irm_rho * penalty`` (see :meth:`_irm_terms`), which
                needs a second-order backward and roughly doubles step cost.
                Resolve any warmup SCHEDULE in the caller and pass the value for
                this round: keeping the schedule out of here means the model
                holds no training-progress state, which matters because FedAvg
                builds a fresh optimizer and calls this once per client per
                round, so anything counted here would reset 400 times.
                After each epoch ``self.last_irm`` holds that epoch's mean
                ``{"risk", "penalty", "rho"}`` -- read it from ``on_epoch_end``
                to log the two terms as separate curves.
            adapter_mu: FedProx-style L2 pulling the fine-tuned weights back
                toward the federated global. ``0.0`` (default) is off.
            adapter_l1: L1 penalty on the delta ``W - W_global``, applied as a
                PROXIMAL soft-threshold of size ``lr * adapter_l1`` after each
                optimiser step -- not as a term added to the loss. The
                distinction is the whole point: a subgradient L1 shrinks
                coordinates but never lands on zero, so it regularises without
                sparsifying and the run reports 0% sparsity. Requires a delta
                -sparse adapter variant; ``0.0`` (default) is off.
            adapter_sparsity: Iterative hard thresholding. Every
                ``adapter_iht_every`` epochs the delta is projected onto its
                largest ``adapter_sparsity`` fraction of coordinates, ranked
                globally across the fine-tuned tensors. Training between
                projections is dense, so a pruned coordinate can re-grow -- that
                is what makes it iterative rather than one-shot pruning. Also
                projected on the final epoch, so the saved model is actually
                sparse. ``0.0`` (default) is off.
            adapter_iht_every: Epochs between hard-threshold projections.
            adapter_optim: ``"adam"`` (default, matching every other regime) or
                ``"sgd"``. SGD is worth having for the sparse variants
                specifically: ``lr * lam`` is the EXACT proximal operator for
                SGD (this is ISTA) but only a heuristic under Adam, and plain
                SGD carries no optimiser state, so hard thresholding cannot be
                undone by momentum re-inflating a coordinate that was just
                zeroed. It is a correctness cross-check, not a better optimiser.
            adapter_lr: Learning rate override for the fine-tuning optimiser.
                ``0.0`` (default) uses the model's own ``lr``. Effectively
                required with ``adapter_optim="sgd"``: the configured ``1e-4``
                is an ADAM learning rate, and SGD at that value barely moves, so
                an untuned SGD arm underperforms for reasons unrelated to
                sparsity.
            xm_k: Best-of-K exploration ("Forward XM"). ``1`` (default) trains
                the ordinary loss and costs nothing. Above 1, each batch is
                pushed through the model ``K`` times -- the candidates differ
                because dropout draws a fresh mask each pass -- and only the
                LOWEST loss per patient is trained on.

                The motivation is specific rather than fashionable: BCE under
                maximum likelihood rewards hedging, spreading a little
                probability over every plausible code, and the centralized arm
                on this cohort emits 25.2 codes/visit against a real 11.5.
                Best-of-K lets the model commit each candidate to a coherent
                phenotype instead, because it only has to be right ONCE. The
                same argument applies with more force to rare codes: under MLE a
                0.3%-prevalence code is smoothed toward zero, while here the
                model is rewarded for SOMETIMES emitting it.

                The minimum is taken PER PATIENT, not per batch. A per-batch
                minimum would select one dropout mask for 256 patients at once,
                which is nearly no selection at all -- that is why
                :meth:`HALOModel.forward` grew a ``reduce=False`` path.

                After each epoch ``self.last_xm`` holds ``{"spread", "k",
                "win_entropy"}``; read them, because this fails silently. If
                dropout does not separate the candidates their losses are equal,
                ``min`` picks arbitrarily, and the run quietly trains on 1/K of
                the gradient while looking perfectly healthy.
            sample_weight_fn: Optional ``batch -> (batch_size,)`` tensor giving
                each patient's weight in the loss. Called once per batch on the
                raw collated batch, so it can key off any field the dataset
                carries. ``None`` (or returning ``None``) trains unweighted,
                which is the default and leaves existing callers unchanged.
                Validation is deliberately NOT weighted: val loss is a
                model-selection signal and has to stay comparable to runs that
                weighted differently, or across a change in the weighting rule.
            device: Device to train on, e.g. ``"cuda"``, ``"cuda:1"``, or
                ``"cpu"``. If ``None`` (default), uses CUDA when available and
                falls back to CPU.
            on_epoch_end: Optional callback ``(epoch, mean_train_loss)`` invoked
                after each epoch with that epoch's mean batch loss. Lets callers
                (e.g. the federated example) log per-epoch loss curves without
                this method needing to know about TensorBoard. Returning
                ``False`` stops training after that epoch, letting a caller
                early-stop on its own criterion -- a validation loss it
                computes itself, a time budget -- without this loop having to
                know what the criterion is. Any other return value continues.
                Default ``None`` is a no-op, so existing callers are
                unaffected.
        """
        # xm_k > 1 with an IRM penalty IS supported: _irm_terms takes the
        # min over K inside each half-batch and the penalty follows the winning
        # candidate. An earlier version refused the combination outright, on the
        # grounds that it was "not a defined objective" -- that was too strong.
        # It is well defined; what is uncertain is whether IRM's question ("is
        # one predictor optimal everywhere?") still means the same thing when
        # the risk is already a per-patient selection over K candidates. Read
        # the penalty curve with that in mind.
        device = self._resolve_device(device)
        self.to(device)
        print(f"Training on: {device}")

        os.makedirs(self.save_dir, exist_ok=True)
        # Only what is unfrozen. Identical to optimising every parameter when
        # nothing is frozen, but an adapter run freezes the trunk and handing
        # Adam frozen tensors would build moment buffers for 6M parameters that
        # never move -- and would silently train them if anything later flipped
        # requires_grad back on.
        trainable = [p for p in self.halo_model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "no trainable parameters: every weight is frozen. An adapter "
                "variant must leave something with requires_grad=True.")
        lr = float(adapter_lr) if adapter_lr else self._lr
        if adapter_optim == "adam":
            optimizer = torch.optim.Adam(trainable, lr=lr)
        elif adapter_optim == "sgd":
            # Momentum deliberately left at zero. It is what makes SGD stateless
            # here, so a hard-threshold projection cannot be undone by a
            # momentum buffer pushing the pruned coordinate straight back off
            # zero -- the failure hard_threshold_ has to mask around for Adam.
            optimizer = torch.optim.SGD(trainable, lr=lr)
        else:
            raise ValueError(
                f"adapter_optim must be 'adam' or 'sgd', got {adapter_optim!r}")

        checkpoint_path = os.path.join(self.save_dir, "halo_model")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.halo_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        train_loader = get_dataloader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        global_loss = 1e10
        for epoch in tqdm(range(self._epochs), desc="Epochs"):
            rho = float(irm_rho(epoch)) if callable(irm_rho) else float(irm_rho)
            self.halo_model.train()
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            epoch_loss_sum, epoch_batches = 0.0, 0
            epoch_risk_sum, epoch_penalty_sum = 0.0, 0.0
            xm_spread_sum, xm_ent_sum, xm_batches = 0.0, 0.0, 0
            for batch in batch_iter:
                visits = batch["visits"].to(self.device)
                batch_ehr, batch_mask = self._encode_visits(visits)

                # The weighting POLICY lives in the caller, not here: this loop
                # only knows that some patients may count for more than others.
                # Derived from the batch rather than carried as a dataset field,
                # so no cached dataset has to be rebuilt to try a new rule.
                sw = None
                if sample_weight_fn is not None:
                    sw = sample_weight_fn(batch)
                    if sw is not None:
                        sw = sw.to(self.device)

                optimizer.zero_grad()
                if rho > 0.0:
                    risk, penalty = self._irm_terms(batch_ehr, batch_mask, sw,
                                                    xm_k=xm_k)
                    loss = risk + rho * penalty
                    # Rescale once rho is large. IRMv1's rho jumps by orders of
                    # magnitude after warmup, and without this the whole loss --
                    # risk included -- is scaled up with it, which is an
                    # unannounced learning-rate increase. Dividing keeps the
                    # gradient magnitude comparable to the rho=1 regime, so the
                    # only thing rho changes is the BALANCE between the terms.
                    if rho > 1.0:
                        loss = loss / rho
                    epoch_risk_sum += risk.item()
                    epoch_penalty_sum += penalty.item()
                elif xm_k > 1:
                    # All K candidates are kept in the graph rather than found
                    # under no_grad and recomputed: recomputing would draw a
                    # FRESH dropout mask, so the pass that won would not be the
                    # pass that trains. Memory is the cheaper thing to spend --
                    # measured peak here is under 1 GB of 42.
                    # A DIFFERENT z per candidate is the entire mechanism.
                    # With a latent, the K candidates are K coherent alternative
                    # phenotypes rather than K noise realisations, and best-of-K
                    # is what rewards the model for making z mean something --
                    # under an ordinary loss it would average over z and ignore
                    # it.
                    cands = torch.stack([
                        self.halo_model(
                            batch_ehr, position_ids=None, ehr_labels=batch_ehr,
                            ehr_masks=batch_mask,
                            pos_loss_weight=self.config.pos_loss_weight,
                            sample_weights=sw, reduce=False,
                            z=self._draw_z(batch_ehr.shape[0]),
                        )[0] for _ in range(xm_k)
                    ])                                   # (K, batch)
                    best = cands.min(dim=0)
                    risk = best.values.mean()
                    loss = risk
                    epoch_risk_sum += risk.item()
                    with torch.no_grad():
                        # How far apart the candidates are. Zero means dropout
                        # did not differentiate them and the min is arbitrary.
                        xm_spread_sum += (cands.max(0).values
                                          - best.values).mean().item()
                        # Which candidate wins should be near-uniform. A single
                        # index winning everything is the same degenerate case
                        # seen from the other side.
                        counts = torch.bincount(best.indices.flatten(),
                                                minlength=xm_k).float()
                        pk = counts / counts.sum()
                        xm_ent_sum += float(
                            -(pk * torch.log(pk.clamp_min(1e-12))).sum()
                            / math.log(xm_k))
                    xm_batches += 1
                else:
                    risk, _, _ = self.halo_model(
                        batch_ehr,
                        position_ids=None,
                        ehr_labels=batch_ehr,
                        ehr_masks=batch_mask,
                        pos_loss_weight=self.config.pos_loss_weight,
                        sample_weights=sw,
                        z=self._draw_z(batch_ehr.shape[0]),
                    )
                    loss = risk
                    epoch_risk_sum += risk.item()
                if adapter_mu > 0.0:
                    # FedProx, specialised to a frozen trunk: a LoRA delta of
                    # zero IS the federated global, so L2 on the trainable
                    # parameters is exactly ||theta - theta_global||^2 and mu
                    # sets how far a site may personalise away from the shared
                    # model. See adapters.adapter_l2 for why last_mlp is the
                    # exception.
                    from pyhealth.models.generators.adapters import adapter_l2
                    loss = loss + (adapter_mu / 2.0) * adapter_l2(self.halo_model)
                loss.backward()
                optimizer.step()
                if adapter_l1 > 0.0:
                    # AFTER the step, not inside the loss. Soft-thresholding is
                    # the proximal operator of the L1; adding lam*||D||_1 to the
                    # loss instead leaves the optimiser chattering in a band of
                    # width lr*lam around zero and never landing on it, so the
                    # run would regularise without ever sparsifying.
                    from pyhealth.models.generators.adapters import prox_l1_
                    prox_l1_(self.halo_model, lr * adapter_l1)
                epoch_loss_sum += loss.item()
                epoch_batches += 1
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            # The two IRM terms, kept APART rather than reported as their
            # sum. rho cannot be tuned without seeing their relative magnitude,
            # and IRM's characteristic failure -- the penalty going to zero by
            # making the model useless -- is invisible in the total. Set before
            # the callback so a logger can read them off the model.
            n_b = max(1, epoch_batches)
            self.last_irm = {"risk": epoch_risk_sum / n_b,
                             "penalty": epoch_penalty_sum / n_b,
                             "rho": rho}

            # Project BEFORE the callback, so any snapshot an early stopper
            # takes from on_epoch_end is of the projected weights. The final
            # epoch always projects, or the model that gets saved is the dense
            # one that happened to be mid-cycle when training ran out.
            if adapter_sparsity > 0.0:
                from pyhealth.models.generators.adapters import hard_threshold_
                due = (epoch + 1) % max(1, adapter_iht_every) == 0
                if due or epoch == self._epochs - 1:
                    hard_threshold_(self.halo_model, adapter_sparsity, optimizer)

            # Sparsity is the claim these variants make, and it is invisible in
            # the loss -- a fully dense run and a 7%-dense one can sit at the
            # same training loss. Recorded per epoch so ||D||_1 and the non-zero
            # fraction can be traced as their own curves, the way risk and
            # penalty are for IRM.
            from pyhealth.models.generators.adapters import delta_stats
            self.last_sparse = delta_stats(self.halo_model)

            # Best-of-K's silent failure, recorded where a logger can read it.
            self.last_xm = ({"k": xm_k,
                             "spread": xm_spread_sum / max(1, xm_batches),
                             "win_entropy": xm_ent_sum / max(1, xm_batches)}
                            if xm_k > 1 else None)
            if self.last_xm is not None:
                # To stdout, not only TensorBoard: this is the run's evidence
                # that exploration is actually happening, and it has to survive
                # a --no-tb smoke run. spread -> 0 or win_entropy -> 0 means the
                # K candidates are not distinct and the minimum is arbitrary.
                print(f"  xm K={xm_k}  spread={self.last_xm['spread']:.3e}  "
                      f"win_entropy={self.last_xm['win_entropy']:.3f}")

            if on_epoch_end is not None:
                # Returning False is a stop request -- this is what lets a
                # caller early-stop on a criterion this loop knows nothing
                # about (e.g. a validation loss it computes itself). Any other
                # return value, including the None an ordinary logging callback
                # gives, continues training.
                if on_epoch_end(epoch, epoch_loss_sum / max(1, epoch_batches)) \
                        is False:
                    break

            if val_dataset is not None:
                self.halo_model.eval()
                val_loader = get_dataloader(
                    val_dataset, batch_size=self._batch_size, shuffle=False
                )
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        visits = val_batch["visits"].to(self.device)
                        batch_ehr, batch_mask = self._encode_visits(visits)
                        val_loss, _, _ = self.halo_model(
                            batch_ehr,
                            position_ids=None,
                            ehr_labels=batch_ehr,
                            ehr_masks=batch_mask,
                            pos_loss_weight=self.config.pos_loss_weight,
                        )
                        val_losses.append(val_loss.item())

                cur_val_loss = float(np.mean(val_losses))
                print(f"Epoch {epoch} Validation Loss: {cur_val_loss:.7f}")
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                        "model": self.halo_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(state, checkpoint_path)
                    print("------------ Save best model ------------")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def generate(
        self, num_samples: int, random_sampling: bool = True, device=None,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """Generate synthetic patients using the trained HALO model.

        Autoregressive sampling: feed a start token and repeatedly call
        ``halo_model.sample`` until an end token is produced or ``n_ctx`` steps
        are reached, then decode code indices back to code strings.

        Args:
            num_samples: Number of synthetic patients to generate.
            random_sampling: If True, Bernoulli sampling (stochastic). If False,
                rounding (deterministic). Default: True.
            temperature: Logit temperature before the sigmoid; see
                :meth:`HALOModel.sample`. ``1.0`` (default) is the trained
                model. Lower emits fewer codes per visit, higher emits more.
            device: Device to generate on, e.g. ``"cuda"``, ``"cuda:1"``, or
                ``"cpu"``. If ``None`` (default), uses CUDA when available and
                falls back to CPU.

        Returns:
            List of dicts, each ``{"patient_id": "synthetic_i",
            "visits": [[code, ...], ...]}`` with decoded code strings.
        """
        device = self._resolve_device(device)
        self.to(device)

        cfg = self.config
        index_to_code = {v: k for k, v in self.visits_processor.code_vocab.items()}
        end_token_idx = cfg.code_vocab_size + cfg.label_vocab_size + 1
        start_token_idx = cfg.code_vocab_size + cfg.label_vocab_size

        self.halo_model.eval()
        synthetic_dataset: List[Dict] = []
        sample_batch_size = min(num_samples, 256)
        generated = 0
        pbar = tqdm(total=num_samples, desc="Generating patients")

        with torch.no_grad():
            while generated < num_samples:
                bs = min(sample_batch_size, num_samples - generated)
                stoken = torch.zeros(
                    cfg.total_vocab_size, device=self.device, dtype=torch.float32
                )
                stoken[start_token_idx] = 1
                prev = stoken.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
                empty = torch.zeros(
                    bs, 1, cfg.total_vocab_size,
                    device=self.device, dtype=torch.float32,
                )

                # One z per generated patient, held FIXED across that
                # patient's visits -- the latent is a property of the person,
                # not of each visit. Redrawing per step would make it noise.
                z = self._draw_z(bs)
                for _ in range(cfg.n_ctx - 1):
                    prev = self.halo_model.sample(
                        torch.cat((prev, empty), dim=1), random_sampling,
                        temperature, z=z,
                    )
                    has_end = prev[:, :, end_token_idx].sum(dim=1).bool()
                    if has_end.all():
                        break

                batch_ehrs = prev.cpu().detach().numpy()
                for i in range(bs):
                    ehr = batch_ehrs[i]  # (seq_len, total_vocab_size)
                    visits_out: List[List[str]] = []
                    # Position 0 is the start token; visits occupy positions 1+.
                    for j in range(1, len(ehr)):
                        indices = np.nonzero(ehr[j])[0]
                        visit_codes: List[str] = []
                        hit_end = False
                        for idx in indices:
                            if idx < cfg.code_vocab_size:
                                code = index_to_code.get(int(idx))
                                if code not in (None, "<pad>", "<unk>"):
                                    visit_codes.append(code)
                            elif idx == end_token_idx:
                                hit_end = True
                        if visit_codes:
                            visits_out.append(visit_codes)
                        if hit_end:
                            break

                    synthetic_dataset.append(
                        {
                            "patient_id": f"synthetic_{generated + i}",
                            "visits": visits_out,
                        }
                    )
                generated += bs
                pbar.update(bs)
            pbar.close()

        return synthetic_dataset
