"""Parameter-efficient adapters for HALO.

Four properties carry these, and three of them fail silently if broken:

1. An adapter starts as an exact no-op. LoRA's ``B`` is zero-initialised, so the
   adapted model must reproduce the trunk before any optimiser step -- that is
   what makes "adapter = 0" mean "be the federated global", which the FedProx
   term relies on.
2. The trunk is actually frozen. If ``requires_grad`` leaks back on, the run is
   a full fine-tune wearing an adapter's name and its parameter count is a lie.
3. ``lora_head`` preserves within-visit causality. A naive LoRA writes into the
   masked-out upper triangle and lets the head see codes it is about to predict.
   Training loss IMPROVES when this breaks, which is the worst possible tell.
4. The adapter can actually learn -- gradients reach it and it is far smaller
   than the trunk.

The delta-sparse variants add a fifth, which fails silently in BOTH directions:
sparsity is invisible in the training loss, so a run that never sparsified and a
run at 7% density sit at indistinguishable losses. See TestSparseDelta.
"""

import unittest

import torch

from pyhealth.models.generators.adapters import (
    SPARSE_VARIANTS,
    VARIANTS,
    LoRAMaskedLinear,
    adapter_l2,
    apply_adapter,
    delta_stats,
    hard_threshold_,
    prox_l1_,
)
from pyhealth.models.generators.halo import HALOModel


class _Cfg:
    """Minimal HALO config -- milliseconds on CPU."""
    n_positions = 8
    n_ctx = 8
    n_embd = 16
    n_layer = 2
    n_head = 2
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    total_vocab_size = 12
    code_vocab_size = 8
    label_vocab_size = 0
    special_vocab_size = 4
    pos_loss_weight = None


def _model():
    torch.manual_seed(0)
    return HALOModel(_Cfg())


def _batch(cfg=_Cfg()):
    torch.manual_seed(1)
    return torch.bernoulli(torch.full((4, cfg.n_ctx, cfg.total_vocab_size), 0.3))


class TestAdapterIsInitiallyIdentity(unittest.TestCase):
    """Property 1: adapting must not change the model until it is trained."""

    def test_lora_variants_start_as_no_ops(self):
        x = _batch()
        for variant in ("lora_attn", "lora_head"):
            with self.subTest(variant=variant):
                m = _model()
                before = m(x, ehr_labels=x)[0].item()
                apply_adapter(m, variant, rank=4)
                after = m(x, ehr_labels=x)[0].item()
                rel = abs(after - before) / max(abs(before), 1e-12)
                self.assertLess(
                    rel, 1e-6,
                    f"{variant} changed the loss by {rel:.3e} before training; "
                    "B must be zero-initialised or 'adapter = 0' no longer "
                    "means 'be the trunk'")

    def test_last_mlp_is_also_initially_identity(self):
        """It trains existing weights rather than adding any, so it is trivially
        a no-op at step 0 -- asserted so the three variants stay comparable."""
        x = _batch()
        m = _model()
        before = m(x, ehr_labels=x)[0].item()
        apply_adapter(m, "last_mlp", rank=4)
        self.assertAlmostEqual(before, m(x, ehr_labels=x)[0].item(), places=9)


class TestTrunkIsFrozen(unittest.TestCase):
    """Property 2: the parameter counts must be true."""

    def test_each_variant_trains_far_less_than_the_trunk(self):
        for variant in ("lora_attn", "last_mlp", "lora_head"):
            with self.subTest(variant=variant):
                m = _model()
                trainable = apply_adapter(m, variant, rank=4)
                n_tr = sum(p.numel() for p in trainable)
                n_all = sum(p.numel() for p in m.parameters())
                self.assertLess(n_tr, n_all * 0.5,
                                f"{variant} trains {n_tr}/{n_all} params, which "
                                "is not parameter-efficient")
                frozen = [p for p in m.parameters() if not p.requires_grad]
                self.assertTrue(frozen, f"{variant} froze nothing")

    def test_none_trains_everything(self):
        m = _model()
        trainable = apply_adapter(m, "none", rank=4)
        self.assertEqual(sum(p.numel() for p in trainable),
                         sum(p.numel() for p in m.parameters()))

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            apply_adapter(_model(), "lora_everything", rank=4)

    def test_variants_tuple_matches_what_is_implemented(self):
        for v in VARIANTS:
            apply_adapter(_model(), v, rank=4)


class TestCausalMaskSurvives(unittest.TestCase):
    """Property 3: the one that fails silently and looks like an improvement."""

    def test_upper_triangle_of_the_effective_weight_stays_zero(self):
        m = _model()
        apply_adapter(m, "lora_head", rank=4)
        for name in ("auto1", "auto2"):
            layer = getattr(m.ehr_head, name)
            self.assertIsInstance(layer, LoRAMaskedLinear)
            # Train the adapter away from zero first -- at B = 0 the delta is
            # zero and the mask would hold even in a broken implementation, so
            # testing the untrained state proves nothing.
            with torch.no_grad():
                layer.B.normal_(0, 0.5)
                layer.A.normal_(0, 0.5)
            w = layer.effective_weight()
            masked_out = w[layer.base.mask == 0]
            self.assertEqual(
                masked_out.abs().max().item(), 0.0,
                f"{name}: the LoRA delta leaked into the masked-out upper "
                "triangle, so a code can see codes that follow it in the same "
                "visit. Mask the SUM (W + BA), not just W.")

    def test_a_naive_unmasked_lora_would_fail_this(self):
        """Guards the test itself: if masking were removed the assertion above
        must actually catch it, otherwise it is checking nothing."""
        m = _model()
        apply_adapter(m, "lora_head", rank=4)
        layer = m.ehr_head.auto1
        with torch.no_grad():
            layer.B.normal_(0, 0.5)
            layer.A.normal_(0, 0.5)
        naive = layer.base.mask * layer.base.weight + (layer.B @ layer.A)
        self.assertGreater(naive[layer.base.mask == 0].abs().max().item(), 0.0)


class TestCheckpointRoundTrips(unittest.TestCase):
    """The failure this suite originally missed.

    Training saved an ADAPTED model, whose parameters are renamed by the
    wrapping (``attn.c_attn.weight`` -> ``attn.c_attn.base.weight``, plus new
    ``A``/``B`` entries). generate.py rebuilt a PLAIN model and loaded into it,
    which fails with a wall of missing/unexpected keys -- after six runs had
    already trained. The fix is ordering: adapt, then load. These tests pin it.
    """

    def test_adapted_state_dict_will_not_load_into_a_plain_model(self):
        """Characterises the bug, so the fix cannot be quietly undone."""
        adapted = _model()
        apply_adapter(adapted, "lora_attn", rank=4)
        plain = _model()
        with self.assertRaises(RuntimeError):
            plain.load_state_dict(adapted.state_dict())

    def test_it_loads_once_the_adapter_is_applied_first(self):
        for variant in ("lora_attn", "last_mlp", "lora_head"):
            with self.subTest(variant=variant):
                src = _model()
                apply_adapter(src, variant, rank=4)
                with torch.no_grad():          # move it off the identity
                    for q in src.parameters():
                        if q.requires_grad:
                            q.add_(0.05)
                dst = _model()
                apply_adapter(dst, variant, rank=4)
                dst.load_state_dict(src.state_dict())
                x = _batch()
                self.assertAlmostEqual(src(x, ehr_labels=x)[0].item(),
                                       dst(x, ehr_labels=x)[0].item(), places=6)


class TestAdapterLearns(unittest.TestCase):
    """Property 4: gradients reach the adapter, and the proximal term works."""

    def test_gradients_reach_the_adapter(self):
        x = _batch()
        for variant in ("lora_attn", "last_mlp", "lora_head"):
            with self.subTest(variant=variant):
                m = _model()
                trainable = apply_adapter(m, variant, rank=4)
                m(x, ehr_labels=x)[0].backward()
                self.assertTrue(
                    any(p.grad is not None and torch.any(p.grad != 0)
                        for p in trainable),
                    f"{variant}: no gradient reached any trainable parameter")

    def test_frozen_trunk_gets_no_gradient(self):
        x = _batch()
        m = _model()
        apply_adapter(m, "lora_attn", rank=4)
        m(x, ehr_labels=x)[0].backward()
        leaked = [p for p in m.parameters()
                  if not p.requires_grad and p.grad is not None]
        self.assertFalse(leaked, "a frozen parameter accumulated a gradient")

    def test_adapter_l2_grows_as_the_adapter_moves(self):
        """mu must actually restrain personalisation.

        Note what this does NOT assert: that the penalty is zero at init. It is
        not. ``A`` is random-initialised while ``B`` is zero, so the DELTA is
        zero but ``||A||^2 + ||B||^2`` is positive. That is expected -- the
        penalty is the variational form of the nuclear norm of the update, not
        the Frobenius proximal term; see adapter_l2's docstring.
        """
        m = _model()
        apply_adapter(m, "lora_attn", rank=4)
        at_init = adapter_l2(m).item()
        self.assertGreater(at_init, 0.0, "A is random-initialised, so this "
                                         "should not be zero")
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(0.5)
        self.assertGreater(adapter_l2(m).item(), at_init)

    def test_the_delta_is_zero_at_init_even_though_the_penalty_is_not(self):
        """The property the FedProx reading actually rests on: the model IS the
        trunk at init, whatever the penalty says about it."""
        m = _model()
        apply_adapter(m, "lora_head", rank=4)
        layer = m.ehr_head.auto1
        delta = (layer.B @ layer.A) * layer.scale
        self.assertEqual(delta.abs().max().item(), 0.0)


def _sparse(variant="last_mlp_iht", scale=0.1):
    """An adapted model whose delta has already moved off zero."""
    m = _model()
    trainable = apply_adapter(m, variant, rank=4)
    torch.manual_seed(2)
    with torch.no_grad():
        for p in trainable:
            p.add_(torch.randn_like(p) * scale)
    return m, trainable


class TestSparseDeltaBasics(unittest.TestCase):
    """The reference is the whole design: sparsity is measured on
    ``D = W - W_global``, not on ``W``. A last-MLP's weights are the trunk's, and
    forcing THOSE to be mostly zero deletes what the trunk knows -- that is a
    pruning experiment, not a personalisation one."""

    def test_sparse_variants_start_as_no_ops(self):
        x = _batch()
        for variant in SPARSE_VARIANTS:
            with self.subTest(variant=variant):
                m = _model()
                before = m(x, ehr_labels=x)[0].item()
                apply_adapter(m, variant, rank=4)
                self.assertAlmostEqual(before, m(x, ehr_labels=x)[0].item(),
                                       places=9)

    def test_delta_is_zero_at_init(self):
        for variant in SPARSE_VARIANTS:
            with self.subTest(variant=variant):
                m = _model()
                apply_adapter(m, variant, rank=4)
                st = delta_stats(m)
                self.assertEqual(st["l1"], 0.0)
                self.assertEqual(st["nnz_frac"], 0.0)
                self.assertGreater(st["n"], 0)

    def test_plain_last_mlp_gets_no_reference(self):
        """Deliberate, and load-bearing.

        Adding buffers to plain ``last_mlp`` would change its state dict, and
        the last_mlp checkpoints already on disk -- written without them --
        would stop loading in generate.py. The sparse variants are new, so they
        can carry the reference from their first run.
        """
        m = _model()
        apply_adapter(m, "last_mlp", rank=4)
        self.assertIsNone(getattr(m, "sparse_ft", None))
        self.assertEqual(delta_stats(m), {})

    def test_reference_does_not_double_register_the_parameters(self):
        """The live weights are held as a plain list, not a ParameterList. A
        ParameterList would register them a second time, so the optimiser would
        see each weight twice and the parameter count would be a lie."""
        m = _model()
        trainable = apply_adapter(m, "last_mlp_iht", rank=4)
        ids = [id(p) for p in m.parameters()]
        self.assertEqual(len(ids), len(set(ids)))
        for p in trainable:
            self.assertEqual(sum(1 for q in m.parameters() if q is p), 1)

    def test_adapter_l2_measures_the_delta_not_the_weights(self):
        """mu on a sparse variant is a real proximal term, because the reference
        exists. Penalising the raw weights would pull a trunk-initialised MLP
        toward the origin, which is damage rather than regularisation."""
        m = _model()
        apply_adapter(m, "last_mlp_iht", rank=4)
        self.assertAlmostEqual(adapter_l2(m).item(), 0.0, places=9)
        m2, _ = _sparse()
        self.assertGreater(adapter_l2(m2).item(), 0.0)


class TestProximalL1MakesActualZeros(unittest.TestCase):
    """The failure this variant exists to avoid.

    Adding ``lam * ||D||_1`` to the loss does not sparsify: the subgradient of
    ``|x|`` is ``sign(x)``, so the update chatters in a band of width ``lr*lam``
    around zero and essentially never lands on it. The run then reports 0%
    sparsity and reads as a failed idea when it is a failed representation.
    """

    def test_soft_threshold_produces_exact_zeros(self):
        m, _ = _sparse("last_mlp_l1", scale=0.1)
        self.assertEqual(delta_stats(m)["nnz_frac"], 1.0)
        prox_l1_(m, 0.08)
        st = delta_stats(m)
        self.assertLess(st["nnz_frac"], 1.0,
                        "soft-thresholding produced no exact zeros")
        self.assertGreater(st["nnz_frac"], 0.0,
                           "threshold wiped the delta out entirely; pick a "
                           "smaller one or the test proves nothing")

    def test_a_big_enough_threshold_zeros_everything(self):
        m, _ = _sparse("last_mlp_l1", scale=0.1)
        prox_l1_(m, 10.0)
        st = delta_stats(m)
        self.assertEqual(st["nnz_frac"], 0.0)
        self.assertEqual(st["l1"], 0.0)

    def test_it_shrinks_rather_than_clipping(self):
        """Soft, not hard: a surviving coordinate must move TOWARD zero by the
        threshold. A no-op on survivors would be hard thresholding wearing the
        wrong name, and would leave the L1 norm unchanged."""
        m, _ = _sparse("last_mlp_l1", scale=0.1)
        before = delta_stats(m)
        prox_l1_(m, 0.02)
        after = delta_stats(m)
        self.assertLess(after["l1"], before["l1"])
        self.assertLess(after["linf"], before["linf"])

    def test_a_zero_threshold_is_a_no_op(self):
        m, _ = _sparse("last_mlp_l1", scale=0.1)
        before = delta_stats(m)["l1"]
        prox_l1_(m, 0.0)
        self.assertAlmostEqual(delta_stats(m)["l1"], before, places=6)


class TestIterativeHardThresholding(unittest.TestCase):

    def test_it_hits_the_requested_density(self):
        for keep in (0.5, 0.1):
            with self.subTest(keep=keep):
                m, _ = _sparse("last_mlp_iht")
                st = hard_threshold_(m, keep)
                self.assertAlmostEqual(st["nnz_frac"], keep, delta=0.02)

    def test_it_keeps_the_LARGEST_coordinates(self):
        m, trainable = _sparse("last_mlp_iht")
        ref = m.sparse_ft
        before = torch.cat([(p - w0).reshape(-1).clone()
                            for p, w0 in ref.pairs()])
        hard_threshold_(m, 0.25)
        after = torch.cat([(p - w0).reshape(-1) for p, w0 in ref.pairs()])
        kept = after != 0
        self.assertGreater(before[kept].abs().min().item(),
                           before[~kept].abs().max().item() - 1e-9,
                           "a dropped coordinate was larger than a kept one")
        # Survivors are untouched -- hard, not soft.
        self.assertTrue(torch.allclose(after[kept], before[kept]))

    def test_it_zeros_the_optimizer_state_for_pruned_coordinates(self):
        """The silent one. Zeroing a weight but leaving Adam's exp_avg intact
        means the next step pushes it straight back off zero, so the run ends
        dense while every projection looked like it worked."""
        m = _model()
        trainable = apply_adapter(m, "last_mlp_iht", rank=4)
        opt = torch.optim.Adam(trainable, lr=0.05)
        x = _batch()
        for _ in range(3):                     # build real Adam state
            opt.zero_grad()
            m(x, ehr_labels=x)[0].backward()
            opt.step()
        self.assertGreater(delta_stats(m)["nnz_frac"], 0.5)
        hard_threshold_(m, 0.25, opt)
        ref = m.sparse_ft
        for p, w0 in ref.pairs():
            pruned = (p - w0) == 0
            state = opt.state[p]
            self.assertEqual(state["exp_avg"][pruned].abs().max().item(), 0.0)
            self.assertEqual(state["exp_avg_sq"][pruned].abs().max().item(), 0.0)

    def test_without_the_optimizer_momentum_re_inflates_the_pruned_weights(self):
        """Guards the test above: if masking the state were dropped, the
        assertion must actually catch something."""
        m = _model()
        trainable = apply_adapter(m, "last_mlp_iht", rank=4)
        opt = torch.optim.Adam(trainable, lr=0.05)
        x = _batch()
        for _ in range(3):
            opt.zero_grad()
            m(x, ehr_labels=x)[0].backward()
            opt.step()
        hard_threshold_(m, 0.25)               # NOTE: optimizer not passed
        opt.zero_grad()
        m(x, ehr_labels=x)[0].backward()
        opt.step()
        self.assertGreater(
            delta_stats(m)["nnz_frac"], 0.9,
            "the un-masked optimizer state was expected to re-inflate almost "
            "every pruned coordinate; if it does not, the masking in "
            "hard_threshold_ is untested")

    def test_pruned_coordinates_can_re_grow_between_projections(self):
        """What makes it ITERATIVE hard thresholding rather than one-shot
        pruning: training between projections is dense, so a coordinate zeroed
        at one projection is free to come back and survive the next."""
        m = _model()
        trainable = apply_adapter(m, "last_mlp_iht", rank=4)
        opt = torch.optim.Adam(trainable, lr=0.05)
        x = _batch()
        for _ in range(2):
            opt.zero_grad()
            m(x, ehr_labels=x)[0].backward()
            opt.step()
        hard_threshold_(m, 0.25, opt)
        self.assertAlmostEqual(delta_stats(m)["nnz_frac"], 0.25, delta=0.02)
        for _ in range(2):
            opt.zero_grad()
            m(x, ehr_labels=x)[0].backward()
            opt.step()
        self.assertGreater(delta_stats(m)["nnz_frac"], 0.25)

    def test_bad_keep_frac_raises(self):
        m, _ = _sparse("last_mlp_iht")
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(keep=bad), self.assertRaises(ValueError):
                hard_threshold_(m, bad)


class TestSparseCheckpointRoundTrips(unittest.TestCase):
    """The reference buffers change the state dict, so generate.py's
    adapt-then-load ordering has to cover them too."""

    def test_it_loads_once_the_adapter_is_applied_first(self):
        for variant in SPARSE_VARIANTS:
            with self.subTest(variant=variant):
                src, trainable = _sparse(variant)
                dst = _model()
                apply_adapter(dst, variant, rank=4)
                dst.load_state_dict(src.state_dict())
                x = _batch()
                self.assertAlmostEqual(src(x, ehr_labels=x)[0].item(),
                                       dst(x, ehr_labels=x)[0].item(), places=6)

    def test_the_reference_survives_the_round_trip(self):
        """Not just the weights: without W_global the loaded model cannot say
        what its own delta or sparsity was, and a finished run stops being
        auditable from its checkpoint alone."""
        src, _ = _sparse("last_mlp_iht")
        hard_threshold_(src, 0.25)
        dst = _model()
        apply_adapter(dst, "last_mlp_iht", rank=4)
        dst.load_state_dict(src.state_dict())
        self.assertAlmostEqual(delta_stats(dst)["nnz_frac"],
                               delta_stats(src)["nnz_frac"], places=9)
        self.assertAlmostEqual(delta_stats(dst)["l1"],
                               delta_stats(src)["l1"], places=5)


if __name__ == "__main__":
    unittest.main()
