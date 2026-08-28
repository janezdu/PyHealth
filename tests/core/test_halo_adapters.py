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
"""

import unittest

import torch

from pyhealth.models.generators.adapters import (
    VARIANTS,
    LoRAMaskedLinear,
    adapter_l2,
    apply_adapter,
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


if __name__ == "__main__":
    unittest.main()
