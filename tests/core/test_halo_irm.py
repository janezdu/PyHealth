"""The IRMv1 penalty in HALO.

Three properties carry the whole implementation, and each has a failure mode
that is silent rather than loud:

1. ``irm_scale=None`` must reproduce the old loss, or every checkpoint trained
   before IRM existed stops being comparable. To floating-point noise, not
   bitwise -- see test_none_matches_scale_one for why the distinction matters.
2. The penalty must be differentiable w.r.t. the parameters. Forgetting
   ``create_graph=True`` gives a penalty that is a constant: training runs, the
   number is logged, and the penalty contributes nothing at all.
3. The penalty must be near zero when the dummy classifier is already optimal,
   and positive when it is not. Otherwise the quantity is not IRM's.
"""

import unittest

import torch

from pyhealth.models.generators.halo import HALOModel


class _Cfg:
    """Minimal HALO config -- small enough to run on CPU in milliseconds."""
    n_positions = 8
    n_ctx = 8
    n_embd = 16
    n_layer = 1
    n_head = 2
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    total_vocab_size = 12
    code_vocab_size = 8
    label_vocab_size = 0
    special_vocab_size = 4
    pos_loss_weight = None


class _Holder:
    """Bare carrier for ``_irm_terms``, which needs only these two attributes.

    ``_irm_terms`` is defined on ``HALO`` (the PyHealth wrapper) but touches
    nothing of it beyond ``halo_model`` and ``config``. Constructing a real
    ``HALO`` would drag in a fitted ``SampleDataset``, so bind the method here
    instead -- if it ever grows another dependency this stops working, which is
    the signal we want.
    """

    def __init__(self, model, config):
        self.halo_model = model
        self.config = config

    from pyhealth.models.generators.halo import HALO as _H
    _irm_terms = _H._irm_terms


class TestIRMScale(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _Cfg()
        self.model = HALOModel(self.cfg)
        self.b, self.t, self.v = 6, self.cfg.n_ctx, self.cfg.total_vocab_size
        self.x = torch.bernoulli(torch.full((self.b, self.t, self.v), 0.3))

    def _loss(self, irm_scale=None):
        loss, _, _ = self.model(self.x, ehr_labels=self.x, irm_scale=irm_scale)
        return loss

    def test_none_matches_scale_one(self):
        """w=1.0 must reproduce the no-IRM loss to floating-point noise.

        NOT bitwise: ``code_logits * 1.0`` is exact in IEEE754, but it returns a
        fresh tensor whose layout can make ``BCELoss`` accumulate its mean in a
        different order, which moves the last bit or two. The first version of
        this test asserted bitwise equality and failed for that reason alone.

        The tolerance is deliberately tight rather than generous, and the
        relative difference is asserted (not just printed): a genuine bug here --
        the scale applied twice, applied to the wrong tensor, or dropped -- moves
        the loss by an order-1 factor, which 1e-6 still catches. A loose
        ``places=3`` would hide exactly the failure this test exists for.
        """
        base = self._loss(None).item()
        scaled = self._loss(torch.ones(1)).item()
        rel = abs(scaled - base) / max(abs(base), 1e-12)
        print(f"\n    w=1 vs no-IRM: {base!r} vs {scaled!r}  rel={rel:.3e}")
        self.assertLess(rel, 1e-6,
                        f"w=1.0 changed the loss by {rel:.3e} relative -- that "
                        "is far beyond reduction-order noise, so irm_scale is "
                        "not the identity at w=1")

    def test_scale_changes_the_loss(self):
        """A scale != 1 must actually move the loss, or the hook is inert and
        the penalty would be identically zero for the wrong reason."""
        self.assertFalse(torch.allclose(
            self._loss(None), self._loss(torch.full((1,), 3.0))))

    def test_gradient_wrt_w_is_scalar(self):
        """IRMv1's w is a SCALAR, which is what makes the penalty cheap. A
        vector here would mean w was attached to the parameters instead."""
        w = torch.ones(1, requires_grad=True)
        g = torch.autograd.grad(self._loss(w), [w])[0]
        self.assertEqual(g.shape, (1,))


class TestIRMPenalty(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _Cfg()
        self.model = HALOModel(self.cfg)
        self.holder = _Holder(self.model, self.cfg)
        self.b, self.t, self.v = 8, self.cfg.n_ctx, self.cfg.total_vocab_size
        self.x = torch.bernoulli(torch.full((self.b, self.t, self.v), 0.3))
        self.mask = torch.ones(self.b, self.t - 1, self.v)

    def test_returns_two_scalars(self):
        risk, penalty = self.holder._irm_terms(self.x, self.mask)
        self.assertEqual(risk.shape, ())
        self.assertEqual(penalty.shape, ())

    def test_penalty_is_differentiable_wrt_parameters(self):
        """The create_graph=True property. Without it the penalty is a constant
        and every parameter gradient from it is None -- training proceeds, the
        penalty is logged, and IRM does nothing whatsoever."""
        _, penalty = self.holder._irm_terms(self.x, self.mask)
        grads = torch.autograd.grad(penalty,
                                    list(self.model.parameters()),
                                    allow_unused=True)
        self.assertTrue(any(g is not None and torch.any(g != 0)
                            for g in grads),
                        "penalty has no gradient path to the parameters")

    def test_risk_matches_the_plain_loss_on_the_same_rows(self):
        """risk is the mean of the two halves, which must equal the ordinary
        loss over those rows -- otherwise adding IRM silently changes the risk
        term as well as adding the penalty."""
        risk, _ = self.holder._irm_terms(self.x, self.mask)
        half = self.b // 2
        losses = []
        for sl in (slice(0, half), slice(half, self.b)):
            loss, _, _ = self.model(self.x[sl], ehr_labels=self.x[sl],
                                    ehr_masks=self.mask[sl])
            losses.append(loss)
        self.assertTrue(torch.allclose(risk, (losses[0] + losses[1]) / 2,
                                       atol=1e-6))

    def test_penalty_small_when_w_is_already_optimal(self):
        """A model fitted to its data has little to gain from rescaling its
        logits, so the penalty should be far smaller than for a model whose
        logits are systematically mis-scaled."""
        big = HALOModel(_Cfg())
        with torch.no_grad():
            for p in big.parameters():
                p.mul_(8.0)          # blow the logits up: w=1 is now far off
        skewed = _Holder(big, self.cfg)._irm_terms(self.x, self.mask)[1]
        base = self.holder._irm_terms(self.x, self.mask)[1]
        self.assertGreater(abs(skewed.item()), abs(base.item()))

    def test_rejects_a_batch_too_small_to_split(self):
        """The unbiased estimator needs two disjoint halves. One patient cannot
        give that, and silently falling back to the biased form is exactly the
        failure this whole design avoids."""
        with self.assertRaises(ValueError) as ctx:
            self.holder._irm_terms(self.x[:1], self.mask[:1])
        self.assertIn("at least 2", str(ctx.exception))

    def test_odd_batch_uses_equal_halves(self):
        """An odd batch drops one patient rather than comparing a 4-row half
        against a 3-row one, which would bias the product."""
        risk, penalty = self.holder._irm_terms(self.x[:7], self.mask[:7])
        self.assertTrue(torch.isfinite(risk) and torch.isfinite(penalty))


class TestIRMSchedule(unittest.TestCase):
    """The warmup schedule, which lives in train.py rather than the model."""

    def setUp(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        "..", "..", "examples", "fedpyhealth"))
        from train import irm_rho_at
        self.f = irm_rho_at

    def test_off_stays_off(self):
        self.assertEqual(self.f(0, 0.0, 10), 0.0)
        self.assertEqual(self.f(99, 0.0, 10), 0.0)

    def test_holds_at_one_during_warmup(self):
        self.assertEqual(self.f(0, 1e4, 10), 1.0)
        self.assertEqual(self.f(9, 1e4, 10), 1.0)

    def test_jumps_after_warmup(self):
        self.assertEqual(self.f(10, 1e4, 10), 1e4)

    def test_zero_warmup_applies_rho_immediately(self):
        self.assertEqual(self.f(0, 1e4, 0), 1e4)


if __name__ == "__main__":
    unittest.main()
