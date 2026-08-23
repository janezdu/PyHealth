"""Per-sample loss weighting in HALO.

The weighting is opt-in and must be inert when unused: every existing caller
passes nothing, and a change in their loss would silently invalidate every
checkpoint trained before it.
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


class TestHALOSampleWeights(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = _Cfg()
        self.model = HALOModel(self.cfg)
        self.b, self.t, self.v = 4, self.cfg.n_ctx, self.cfg.total_vocab_size
        self.x = torch.bernoulli(torch.full((self.b, self.t, self.v), 0.3))

    def _loss(self, w=None, pos_loss_weight=None):
        loss, _, _ = self.model(self.x, ehr_labels=self.x,
                                pos_loss_weight=pos_loss_weight,
                                sample_weights=w)
        return loss

    def test_none_is_unchanged(self):
        """Omitting weights must reproduce the pre-feature loss exactly."""
        self.assertTrue(torch.equal(self._loss(), self._loss(None)))

    def test_uniform_weights_are_a_no_op(self):
        """All-ones weights leave the loss identical -- because 1 is the
        identity, NOT because the reduction renormalises. See
        test_constant_scaling_scales_the_loss for what a constant other than 1
        actually does."""
        base = self._loss()
        ones = self._loss(torch.ones(self.b))
        self.assertAlmostEqual(base.item(), ones.item(), places=6)

    def test_constant_scaling_scales_the_loss(self):
        """Weights do NOT self-normalise. BCELoss with reduction='mean' divides
        by the element count, not by the sum of weights, so scaling every weight
        by c scales the loss -- and therefore every gradient -- by c.

        This is the reason callers must normalise to mean 1: weights averaging
        20 would silently be a 20x learning rate, and a weighted run would then
        differ from its baseline for two reasons at once.
        """
        a = self._loss(torch.full((self.b,), 1.0))
        b = self._loss(torch.full((self.b,), 7.0))
        self.assertAlmostEqual(b.item(), 7.0 * a.item(), places=5)

    def test_all_weight_on_one_patient_isolates_that_patient(self):
        """Weight [1,0,0,0] must leave exactly patient 0's loss in the numerator
        -- divided by the full batch's element count, per the reduction above."""
        weighted = self._loss(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        solo_in = self.x[:1]
        solo, _, _ = self.model(solo_in, ehr_labels=solo_in)
        self.assertAlmostEqual(weighted.item(), solo.item() / self.b, places=5)

    def test_composes_with_pos_loss_weight(self):
        """sample_weights and pos_loss_weight are independent knobs: one says
        which patients matter, the other which codes. Using both must not error
        and must differ from using either alone."""
        both = self._loss(torch.tensor([3.0, 1.0, 1.0, 1.0]), pos_loss_weight=2.0)
        only_pos = self._loss(None, pos_loss_weight=2.0)
        self.assertFalse(torch.isnan(both))
        self.assertNotAlmostEqual(both.item(), only_pos.item(), places=6)

    def test_gradient_flows(self):
        loss = self._loss(torch.tensor([2.0, 1.0, 0.5, 1.0]))
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads and any(g.abs().sum() > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
