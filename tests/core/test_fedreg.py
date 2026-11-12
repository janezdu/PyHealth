"""Tests for the federated regularizers behind FedProx, Ditto and PerAda."""

import importlib.util
import os
import unittest

import torch
import torch.nn as nn

# Loaded straight from the file rather than as pyhealth.models.generators.fedreg.
# That import path runs pyhealth/models/__init__.py, which pulls in all ~45
# models and their dependencies (transformers, torch_geometric, rdkit) -- minutes
# of imports for a module that needs only torch. fedreg.py has no pyhealth
# imports of its own, so loading it standalone is exact, not a stub.
_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "pyhealth",
                     "models", "generators", "fedreg.py")
_spec = importlib.util.spec_from_file_location("fedreg", _PATH)
fedreg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fedreg)

ProxRef = fedreg.ProxRef
binary_kl = fedreg.binary_kl
clear_prox_ref = fedreg.clear_prox_ref
get_prox_ref = fedreg.get_prox_ref
prox_penalty = fedreg.prox_penalty
set_prox_ref = fedreg.set_prox_ref


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


class TestProxRef(unittest.TestCase):
    def setUp(self):
        self.net = _net()
        self.ref = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

    def test_state_dict_is_unchanged_by_attaching_a_reference(self):
        """The property the whole design exists for: a reference registered as a
        buffer would be averaged by FedAvg and would break loading every
        existing checkpoint."""
        before = set(self.net.state_dict())
        set_prox_ref(self.net, self.ref)
        self.assertEqual(before, set(self.net.state_dict()))
        self.assertEqual([n for n, _ in self.net.named_buffers()], [])

    def test_zero_at_the_reference_and_positive_after_moving(self):
        set_prox_ref(self.net, self.ref)
        self.assertAlmostEqual(float(prox_penalty(self.net)), 0.0, places=6)
        with torch.no_grad():
            self.net[0].weight.add_(0.1)
        self.assertGreater(float(prox_penalty(self.net)), 0.0)

    def test_unset_gives_a_differentiable_zero_not_a_crash(self):
        """train_model calls this unconditionally once prox_mu > 0; an unset
        reference must not explode mid-batch."""
        self.assertAlmostEqual(float(prox_penalty(_net())), 0.0)

    def test_missing_reference_weights_raise(self):
        """A proximal term silently covering only half the model is not the
        method the paper specifies."""
        partial = {k: v for k, v in self.ref.items() if not k.startswith("0.")}
        with self.assertRaises(KeyError):
            ProxRef(self.net, partial)


class TestBinaryKL(unittest.TestCase):
    def test_zero_when_matched_positive_when_not(self):
        p = torch.rand(4, 7).clamp(0.05, 0.95)
        self.assertAlmostEqual(float(binary_kl(p, p)), 0.0, places=6)
        self.assertGreater(
            float(binary_kl(torch.full((4, 7), 0.3), torch.full((4, 7), 0.6))), 0.0)

    def test_finite_at_the_zero_one_ends(self):
        """Sigmoid outputs reach the ends; an unclamped log would NaN the loss."""
        self.assertTrue(torch.isfinite(
            binary_kl(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))))

    def test_shape_mismatch_raises(self):
        """Otherwise it broadcasts into a plausible-looking number."""
        with self.assertRaises(ValueError):
            binary_kl(torch.rand(2, 3), torch.rand(2, 4))


if __name__ == "__main__":
    unittest.main()
