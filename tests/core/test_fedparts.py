"""Tests for the shared/personal parameter split behind FedRep and LG-FedAvg."""

import os
import sys
import unittest

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..",
                    "examples", "fedpyhealth"))

from utils.fedparts import (  # noqa: E402
    describe, merge_state, personal_keys, split_state,
)


def _state():
    """A state dict shaped like HALO's, with both halves and a LayerNorm."""
    return {
        "halo_model.transformer.wte.weight": torch.ones(4, 2),
        "halo_model.transformer.h.0.ln_1.weight": torch.ones(2),
        "halo_model.transformer.h.0.attn.c_attn.weight": torch.ones(2, 6),
        "halo_model.transformer.ln_f.weight": torch.ones(2),
        "halo_model.ehr_head.auto1.weight": torch.ones(3, 3),
        "halo_model.ehr_head.auto2.bias": torch.ones(3),
    }


class TestFedParts(unittest.TestCase):
    def test_head_and_body_are_complementary(self):
        """A key in neither set is never averaged AND never kept -- it would
        silently vanish from training."""
        s = _state()
        head, body = personal_keys(s, "head"), personal_keys(s, "body")
        self.assertEqual(head & body, set())
        self.assertEqual(head | body, set(s))

    def test_a_mode_matching_nothing_raises(self):
        """If the state-dict naming drifts, the run must fail rather than
        quietly continue as plain FedAvg."""
        with self.assertRaises(ValueError):
            personal_keys({"unrelated.weight": torch.ones(1)}, "head")

    def test_empty_personal_gives_back_shared(self):
        """Round 0: nobody has a stored slice yet, so every client has to start
        from the identical broadcast weights."""
        s = _state()
        self.assertEqual(set(merge_state(s, {})), set(s))
        shared, personal = split_state(s, personal_keys(s, "head"))
        self.assertEqual(set(merge_state(shared, personal)), set(s))


if __name__ == "__main__":
    unittest.main()
