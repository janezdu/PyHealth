"""Fed-SGM aggregation and the DP noise boundary."""

import math
import os
import sys
import unittest

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "examples",
                    "fedpyhealth"))

from utils.dpaccount import sgm_epsilon, sgm_sigma_for  # noqa: E402
from utils.dpfed import dp_aggregate, sgm_aggregate  # noqa: E402


class DPKeysTest(unittest.TestCase):
    def setUp(self):
        self.ref = {"w": torch.zeros(40), "mask": torch.ones(5, 5)}
        self.states = [{"w": torch.full((40,), 0.01 * (i + 1)),
                        "mask": torch.ones(5, 5)} for i in range(4)]

    def test_gaussian_leaves_float_buffer_alone(self):
        out, _ = dp_aggregate(self.states, self.ref, 1.0, 2.0, keys={"w"})
        self.assertTrue(torch.equal(out["mask"], self.ref["mask"]))
        self.assertGreater(out["w"].std().item(), 0.0)

    def test_sgm_leaves_float_buffer_alone(self):
        out, _ = sgm_aggregate(self.states, self.ref, 1.0, 0.5, 200, {"w"}, 0)
        self.assertTrue(torch.equal(out["mask"], self.ref["mask"]))

    def test_sgm_recovers_mean_at_large_b_without_noise(self):
        states = [{"w": torch.arange(40.0) * 1e-3 * (i + 1),
                   "mask": torch.ones(5, 5)} for i in range(4)]
        want = sum(s["w"] for s in states) / 4
        out, _ = sgm_aggregate(states, self.ref, 10.0, 0.0, 40000, {"w"}, 1,
                               chunk=512)
        rel = (out["w"] - want).norm() / want.norm()
        self.assertLess(rel.item(), 0.1)

    def test_sgm_chunking_does_not_change_the_result(self):
        a, _ = sgm_aggregate(self.states, self.ref, 1.0, 0.0, 256, {"w"}, 3,
                             chunk=256)
        b, _ = sgm_aggregate(self.states, self.ref, 1.0, 0.0, 256, {"w"}, 3,
                             chunk=64)
        self.assertTrue(torch.allclose(a["w"], b["w"], atol=1e-5))


class AccountantTest(unittest.TestCase):
    def test_matches_closed_form(self):
        per = 10 / (1000 * 0.5 ** 4)
        log_d = math.log(1e5)
        s = math.sqrt(1 + log_d / per)
        got, _ = sgm_epsilon(0.5, 1000, 10)
        self.assertAlmostEqual(got, 2 * per * (s + 1), delta=0.02)

    def test_sigma_for_inverts_epsilon(self):
        sigma = sgm_sigma_for(9.0, 32768, 10, clip=0.5)
        self.assertAlmostEqual(sgm_epsilon(sigma, 32768, 10, 0.5)[0], 9.0,
                               delta=0.01)


if __name__ == "__main__":
    unittest.main()
