"""``train.py --no-eval``: train and write synthetic data, skip scoring."""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..",
                    "examples", "fedpyhealth"))

from train import build_config  # noqa: E402


class TestNoEvalFlag(unittest.TestCase):
    def test_scoring_runs_by_default(self):
        self.assertTrue(build_config([])["run_eval"])

    def test_no_eval_disables_scoring(self):
        self.assertFalse(build_config(["--no-eval"])["run_eval"])

    def test_no_eval_keeps_run_identity(self):
        # Skipping scoring must not move the save dir or change what a later
        # eval.py re-score computes.
        base, off = build_config([]), build_config(["--no-eval"])
        self.assertEqual(base["run_name"], off["run_name"])
        self.assertEqual(base["metrics"], off["metrics"])


if __name__ == "__main__":
    unittest.main()
