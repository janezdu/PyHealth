"""Tests for the intra-round self-training split point.

``_selftrain_split_epoch`` decides how many real-only epochs run before the
generator is sampled. Its edge cases are the ones that fail SILENTLY -- a split
past the last epoch generates data and then never trains on it, which looks like
a valid run and measures nothing.
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..",
                    "examples", "fedpyhealth"))

from train import _selftrain_split_epoch  # noqa: E402


class TestSelftrainSplitEpoch(unittest.TestCase):

    def test_zero_means_no_split(self):
        """0 is the original behaviour and must resolve to no real-only block."""
        self.assertEqual(_selftrain_split_epoch(0.0, 10), 0)
        self.assertEqual(_selftrain_split_epoch(None, 10), 0)

    def test_the_eight_two_split(self):
        """The motivating case: 8 real epochs, generate, 2 mixed."""
        self.assertEqual(_selftrain_split_epoch(0.8, 10), 8)

    def test_fraction_travels_across_local_epochs(self):
        """A fraction means the same thing at a different --local-epochs.

        This is why the flag is a fraction and not an epoch index: 0.8 stays
        "four fifths of the way in" whether local_epochs is 5 or 20.
        """
        self.assertEqual(_selftrain_split_epoch(0.8, 5), 4)
        self.assertEqual(_selftrain_split_epoch(0.8, 20), 16)

    def test_never_splits_past_the_last_epoch(self):
        """at=1.0 would generate and then never train on the result."""
        self.assertEqual(_selftrain_split_epoch(1.0, 10), 9)
        self.assertEqual(_selftrain_split_epoch(2.0, 10), 9)

    def test_negative_is_treated_as_off(self):
        self.assertEqual(_selftrain_split_epoch(-0.5, 10), 0)

    def test_single_local_epoch_cannot_split(self):
        """With one local epoch there is no boundary to split on."""
        self.assertEqual(_selftrain_split_epoch(0.8, 1), 0)


class TestRunNaming(unittest.TestCase):
    """Two runs differing in split or source must not share a save_dir."""

    def setUp(self):
        from train import make_run_name
        self.build = make_run_name
        self.base = {
            "regime": "fedavg", "local_epochs": 10, "n_rounds": 10,
            "cohort_cache": "/x/hilo8_random", "metrics": "privacy",
            "selftrain_frac": 0.5,
        }

    def test_default_split_and_source_keep_the_existing_name(self):
        """at=0 / source=global is the old behaviour; its name must not move,
        or every existing st run becomes unfindable."""
        self.assertNotIn("@", self.build(self.base))
        self.assertNotIn("src", self.build(self.base))

    def test_split_and_source_are_in_the_name(self):
        split = self.build({**self.base, "selftrain_at": 0.8})
        local = self.build({**self.base, "selftrain_at": 0.8,
                            "selftrain_source": "local"})
        self.assertIn("@8", split)
        self.assertIn("srclocal", local)
        self.assertNotEqual(split, local)


if __name__ == "__main__":
    unittest.main()
