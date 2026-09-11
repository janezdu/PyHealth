"""Tests for the torch-free cohort helpers Test 1 and Test 2 share."""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..",
                    "examples", "fedpyhealth"))

from utils.cohort import fold_support, sample_eval_codes  # noqa: E402


class TestFoldSupport(unittest.TestCase):
    """``fold_support`` counts PATIENTS, which is what the bands assume."""

    def setUp(self):
        # h1/p1 charts "A" three times across two visits -- one patient, not
        # three positives. p2 shares "A"; "C" is unique to the second hospital.
        self.traj = {
            "h1": {"p1": [["A", "A", "B"], ["A"]],
                   "p2": [["A"]]},
            "h2": {"p3": [["B", "C"]]},
        }

    def test_repeats_within_a_patient_count_once(self):
        self.assertEqual(fold_support(self.traj)["A"], 2)

    def test_counts_pool_across_hospitals(self):
        self.assertEqual(fold_support(self.traj)["B"], 2)

    def test_a_code_in_one_hospital_only(self):
        self.assertEqual(fold_support(self.traj)["C"], 1)

    def test_no_code_is_invented(self):
        self.assertEqual(set(fold_support(self.traj)), {"A", "B", "C"})

    def test_empty_fold_is_empty_not_an_error(self):
        self.assertEqual(fold_support({}), {})

    def test_a_patient_with_no_visits_contributes_nothing(self):
        self.assertEqual(fold_support({"h1": {"p1": []}}), {})


class TestSampleEvalCodes(unittest.TestCase):
    """The draw is the rejection sample's second half; it must be stable."""

    def test_same_seed_same_draw(self):
        pool = [f"c{i}" for i in range(50)]
        self.assertEqual(sample_eval_codes(pool, 10, 0),
                         sample_eval_codes(pool, 10, 0))

    def test_different_seeds_differ(self):
        pool = [f"c{i}" for i in range(50)]
        self.assertNotEqual(sample_eval_codes(pool, 10, 0),
                            sample_eval_codes(pool, 10, 1))

    def test_input_order_does_not_change_the_draw(self):
        pool = [f"c{i}" for i in range(50)]
        self.assertEqual(sample_eval_codes(pool, 10, 0),
                         sample_eval_codes(list(reversed(pool)), 10, 0))

    def test_draw_is_a_subset_without_repeats(self):
        pool = [f"c{i}" for i in range(50)]
        drawn = sample_eval_codes(pool, 10, 3)
        self.assertEqual(len(drawn), 10)
        self.assertEqual(len(set(drawn)), 10)
        self.assertTrue(set(drawn) <= set(pool))


if __name__ == "__main__":
    unittest.main()
