import unittest

from acoustitrace.aggregate import ScoreRecord, bootstrap_cell, matched_valid


class AggregateTests(unittest.TestCase):
    def test_invalid_outputs_are_excluded_but_counted_in_coverage(self):
        rows = [
            ScoreRecord("a", "M", "t2av", "range", True, 80.0),
            ScoreRecord("b", "M", "t2av", "range", False, None),
            ScoreRecord("c", "M", "t2av", "range", True, 100.0),
        ]
        summary = bootstrap_cell(rows, n_resamples=100, seed=1)
        self.assertEqual(summary["n_total"], 3)
        self.assertEqual(summary["n_valid"], 2)
        self.assertAlmostEqual(summary["mean"], 90.0)

    def test_matched_valid_intersects_prompt_ids(self):
        rows = [
            ScoreRecord("a", "M1", "t2av", "range", True, 80.0),
            ScoreRecord("a", "M2", "t2av", "range", True, 70.0),
            ScoreRecord("b", "M1", "t2av", "range", True, 90.0),
            ScoreRecord("b", "M2", "t2av", "range", False, None),
        ]
        result = matched_valid(rows, ["M1", "M2"])
        self.assertEqual([row.sample_id for row in result["M1"]], ["a"])
        self.assertEqual([row.sample_id for row in result["M2"]], ["a"])


if __name__ == "__main__":
    unittest.main()

