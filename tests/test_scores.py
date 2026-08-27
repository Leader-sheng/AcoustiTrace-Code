import math
import unittest

from acoustitrace.scores import (
    causality_violation,
    impact_decay,
    log_attack_time,
    motion_loudness,
    native_receiver_score,
    range_attenuation,
    range_reference_db,
    rt60_consistency,
)


class ScoreTests(unittest.TestCase):
    def test_motion_loudness_uses_no_margin_ordering(self):
        result = motion_loudness([(3.0, 2.0), (1.0, 1.0), (0.0, -1.0)])
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.score, 200.0 / 3.0)

    def test_lat_identity_is_full_score(self):
        self.assertAlmostEqual(log_attack_time(0.02, 0.02).score, 100.0)

    def test_impact_decay_has_no_hard_r2_gate(self):
        result = impact_decay(0.2, 0.0, 10.0)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.score, 20.0)

    def test_rt60_tolerance_and_zero_boundaries(self):
        self.assertAlmostEqual(rt60_consistency(1.5, 1.0).score, 100.0)
        self.assertAlmostEqual(rt60_consistency(3.0, 1.0).score, 0.0)

    def test_causality_uses_one_millisecond_threshold(self):
        result = causality_violation([-0.002, -0.001, 0.0])
        self.assertAlmostEqual(result.score, 200.0 / 3.0)

    def test_range_reference(self):
        values = range_reference_db([1.0, 2.0])
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], -20.0 * math.log10(2.0))

    def test_range_and_receiver_normalization(self):
        self.assertAlmostEqual(range_attenuation([1.0, 0.5]).score, 75.0)
        self.assertAlmostEqual(
            native_receiver_score(0.81, evaluator="approach_gain").score, 81.0
        )

    def test_invalid_evidence_has_no_zero_score(self):
        result = rt60_consistency(float("nan"), 1.0)
        self.assertFalse(result.valid)
        self.assertIsNone(result.score)


if __name__ == "__main__":
    unittest.main()

