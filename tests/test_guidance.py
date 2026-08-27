import unittest

from acoustitrace.guidance import decoded_mel_pairwise_loss, deterministic_pairs


class GuidanceTests(unittest.TestCase):
    def test_inverse_distance_envelope_has_near_zero_loss(self):
        result = decoded_mel_pairwise_loss(
            [1.0, 0.5, 0.25], [1.0, 2.0, 4.0], epsilon=1e-12
        )
        self.assertLess(result["loss"], 1e-16)
        self.assertEqual(result["n_pairs"], 3)

    def test_pair_limit_is_deterministic(self):
        first = deterministic_pairs(200, max_pairs=17)
        second = deterministic_pairs(200, max_pairs=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 17)


if __name__ == "__main__":
    unittest.main()

