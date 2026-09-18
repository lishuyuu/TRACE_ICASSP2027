import unittest
import numpy as np
from trace_tts.expresso_transform import endpoint_prominence, RelationMeasurementError
from trace_tts.metrics import rank_correct

class ExpressoEndpointTests(unittest.TestCase):

    def test_endpoint_word_order_uses_frozen_calibration(self):
        raw = np.array([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        mask = np.ones_like(raw, dtype=bool)
        scores = endpoint_prominence(raw, mask, mask, np.array([2.0, 1.0]), np.array([1.0, 0.0]))
        np.testing.assert_allclose(scores, [0.5, -0.5, 0.0])
        self.assertTrue(rank_correct(scores.tolist(), 0))
        self.assertFalse(rank_correct(scores.tolist(), 1))

    def test_identical_endpoint_reuse_does_not_remove_word_structure(self):
        raw = np.array([[4.0], [1.0]])
        mask = np.ones_like(raw, dtype=bool)
        first = endpoint_prominence(raw, mask, mask, np.ones(1), np.ones(1))
        second = endpoint_prominence(raw.copy(), mask, mask, np.ones(1), np.ones(1))
        np.testing.assert_array_equal(first, second)
        self.assertGreater(first[0], first[1])
        self.assertNotEqual(rank_correct(first.tolist(), 0), rank_correct(second.tolist(), 1))

    def test_missing_supported_value_is_not_imputed(self):
        raw = np.array([[np.nan], [1.0]])
        with self.assertRaises(RelationMeasurementError):
            endpoint_prominence(raw, np.isfinite(raw), np.ones_like(raw, dtype=bool), np.ones(1), np.ones(1))

    def test_unsupported_values_do_not_affect_prominence(self):
        raw = np.array([[3.0, np.nan], [1.0, np.nan]])
        support = np.isfinite(raw)
        scores = endpoint_prominence(raw, support, support, np.ones(2), np.array([1.0, 0.0]))
        np.testing.assert_allclose(scores, [1.0, -1.0])

    def test_direction_and_scales_are_required_valid_inputs(self):
        raw = np.array([[3.0], [1.0]])
        mask = np.ones_like(raw, dtype=bool)
        for scales, direction in (([0.0], [1.0]), ([1.0], [2.0]), ([float('nan')], [1.0])):
            with self.subTest(scales=scales, direction=direction), self.assertRaises(RelationMeasurementError):
                endpoint_prominence(raw, mask, mask, scales, direction)
if __name__ == '__main__':
    unittest.main()
