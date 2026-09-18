import unittest
import numpy as np
from trace_tts.trace_phone_clock import PhoneClockError, map_phone_positions, shared_phone_boundaries, validate_phone_clock, warp_phone_features

class PhoneClockTests(unittest.TestCase):

    def setUp(self):
        self.ids = np.array(['AA', 'B', 'AA', 'D'])
        self.words = np.array([0, 0, 1, 1], dtype=np.int64)
        self.edges = np.array([0, 1, 4, 5, 8], dtype=np.float64)

    def test_exact_noop_copy_and_native_composition(self):
        values = (np.arange(24, dtype=np.float32).reshape(8, 3) / 7).copy()
        edges = shared_phone_boundaries(self.ids, self.words, self.edges, np.array([0.5, 0.5]))
        self.assertEqual(edges.tobytes(), self.edges.tobytes())
        result, info = warp_phone_features(values, self.ids, self.words, self.edges, self.ids.copy(), self.words.copy(), edges)
        self.assertEqual(result.dtype, values.dtype)
        self.assertEqual(result.tobytes(), values.tobytes())
        self.assertFalse(np.shares_memory(result, values))
        self.assertTrue(info['identity'])

    def test_shared_clock_preserves_total_and_within_word_ratios(self):
        before = self.edges.copy()
        result = shared_phone_boundaries(self.ids, self.words, self.edges, np.array([0.75, 0.25]))
        np.testing.assert_array_equal(result, np.array([0, 1.5, 6, 6.5, 8]))
        np.testing.assert_array_equal(self.edges, before)
        self.assertEqual(result[-1], 8)
        np.testing.assert_allclose(np.diff(result)[:2] / 6, np.diff(before)[:2] / 4)
        np.testing.assert_allclose(np.diff(result)[2:] / 2, np.diff(before)[2:] / 4)
        rounded = shared_phone_boundaries(self.ids, self.words, self.edges, np.array([0.3, 0.7], dtype=np.float32))
        self.assertEqual(rounded[-1], 8)

    def test_piecewise_phone_mapping_and_boundary_identity(self):
        ids, words = (np.array(['A', 'B', 'C']), np.array([0, 0, 1]))
        source, target = (np.array([0.0, 2.0, 6.0, 8.0]), np.array([0.0, 4.0, 6.0, 12.0]))
        points = np.array([0.5, 3.5, 4.5, 5.5, 6.5, 11.5])
        actual = map_phone_positions(points, ids, words, source, ids, words, target)
        np.testing.assert_allclose(actual, [0.25, 1.75, 3.0, 5.0, 6 + 1 / 6, 8 - 1 / 6])
        np.testing.assert_array_equal(map_phone_positions(target, ids, words, source, ids, words, target), source)
        self.assertTrue(np.all(actual[:2] < 2))
        self.assertTrue(np.all((actual[2:4] >= 2) & (actual[2:4] < 6)))
        self.assertTrue(np.all(actual[4:] >= 6))

    def test_linear_feature_sampling_on_frame_centers(self):
        ids, words = (np.array(['A', 'B']), np.array([0, 1]))
        source, target = (np.array([0.0, 2.0, 4.0]), np.array([0.0, 4.0, 6.0]))
        centers = np.arange(4, dtype=np.float32) + 0.5
        values = np.stack((centers, 2 * centers + 1), axis=1)
        result, info = warp_phone_features(values, ids, words, source, ids, words, target)
        expected_positions = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5])
        clipped = np.clip(expected_positions, 0.5, 3.5)
        np.testing.assert_allclose(result, np.stack((clipped, 2 * clipped + 1), axis=1))
        np.testing.assert_array_equal(info['source_sample_positions'], expected_positions)
        self.assertEqual(result.shape, (6, 2))
        self.assertEqual(result.dtype, values.dtype)
        self.assertFalse(info['identity'])

    def test_subframe_phones_are_diagnosed_not_padded(self):
        ids, words = (np.array(['A', 'B', 'C']), np.array([0, 0, 1]))
        source, target = (np.array([0.0, 0.25, 1.75, 3.0]), np.array([0.0, 0.1, 1.0, 3.0]))
        result, info = warp_phone_features(np.arange(3, dtype=np.float32), ids, words, source, ids, words, target)
        np.testing.assert_array_equal(info['source_subframe_phone_indices'], [0])
        np.testing.assert_array_equal(info['source_zero_center_phone_indices'], [0])
        np.testing.assert_array_equal(info['target_zero_center_phone_indices'], [0])
        self.assertEqual(result.shape, (3,))
        np.testing.assert_array_equal(target, [0, 0.1, 1, 3])

    def test_phone_and_word_identity_mismatches_are_rejected(self):
        values = np.ones((8, 2), dtype=np.float32)
        for ids, words in ((np.array(['AA', 'X', 'AA', 'D']), self.words), (self.ids, np.array([0, 1, 1, 1]))):
            with self.subTest(ids=ids, words=words), self.assertRaises(PhoneClockError):
                warp_phone_features(values, self.ids, self.words, self.edges, ids, words, self.edges)

    def test_invalid_geometry_and_compositions_are_rejected(self):
        for edges in ([0, 1, 1, 5, 8], [0, 1, 0.5, 5, 8], [0, 1, 4, np.nan, 8], [1, 2, 4, 5, 8], [0, 1, 4, 5, 8.5]):
            with self.subTest(edges=edges), self.assertRaises(PhoneClockError):
                validate_phone_clock(self.ids, self.words, np.array(edges))
        for words in ([0, 0, 2, 2], [1, 1, 2, 2], [0, 1, 0, 1], [0.0, 0.0, 1.0, 1.0]):
            with self.subTest(words=words), self.assertRaises(PhoneClockError):
                validate_phone_clock(self.ids, np.array(words), self.edges)
        for pi in ([0, 1], [-0.1, 1.1], [0.5, 0.6], [np.nan, 0.5], [1]):
            with self.subTest(pi=pi), self.assertRaises(PhoneClockError):
                shared_phone_boundaries(self.ids, self.words, self.edges, np.array(pi))

    def test_invalid_features_positions_and_nonarrays_are_rejected(self):
        for values in (np.ones(7), np.full(8, np.inf), np.ones(8, dtype=np.int64)):
            with self.subTest(values=values), self.assertRaises(PhoneClockError):
                warp_phone_features(values, self.ids, self.words, self.edges, self.ids, self.words, self.edges)
        for points in (np.array([-0.1]), np.array([8.1]), np.array([np.nan])):
            with self.subTest(points=points), self.assertRaises(PhoneClockError):
                map_phone_positions(points, self.ids, self.words, self.edges, self.ids, self.words, self.edges)
        with self.assertRaises(PhoneClockError):
            validate_phone_clock(self.ids.tolist(), self.words, self.edges)
        with self.assertRaises(PhoneClockError):
            validate_phone_clock(np.array(['', 'B', 'A', 'D']), self.words, self.edges)
if __name__ == '__main__':
    unittest.main()
