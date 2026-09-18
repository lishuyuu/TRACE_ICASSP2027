import unittest
from trace_tts.metrics import cast_pair, rank_correct, aggregate_cell, aggregate_seeds, complete_mean, switching_outcomes
from trace_tts.calibration import select_gain

class MetricTests(unittest.TestCase):

    def test_two_directions_required(self):
        row = cast_pair([0], [1], [0.8, 0.2], [0.1, 0.7], 0, 1)
        self.assertTrue(row['pair_correct'])
        self.assertAlmostEqual(row['delta_P'], 0.6)
        self.assertFalse(cast_pair([0], [0], None, None, 0, 1)['pair_correct'])

    def test_competitor_rejects_endpoint(self):
        self.assertFalse(cast_pair([0, 1], [1], None, None, 0, 1)['pair_correct'])

    def test_missing_is_not_zero(self):
        self.assertIsNone(cast_pair(None, [1], None, None, 0, 1)['pair_correct'])
        self.assertIsNone(rank_correct([float('nan'), 1], 0))

    def test_ties_fail(self):
        self.assertFalse(rank_correct([0.5, 0.5], 0))

    def test_native_same_output(self):
        row = cast_pair([0], [0], [0.8, 0.2], [0.8, 0.2], 0, 1)
        self.assertFalse(row['pair_correct'])
        self.assertAlmostEqual(row['delta_P'], 0)

    def test_exact_denominators(self):
        row = cast_pair([0], [1], [1.0, 0.0], [0.0, 1.0], 0, 1)
        actual = aggregate_cell([row], [True], [2.0, 4.0], expected_pairs=1, expected_expresso=1)
        self.assertEqual(actual['Pair-Correct'], 100)
        self.assertEqual(actual['UTMOS'], 3)
        with self.assertRaises(ValueError):
            aggregate_cell([row], [True], [2.0], expected_pairs=1, expected_expresso=1)

    def test_prompt_then_seed_sample_sd(self):
        prompts = ['a', 'b', 'c', 'd']
        cells = {(s, p): float(s - 2702) for s in (2703, 2704, 2705) for p in prompts}
        self.assertEqual(aggregate_seeds(cells, prompts=prompts), {'mean': 2.0, 'sample_sd': 1.0})

    def test_dev_gain_tie(self):
        self.assertEqual(select_gain({1.0: 25, 1.5: 25}, candidates=[1.0, 1.5], split='dev'), 1.0)
        with self.assertRaises(ValueError):
            select_gain({1.0: 25}, candidates=[1.0], split='test')

    def test_invalid_target_types_rejected(self):
        for targets in [(False, True), (0.5, 1.5), ('0', '1')]:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                cast_pair([0], [1], None, None, *targets)

    def test_duplicate_and_out_of_range_detections_rejected(self):
        with self.assertRaises(ValueError):
            cast_pair([0, 0], [1], None, None, 0, 1)
        with self.assertRaises(ValueError):
            cast_pair([0, 3], [1], [0.8, 0.2], [0.2, 0.8], 0, 1)

    def test_repeated_seed_not_counted_as_independent(self):
        prompts = ['a', 'b', 'c', 'd']
        cells = {(s, p): 1.0 for s in (2703, 2704) for p in prompts}
        with self.assertRaisesRegex(ValueError, 'Three distinct'):
            aggregate_seeds(cells, prompts=prompts, seeds=(2703, 2703, 2704))

    def test_nonfinite_mean_and_invalid_strings(self):
        self.assertIsNone(complete_mean([1.0, float('nan')]))
        self.assertIsNone(complete_mean([1.0, None]))
        with self.assertRaises(ValueError):
            complete_mean(['1.0', 2.0])

    def test_boolean_gain_is_not_scalar(self):
        with self.assertRaises(ValueError):
            select_gain({True: 25.0}, candidates=[True], split='dev')

    def test_selected_gain_is_accepted_by_inference(self):
        from trace_tts.gains import scale_plan, validate_gain
        import torch
        gain = select_gain({0.5: 25.0, 1.0: 20.0}, candidates=[0.5, 1.0], split='dev')
        self.assertEqual(validate_gain(gain), 0.5)
        plan = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
        self.assertTrue(torch.equal(scale_plan(plan, gain), plan * 0.5))
        self.assertEqual(validate_gain(0), 0)
        self.assertTrue(torch.equal(scale_plan(plan, 0), torch.zeros_like(plan)))
        for invalid in (-1, True, float('nan')):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_gain(invalid)

    def test_switching_bins_share_pair_correct_and_pair_contrast(self):
        rows = [cast_pair([0], [1], None, None, 0, 1), cast_pair([0], [0], None, None, 0, 1), cast_pair([1], [1], None, None, 0, 1), cast_pair([1], [0], None, None, 0, 1), cast_pair([0], [1], None, None, 0, 1)]
        cell = aggregate_cell(rows, [True], [1.0] * 10, expected_pairs=5, expected_expresso=1)
        bins = switching_outcomes(rows)
        self.assertEqual(bins['f2'], cell['Pair-Correct'])
        self.assertAlmostEqual(bins['f1'], 2 * (cell['Pair-Contrast'] - cell['Pair-Correct']))
        self.assertAlmostEqual(bins['f0'], 100 - 2 * cell['Pair-Contrast'] + cell['Pair-Correct'])
        self.assertAlmostEqual(sum(bins.values()), 100)

    def test_switching_missing_or_inconsistent_outcomes(self):
        row = cast_pair(None, [1], None, None, 0, 1)
        self.assertEqual(switching_outcomes([row]), {'f0': None, 'f1': None, 'f2': None})
        row = cast_pair([0], [1], None, None, 0, 1)
        row['pair_correct'] = False
        with self.assertRaises(ValueError):
            switching_outcomes([row])
if __name__ == '__main__':
    unittest.main()
