from dataclasses import replace
import unittest
import numpy as np
from trace_tts import trace_audio_features as f
from trace_tts.trace_phone_clock import warp_phone_features

def artificial_intervals():
    words = (f.Interval(0, 0.1, ''), f.Interval(0.1, 0.45, 'Take'), f.Interval(0.45, 0.6, ''), f.Interval(0.6, 1, 'two'), f.Interval(1, 1.2, ''))
    phones = (f.Interval(0, 0.1, ''), f.Interval(0.1, 0.2, 'T'), f.Interval(0.2, 0.4, 'EY1'), f.Interval(0.4, 0.45, 'K'), f.Interval(0.45, 0.6, 'sil'), f.Interval(0.6, 0.7, 'T'), f.Interval(0.7, 1, 'UW1'), f.Interval(1, 1.2, ''))
    return (words, phones)

def artificial_grid():
    words, phones = artificial_intervals()
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', 'xmin = 0', 'xmax = 1.2', 'tiers? <exists>', 'size = 2', 'item []:']
    for i, (name, intervals) in enumerate((('words', words), ('ex03 "quoted" phones', phones)), 1):
        escaped = name.replace('"', '""')
        lines += [f'    item [{i}]:', '        class = "IntervalTier"', f'        name = "{escaped}"', '        xmin = 0', '        xmax = 1.2', f'        intervals: size = {len(intervals)}']
        for j, row in enumerate(intervals, 1):
            lines += [f'        intervals [{j}]:', f'            xmin = {row.start}', f'            xmax = {row.end}', f'            text = "{row.label}"']
    return '\n'.join(lines) + '\n'

def artificial_descriptor(values):
    values = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(values)
    missing = tuple((tuple((None if b else 'artificial_missing_fixture' for b in row)) for row in observed))
    return f.Descriptor(tuple((f'word{i}' for i in range(len(values)))), values, observed, missing, np.zeros(len(values), dtype=np.int64))

def artificial_group(n, group_id):
    raw = np.zeros((n, n, 5))
    raw[:] = (np.eye(n) * n)[:, :, None] * np.arange(1, 6)[None, None, :]
    return f.prepare_group([artificial_descriptor(x) for x in raw], list(range(n)), group_id=group_id, corpus='Expresso', split='TRAIN')

class ArtificialFeatureTests(unittest.TestCase):

    def alignment(self):
        return f.build_alignment(*artificial_intervals(), ['take', 'two'], 1.2, 60)

    def test_long_textgrid_binding_keeps_raw_and_escaped_tier(self):
        a = f.alignment_from_textgrid(artificial_grid(), 'TAKE, two!', 60, phone_tier='ex03 "quoted" phones')
        self.assertEqual(a.raw_word_intervals, artificial_intervals()[0])
        self.assertEqual(a.raw_phone_intervals, artificial_intervals()[1])
        self.assertEqual(a.words, ('take', 'two'))
        self.assertEqual(a.phone_ids.tolist(), ['T', 'EY1', 'K', 'T', 'UW1'])
        self.assertEqual(f.canonical_words('It’s 12, A-B.'), ("it's", '12', 'a', 'b'))
        with self.assertRaises(f.FeatureError):
            f.parse_textgrid(artificial_grid().replace('intervals: size = 5', 'intervals: size = 4', 1))

    def test_six_decimal_endpoint_rounding_only_is_tolerated(self):
        rounded = artificial_grid().replace('xmax = 1.2', 'xmax = 1.200000333')
        alignment = f.alignment_from_textgrid(rounded, 'take two', 60, duration_sec=1.2, phone_tier='ex03 "quoted" phones')
        self.assertEqual(alignment.duration_sec, 1.2)
        self.assertEqual(alignment.raw_word_intervals[-1].end, 1.200000333)
        words, phones = artificial_intervals()
        with self.assertRaises(f.FeatureError):
            f.build_alignment(words[:-1] + (replace(words[-1], end=1.200001),), phones, ['take', 'two'], 1.2, 60)

    def test_midpoint_pauses_total_clock_and_exact_noop(self):
        a = self.alignment()
        np.testing.assert_allclose(a.clock_edges_seconds, [0, 0.2, 0.4, 0.525, 0.7, 1.2])
        np.testing.assert_allclose(a.word_pi, [0.4375, 0.5625])
        np.testing.assert_allclose(a.raw_word_durations, [0.35, 0.4])
        self.assertAlmostEqual(a.pause_seconds, 0.45)
        self.assertEqual(len(a.pause_allocations), 3)
        self.assertEqual(a.clock_edges[-1], 60)
        x = np.arange(120, dtype=np.float32).reshape(60, 2)
        y, diagnostic = warp_phone_features(x, a.phone_ids, a.phone_word_indices, a.clock_edges, a.phone_ids, a.phone_word_indices, a.clock_edges.copy())
        self.assertEqual(y.tobytes(), x.tobytes())
        self.assertFalse(np.shares_memory(y, x))

    def test_identity_oov_overlap_and_missing_word_are_failures(self):
        a = self.alignment()
        f.require_matching_phones([a, a])
        changed = a.phone_ids.copy()
        changed[0] = 'D'
        with self.assertRaises(f.FeatureError):
            f.require_matching_phones([a, replace(a, phone_ids=changed)])
        words, phones = artificial_intervals()
        for broken in (phones[:1] + (replace(phones[1], label='spn'),) + phones[2:], phones[:1] + (replace(phones[1], end=phones[1].start),) + phones[2:]):
            with self.assertRaises(f.FeatureError):
                f.build_alignment(words, broken, ['take', 'two'], 1.2, 60)
        with self.assertRaises(f.FeatureError):
            f.build_alignment(words, phones, ['take', 'three'], 1.2, 60)

    def test_primary_nucleus_then_longest_earliest_and_absent(self):
        words = (f.Interval(0, 0.75, 'vowel'), f.Interval(0.75, 1, 't'))
        phones = (f.Interval(0, 0.25, 'AA0'), f.Interval(0.25, 0.5, 'EH1'), f.Interval(0.5, 0.75, 'IY1'), f.Interval(0.75, 1, 'T'))
        a = f.build_alignment(words, phones, ['vowel', 't'], 1, 50)
        self.assertEqual(a.nucleus_phone_indices.tolist(), [1, -1])
        self.assertTrue(np.isnan(a.nucleus_intervals[1]).all())
        unstressed = tuple((replace(p, label=p.label.replace('1', '0')) for p in phones))
        b = f.build_alignment(words, unstressed, ['vowel', 't'], 1, 50)
        self.assertEqual(b.nucleus_phone_indices.tolist(), [0, -1])

    def test_log_f0_legendre_energy_and_clock_clr(self):
        a = self.alignment()
        times = np.arange(240) * 0.005
        pitch = np.zeros_like(times)
        expected = np.array([np.log(150), 0.2, -0.1])
        for start, end in a.nucleus_intervals:
            selected = (times >= start) & (times < end)
            phase = 2 * (times[selected] - start) / (end - start) - 1
            pitch[selected] = np.exp(expected[0] + expected[1] * phase + expected[2] * (3 * phase ** 2 - 1) / 2)
        audio = np.full(28800, 0.2)
        d = f.describe(audio, 24000, a, f0_times=times, f0=pitch)
        np.testing.assert_allclose(d.values[:, :3], np.tile(expected, (2, 1)), atol=1e-12)
        np.testing.assert_allclose(d.values[:, 3], np.log(0.2 + 1e-08))
        np.testing.assert_allclose(d.values[:, 4], np.log(a.word_pi) - np.log(a.word_pi).mean())
        self.assertTrue(d.observed.all())

    def test_pitch_missing_keeps_nan_and_does_not_become_zero_observation(self):
        a = self.alignment()
        times = np.arange(240) * 0.005
        pitch = np.zeros_like(times)
        valid = np.flatnonzero((times >= 0.2) & (times < 0.4))
        pitch[valid[:4]] = 150
        pitch[valid[4:6]] = [np.nan, np.inf]
        pitch[(times >= 0.7) & (times < 1)] = 120
        d = f.describe(np.full(28800, 0.2), 24000, a, f0_times=times, f0=pitch)
        self.assertEqual(d.pitch_sample_count[0], 4)
        self.assertTrue(np.isnan(d.values[0, :3]).all())
        self.assertFalse(d.observed[0, :3].any())
        self.assertTrue(d.observed[:, 3:].all())
        self.assertIn('fewer_than_five', d.raw_missing[0][0])

    def test_completion_support_and_valid_J_keep_original_observation(self):
        raw = np.ones((3, 3, 5))
        raw[:, :, 0] = [[6, 2, 1], [2, 6, 3], [np.nan, 1, 6]]
        g = f.prepare_group([artificial_descriptor(x) for x in raw], [0, 1, 2], group_id='artificial', corpus='Expresso', split='train')
        self.assertTrue(np.isnan(g.raw[2, 0, 0]))
        self.assertEqual(g.completed[2, 0, 0], 4)
        self.assertFalse(g.observed[2, 0, 0])
        self.assertFalse(g.O[2, 0, 0])
        self.assertEqual(g.valid_pairs_by_cue[0], ((0, 1), (1, 2)))
        np.testing.assert_allclose(g.centered.sum(axis=1), 0, atol=1e-15)
        raw[1, 0, 0] = np.nan
        h = f.prepare_group([artificial_descriptor(x) for x in raw], [0, 1, 2], group_id='unsupported', corpus='Expresso', split='train')
        self.assertFalse(h.support[0, 0])
        self.assertTrue(h.observed[0, 0, 0])
        self.assertFalse(h.O[0, 0, 0])
        self.assertEqual(h.completed[:, 0, 0].tolist(), [0, 0, 0])

    def test_group_uniform_scale_active_cues_Eq1_Eq2_inputs(self):
        inputs = f.fit_training_inputs([artificial_group(2, 'small'), artificial_group(3, 'large')])
        self.assertEqual(inputs.cue_indices, (0, 1, 2, 3, 4))
        self.assertEqual(inputs.scale_group_counts[0], 2)
        self.assertAlmostEqual(inputs.scales[0], np.sqrt(1.5))
        np.testing.assert_allclose(inputs.all_scales, np.sqrt(1.5) * np.arange(1, 6))
        self.assertEqual(inputs.valid_pairs_by_cue[0], ((0, 0, 1), (1, 0, 1), (1, 0, 2), (1, 1, 2)))
        self.assertAlmostEqual(inputs.dbar[0], 5.5 / np.sqrt(1.5))
        np.testing.assert_allclose(inputs.omega, np.ones(5) / np.sqrt(5))
        for phi, z in zip(inputs.phi, inputs.z):
            np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-15)
            np.testing.assert_allclose(phi[0] - phi[1], z[0] - z[1])

    def test_five_cues_are_not_silently_reduced(self):
        raw = np.ones((2, 2, 5))
        raw[:, :, 0] = [[2, 0], [0, 2]]
        group = f.prepare_group([artificial_descriptor(x) for x in raw], [0, 1], group_id='synthetic_flat', corpus='Expresso', split='train')
        with self.assertRaisesRegex(f.FeatureError, 'all five cues'):
            f.fit_training_inputs([group])

    def test_support_requires_two_renditions_not_two_words(self):
        raw = np.ones((2, 2, 5))
        raw[:, 1, :3] = np.nan
        group = f.prepare_group([artificial_descriptor(x) for x in raw], [0, 1], group_id='synthetic_support', corpus='Expresso', split='train')
        self.assertTrue(group.support[0, :3].all())
        self.assertFalse(group.support[1, :3].any())

    def test_train_only_fit_and_invalid_mask_or_missing_energy_rejected(self):
        g = artificial_group(2, 'fixture')
        for invalid in (replace(g, corpus='MBOPP'), replace(g, split='validation')):
            with self.assertRaises(f.FeatureError):
                f.fit_training_inputs([invalid])
        with self.assertRaises(f.FeatureError):
            f.fit_training_inputs([g, g])
        d = artificial_descriptor(np.ones((2, 5)))
        with self.assertRaises(f.FeatureError):
            f.prepare_group([d, replace(d, observed=np.zeros((2, 5), dtype=bool))], [0, 1], group_id='bad_mask', corpus='Expresso', split='train')
        values = d.values.copy()
        values[0, 3] = np.nan
        with self.assertRaises(f.FeatureError):
            f.prepare_group([d, artificial_descriptor(values)], [0, 1], group_id='missing_energy', corpus='Expresso', split='train')
if __name__ == '__main__':
    unittest.main(verbosity=2)
