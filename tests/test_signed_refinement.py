import unittest
import torch
from trace_tts.signed_refinement import signed_renderer_loss, SignedRendererError

class SyntheticAdapter:

    def _features(self, state, time, condition):
        return torch.ones_like(state)

    def _project(self, features, commands, valid):
        return commands * features * valid[None, :, None]

class SignedRefinementTests(unittest.TestCase):

    def inputs(self):
        state = torch.zeros(3, 2, 2)
        command = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
        endpoints = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]])
        return (SyntheticAdapter(), state, 0.5, None, command, state, endpoints, torch.ones(2, dtype=torch.bool))

    def test_coefficients_are_explicit_and_affect_only_requested_terms(self):
        base = signed_renderer_loss(*self.inputs(), lambda_cf=0.0, lambda_p=0.0, margin=1.0, cosine_epsilon=0.01)
        weighted = signed_renderer_loss(*self.inputs(), lambda_cf=2.0, lambda_p=3.0, margin=1.0, cosine_epsilon=0.01)
        torch.testing.assert_close(base.loss, base.fit)
        torch.testing.assert_close(weighted.loss, weighted.fit + 2 * weighted.signed)
        torch.testing.assert_close(weighted.signed, weighted.signed_cosine + 3 * weighted.signed_margin)

    def test_finite_nonzero_pair_filter_keeps_both_directions(self):
        result = signed_renderer_loss(*self.inputs(), lambda_cf=1.0, lambda_p=1.0, margin=1.0, cosine_epsilon=0.01)
        pairs = set(zip(result.pair_source.tolist(), result.pair_target.tolist()))
        self.assertEqual(pairs, {(0, 2), (2, 0), (1, 2), (2, 1)})

    def test_no_valid_pair_and_invalid_weights_fail(self):
        args = list(self.inputs())
        args[6] = torch.zeros_like(args[6])
        with self.assertRaises(SignedRendererError):
            signed_renderer_loss(*args, lambda_cf=1.0, lambda_p=1.0, margin=1.0, cosine_epsilon=0.01)
        with self.assertRaises(ValueError):
            signed_renderer_loss(*self.inputs(), lambda_cf=float('nan'), lambda_p=1.0, margin=1.0, cosine_epsilon=0.01)
if __name__ == '__main__':
    unittest.main()
