import inspect
import unittest
import torch
from trace_tts import trace_core as core
from trace_tts import trace_networks as networks
FIXTURE_SEED = 84391
DTYPE = torch.float64

def tensor(values):
    return torch.tensor(values, dtype=DTYPE)

class NetworkTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(FIXTURE_SEED)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-09, atol=1e-09)

    def planner(self, **changes):
        kwargs = dict(native_dim=6, speaker_dim=3, width=8, layers=2, heads=2, feedforward_dim=13, dropout=0.0, omega=core.normalize_direction(tensor([1.0, 2.0, -1.0]), min_norm=1e-12), position_base=10000.0, activation='gelu', norm_first=True, norm_eps=1e-05)
        kwargs.update(changes)
        return networks.RelationalPlanner(**kwargs)

    def backbone(self, **changes):
        kwargs = dict(state_dim=80, native_dim=5, speaker_dim=3, width=8, layers=2, kernel_size=3, dropout=0.0, activation='silu', norm_eps=1e-05)
        kwargs.update(changes)
        return networks.FrameResidualBackbone(**kwargs).double()

    def planner_inputs(self):
        return (torch.randn((2, 5, 6), dtype=DTYPE), torch.tensor([1, 3]), torch.randn((2, 3), dtype=DTYPE), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]))

    def frame_inputs(self):
        z = torch.randn((3, 6, 80), dtype=DTYPE)
        condition = networks.FrameCondition(torch.randn((6, 5), dtype=DTYPE), torch.randn(3, dtype=DTYPE), torch.tensor([1, 1, 1, 1, 0, 0]))
        return (z, condition)

    def test_planner_shapes_target_order_and_actual_word_centering(self):
        planner = self.planner().eval()
        native, target, speaker, valid = self.planner_inputs()
        result = planner(native, target, speaker, valid)
        self.assertEqual(result.coordinates.shape, (2, 5, 3))
        self.assertEqual(result.b.shape, (2, 5))
        self.assertEqual(result.u.shape, (2, 5, 3))
        for row, count in enumerate(valid.sum(dim=1).tolist()):
            q = result.coordinates[row, :count]
            self.assertClose(q.sum(dim=0), torch.zeros(3, dtype=DTYPE))
            scores = q @ planner.omega
            self.assertTrue(bool((scores[target[row]] - scores >= -1e-10).all()))
            self.assertClose(q, core.ordered_plan(result.b[row, :count], result.u[row, :count], planner.omega, int(target[row])))
        self.assertTrue(torch.equal(result.coordinates[0, 3:], torch.zeros((2, 3), dtype=DTYPE)))

    def test_planner_padding_content_and_other_batch_items_cannot_leak(self):
        planner = self.planner().eval()
        native, target, speaker, valid = self.planner_inputs()
        original = planner(native, target, speaker, valid).coordinates
        changed = native.clone()
        changed[0, 3:] = 9999
        changed[1] *= -17
        new = planner(changed, target, speaker, valid).coordinates
        self.assertClose(new[0], original[0])
        alone = planner(native[:1, :3], target[:1], speaker[:1], valid[:1, :3]).coordinates
        self.assertClose(alone[0], original[0, :3])

    def test_planner_is_trainable_but_native_speaker_omega_are_frozen(self):
        omega = core.normalize_direction(tensor([1.0, 2.0, -1.0]), min_norm=1e-12).requires_grad_()
        planner = self.planner(omega=omega)
        native, targets, speaker, valid = self.planner_inputs()
        native.requires_grad_()
        speaker.requires_grad_()
        plan = planner(native, targets, speaker, valid).coordinates
        plan.square().sum().backward()
        self.assertIsNone(native.grad)
        self.assertIsNone(speaker.grad)
        self.assertIsNone(omega.grad)
        self.assertFalse(planner.omega.requires_grad)
        self.assertNotIn('omega', dict(planner.named_parameters()))
        for parameter in (planner.b_head.weight, planner.u_head.weight, planner.target_role.weight, planner.native_projection.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_hard_target_switch_and_no_fixed_length_position_slots(self):
        planner = self.planner().eval()
        native = torch.randn((1, 7, 6), dtype=DTYPE)
        speaker, valid = (torch.randn((1, 3), dtype=DTYPE), torch.ones((1, 7), dtype=torch.bool))
        for target in (0, 6):
            result = planner(native, torch.tensor([target]), speaker, valid)
            scores = result.coordinates[0] @ planner.omega
            self.assertEqual(int(scores.argmax()), target)
        positions = networks.sinusoidal_positions(9, 5, base=10000.0, reference=native)
        self.assertEqual(positions.shape, (9, 5))
        self.assertClose(positions[0], tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
        self.assertFalse(any(('position' in name for name, _ in planner.named_parameters())))

    def test_planner_refuses_invalid_target_masks_omega_and_architecture(self):
        planner = self.planner()
        native, target, speaker, valid = self.planner_inputs()
        with self.assertRaises(ValueError):
            planner(native, torch.tensor([3, 3]), speaker, valid)
        invalid = valid.clone()
        invalid[0] = torch.tensor([1, 0, 1, 0, 0])
        with self.assertRaises(ValueError):
            planner(native, target, speaker, invalid)
        with self.assertRaises(TypeError):
            planner(native, target.to(DTYPE), speaker, valid)
        for changes in (dict(width=7), dict(omega=tensor([1.0, 1.0, 1.0])), dict(dropout=1.0), dict(layers=0), dict(norm_eps=0.0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.planner(**changes)

    def test_targetflag_is_centered_target_only_and_parameter_matched_without_dummy_parameters(self):
        omega = core.normalize_direction(tensor([1.0, 2.0, -1.0]), min_norm=1e-12)
        common = dict(native_dim=5, speaker_dim=4, width=128, layers=2, heads=4, feedforward_dim=512, dropout=0.1, position_base=10000.0, activation='gelu', norm_first=True, norm_eps=1e-05)
        trace = networks.RelationalPlanner(**common, omega=omega)
        baseline = networks.TargetFlagPlanner(**common, cues=3).to(dtype=DTYPE)
        native = torch.randn((2, 5, 5), dtype=DTYPE, requires_grad=True)
        speaker = torch.randn((2, 4), dtype=DTYPE, requires_grad=True)
        valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        targets = torch.tensor([1, 3])
        result = baseline(native, targets, speaker, valid)
        self.assertEqual(result.coordinates.shape, (2, 5, 3))
        self.assertClose(result.coordinates[0, :3].sum(dim=0), torch.zeros(3, dtype=DTYPE))
        self.assertClose(result.coordinates[1].sum(dim=0), torch.zeros(3, dtype=DTYPE))
        self.assertTrue(torch.equal(result.coordinates[0, 3:], torch.zeros((2, 3), dtype=DTYPE)))
        result.coordinates.square().sum().backward()
        self.assertIsNone(native.grad)
        self.assertIsNone(speaker.grad)
        self.assertTrue(all((parameter.grad is not None for parameter in baseline.parameters())))
        trace_count = sum((parameter.numel() for parameter in trace.parameters()))
        baseline_count = sum((parameter.numel() for parameter in baseline.parameters()))
        self.assertGreaterEqual(baseline_count / trace_count, 0.99)
        self.assertLessEqual(baseline_count / trace_count, 1.01)
        self.assertFalse(any(('dummy' in name for name, _ in baseline.named_parameters())))

    def test_backbone_same_clock_no_command_api_or_endpoint_mixing(self):
        backbone = self.backbone().eval()
        z, condition = self.frame_inputs()
        output = backbone(z, 0.4, condition)
        self.assertEqual(output.shape, (3, 6, 8))
        self.assertTrue(torch.equal(output[:, 4:], torch.zeros((3, 2, 8), dtype=DTYPE)))
        changed = z.clone()
        changed[2] *= -13
        self.assertClose(backbone(changed, 0.4, condition)[:2], output[:2])
        self.assertEqual(tuple(inspect.signature(backbone.forward).parameters), ('z', 'time', 'condition'))
        with self.assertRaises(TypeError):
            backbone(z, 0.4, condition, command=torch.zeros((3, 6, 2)))

    def test_backbone_padding_cannot_bleed_through_affine_norm_or_convolution(self):
        backbone = self.backbone().eval()
        with torch.no_grad():
            for norm in backbone.norms:
                norm.bias.fill_(0.7)
        z, condition = self.frame_inputs()
        full = backbone(z, 0.4, condition)
        changed = z.clone()
        changed[:, 4:] = 9999
        native_changed = condition.native_frames.clone()
        native_changed[4:] = -9999
        changed_condition = networks.FrameCondition(native_changed, condition.speaker, condition.valid_frames)
        self.assertClose(backbone(changed, 0.4, changed_condition), full)
        short_condition = networks.FrameCondition(condition.native_frames[:4], condition.speaker, torch.ones(4, dtype=torch.bool))
        self.assertClose(backbone(z[:, :4], 0.4, short_condition), full[:, :4])

    def test_backbone_retains_state_jacobian_and_freezes_native_condition(self):
        backbone = self.backbone()
        z, condition = self.frame_inputs()
        z.requires_grad_()
        native = condition.native_frames.clone().requires_grad_()
        speaker = condition.speaker.clone().requires_grad_()
        condition = networks.FrameCondition(native, speaker, condition.valid_frames)
        result = backbone(z, 0.4, condition)
        result[:, :4, 0].sum().backward()
        self.assertGreater(float(z.grad[:, :4].abs().sum()), 0.0)
        self.assertTrue(torch.equal(z.grad[:, 4:], torch.zeros_like(z.grad[:, 4:])))
        self.assertIsNone(native.grad)
        self.assertIsNone(speaker.grad)
        self.assertGreater(float(backbone.state_projection.weight.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            self.backbone(kernel_size=2)
        with self.assertRaises(ValueError):
            backbone(z, 0.4, networks.FrameCondition(native[None].expand(3, -1, -1), speaker, condition.valid_frames))

    def test_backbone_composes_with_zero_gate_and_same_state_swap(self):
        backbone = self.backbone(dropout=0.3)
        adapter = core.ZeroPreservingResidual(backbone, width=8, output_dim=80).double()
        z, condition = self.frame_inputs()
        zeros = torch.zeros((3, 6, 2), dtype=DTYPE)
        with torch.no_grad():
            adapter.output.weight.fill_(0.2)
        self.assertTrue(torch.equal(adapter(z, 0.4, condition, zeros, condition.valid_frames), torch.zeros_like(z)))
        command = torch.ones_like(zeros)
        first, second = adapter.forward_pair(z, 0.4, condition, command, command.clone(), condition.valid_frames)
        self.assertTrue(torch.equal(first, second))

    def test_huber_exact_observed_normalization_and_masked_gradients(self):
        prediction = tensor([[[0.5, 3.0], [99.0, 99.0]], [[2.0, 99.0], [99.0, 99.0]]]).requires_grad_()
        target = torch.zeros_like(prediction, requires_grad=True)
        observed = torch.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 0]]])
        valid = torch.tensor([[1, 0], [1, 0]])
        result = networks.observed_huber_loss(prediction, target, observed, valid, delta=2.0)
        self.assertClose(result.per_endpoint, tensor([(0.125 + 4) / 2, 2.0]))
        self.assertAlmostEqual(float(result.loss), ((0.125 + 4) / 2 + 2) / 2)
        self.assertTrue(torch.equal(result.observed_counts, torch.tensor([2, 1])))
        result.loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.equal(prediction.grad[observed == 0], torch.zeros(5, dtype=DTYPE)))
        self.assertGreater(float(prediction.grad[observed == 1].abs().sum()), 0.0)

    def test_huber_refuses_empty_supervision_padded_observations_and_nonfinite(self):
        prediction = torch.zeros((2, 2, 3), dtype=DTYPE)
        observed = torch.ones_like(prediction, dtype=torch.bool)
        valid = torch.ones((2, 2), dtype=torch.bool)
        empty = observed.clone()
        empty[0] = False
        with self.assertRaises(ValueError):
            networks.observed_huber_loss(prediction, prediction, empty, valid, delta=1.0)
        with self.assertRaises(ValueError):
            networks.observed_huber_loss(prediction, prediction, observed, torch.tensor([[1, 0], [1, 1]]), delta=1.0)
        for delta in (0.0, -1.0, float('nan')):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                networks.observed_huber_loss(prediction, prediction, observed, valid, delta=delta)
        target = prediction.clone()
        target[0, 0, 0] = float('nan')
        observed[0, 0, 0] = False
        with self.assertRaises(ValueError):
            networks.observed_huber_loss(prediction, target, observed, valid, delta=1.0)
if __name__ == '__main__':
    unittest.main()
