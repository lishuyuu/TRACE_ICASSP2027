import unittest
import torch
from torch import nn
from trace_tts import trace_core as core
DTYPE = torch.float64
FIXTURE_SEED = 73271

def tensor(values):
    return torch.tensor(values, dtype=DTYPE)

class FixtureBranch(nn.Module):

    def __init__(self, width=4):
        super().__init__()
        self.linear = nn.Linear(80, width)
        self.calls = 0

    def forward(self, z, time, condition):
        self.calls += 1
        return self.linear(z + condition) + time

class TraceCoreTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(FIXTURE_SEED)

    def assertClose(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10, **kwargs)

    def test_masked_center_zero_support_and_no_implicit_missing_fill(self):
        values = tensor([2.0, 100.0, 4.0])
        self.assertClose(core.masked_center(values, torch.tensor([1, 0, 1])), tensor([-1.0, 0.0, 1.0]))
        self.assertTrue(torch.equal(core.masked_center(values, torch.zeros(3)), torch.zeros_like(values)))
        with self.assertRaises(ValueError):
            core.masked_center(tensor([2.0, float('nan'), 4.0]), torch.tensor([1, 0, 1]))

    def test_standardization_and_group_center_preserve_every_pair(self):
        values = torch.arange(18, dtype=DTYPE).reshape(3, 3, 2).square()
        phi = core.standardize_descriptors(values, torch.tensor([[1, 1], [1, 0], [1, 1]]), tensor([2.0, 4.0]))
        self.assertClose(phi.sum(dim=1), torch.zeros((3, 2), dtype=DTYPE))
        centered, mean = core.group_center(phi)
        self.assertClose(centered.sum(dim=0), torch.zeros((3, 2), dtype=DTYPE))
        self.assertClose(mean, phi.mean(dim=0))
        for i in range(3):
            for j in range(3):
                self.assertClose(centered[i] - centered[j], phi[i] - phi[j])

    def test_explicit_pair_direction_and_exact_unit_normalization(self):
        group = tensor([[[1.0, -2.0], [-1.0, 2.0]], [[-1.0, 2.0], [1.0, -2.0]]])
        direction, omega = core.estimate_direction_from_pairs([group], [[0, 1]], [[(0, 0, 1)], [(0, 0, 1)]], min_norm=1e-12)
        self.assertClose(direction, tensor([4.0, -8.0]))
        self.assertClose(omega.norm(), tensor(1.0))
        perpendicular = torch.eye(2, dtype=DTYPE) - torch.outer(omega, omega)
        self.assertClose(perpendicular @ omega, torch.zeros(2, dtype=DTYPE))
        for bad in (tensor([0.0, 0.0]), tensor([1e-14, 0.0]), tensor([float('inf'), 1.0])):
            with self.assertRaises(ValueError):
                core.normalize_direction(bad, min_norm=1e-12)
        with self.assertRaises(ValueError):
            core.estimate_direction_from_pairs([group], [[0, 1]], [[], [(0, 0, 1)]], min_norm=1e-12)

    def test_projection_known_solution_allows_ties_without_added_margin(self):
        result = core.project_target_maximum(tensor([[-2.0], [1.0], [1.0]]), torch.ones((3, 1)), tensor([1.0]), 0, lambda_mis=0.3)
        self.assertClose(result.q, torch.zeros((3, 1), dtype=DTYPE))
        self.assertAlmostEqual(result.objective, 6.0)
        self.assertLess(result.stationarity_error, 1e-10)
        self.assertLess(result.complementarity_error, 1e-10)
        empty = core.project_target_maximum(tensor([[5.0], [-3.0]]), torch.zeros((2, 1)), tensor([1.0]), 1, lambda_mis=0.3)
        self.assertTrue(torch.equal(empty.q, torch.zeros_like(empty.q)))
        self.assertEqual(empty.objective, 0.0)

    def test_projection_partial_observation_unique_kkt_and_feasible_identity(self):
        z = tensor([[2.0, -1.0], [-3.0, 4.0], [1.0, -3.0]])
        omega = core.normalize_direction(tensor([2.0, 1.0]), min_norm=1e-12)
        observed = torch.tensor([[1, 0], [1, 1], [0, 1]])
        result = core.project_target_maximum(z, observed, omega, 1, lambda_mis=0.2)
        self.assertClose(result.q.sum(dim=0), torch.zeros(2, dtype=DTYPE))
        scores = result.q @ omega
        self.assertTrue(bool((scores[1] - scores >= -1e-10).all()))
        self.assertLess(result.stationarity_error, 1e-10)
        self.assertLess(result.complementarity_error, 1e-10)
        direct = (observed * (result.q - z)).square().sum() + 0.2 * ((1 - observed) * result.q).square().sum()
        self.assertAlmostEqual(result.objective, float(direct))
        feasible = tensor([[-1.0], [2.0], [-1.0]])
        same = core.project_target_maximum(feasible, torch.ones_like(feasible), tensor([1.0]), 1, lambda_mis=0.2)
        self.assertClose(same.q, feasible)

    def test_projection_invalid_inputs_fail_instead_of_repairing_the_method(self):
        z, mask = (tensor([[1.0], [-1.0]]), torch.ones((2, 1)))
        for bad_lambda in (0.0, -1.0, float('nan')):
            with self.assertRaises(ValueError):
                core.project_target_maximum(z, mask, tensor([1.0]), 0, lambda_mis=bad_lambda)
        with self.assertRaises(ValueError):
            core.project_target_maximum(z, mask, tensor([0.5]), 0, lambda_mis=0.1)
        with self.assertRaises(ValueError):
            core.project_target_maximum(z, tensor([[1.0], [0.5]]), tensor([1.0]), 0, lambda_mis=0.1)

    def test_planner_target_order_zero_mean_and_perpendicular_freedom(self):
        omega = core.normalize_direction(tensor([1.0, 2.0, -1.0]), min_norm=1e-12)
        b = tensor([-0.5, 0.7, 1.0, -1.0])
        u = torch.randn((4, 3), dtype=DTYPE, requires_grad=True)
        plan = core.ordered_plan(b, u, omega, 2)
        self.assertClose(plan.sum(dim=0), torch.zeros(3, dtype=DTYPE))
        scores = plan @ omega
        expected_g = -torch.nn.functional.softplus(b)
        expected_g[2] = 0.0
        self.assertClose(scores, expected_g - expected_g.mean())
        self.assertTrue(bool((scores[2] > scores[torch.tensor([0, 1, 3])]).all()))
        plan.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(u.grad).all()))
        with self.assertRaises(ValueError):
            core.ordered_plan(b, u, omega * 0.99, 2)

    def test_physical_cue_restore_and_positive_duration_only_clock(self):
        physical = core.physical_cues(tensor([[1.0, 2.0], [-1.0, -2.0]]), tensor([0.2, 0.3]), [0, 4])
        self.assertClose(physical[:, 0], tensor([0.2, -0.2]))
        self.assertClose(physical[:, 4], tensor([0.6, -0.6]))
        self.assertTrue(torch.equal(physical[:, 1:4], torch.zeros((2, 3), dtype=DTYPE)))
        native = tensor([0.2, 0.3, 0.5])
        self.assertClose(core.duration_clock(native, torch.zeros(3, dtype=DTYPE)), native)
        changed = core.duration_clock(native, tensor([0.3, -0.2, 0.0]))
        self.assertTrue(bool((changed > 0).all()))
        self.assertAlmostEqual(float(changed.sum()), 1.0)
        self.assertAlmostEqual(float(changed[0] / changed[1]), 0.2 / 0.3 * math_exp(0.5))
        with self.assertRaises(ValueError):
            core.duration_clock(tensor([0.0, 0.5, 0.5]), torch.zeros(3, dtype=DTYPE))

    def test_legendre_command_respects_supplied_voicing_and_padding(self):
        cues = tensor([[1.0, 2.0, 3.0, 4.0, 99.0], [2.0, 0.0, 0.0, 5.0, -99.0]])
        command = core.frame_command(cues, torch.tensor([0, 0, 1, -1]), tensor([0.0, 0.5, 1.0, 0.0]), tensor([1.0, 0.0, 1.0, 1.0]), torch.tensor([1, 1, 1, 0]))
        self.assertClose(command, tensor([[2.0, 4.0], [0.0, 4.0], [2.0, 5.0], [0.0, 0.0]]))
        no_duration = cues.clone()
        no_duration[:, 4] = 0
        self.assertTrue(torch.equal(command, core.frame_command(no_duration, torch.tensor([0, 0, 1, -1]), tensor([0.0, 0.5, 1.0, 0.0]), tensor([1.0, 0.0, 1.0, 1.0]), torch.tensor([1, 1, 1, 0]))))

    def test_shared_flow_path_and_full_group_residual_centering(self):
        endpoints = torch.arange(3 * 2 * 80, dtype=DTYPE).reshape(3, 2, 80) / 100
        noise = torch.full((2, 80), 0.25, dtype=DTYPE)
        state, velocity = core.flow_path(endpoints, noise, 0.4, sigma_min=0.1)
        self.assertClose(state, 0.64 * noise + 0.4 * endpoints)
        self.assertClose(velocity, endpoints - 0.9 * noise)
        base = torch.randn_like(endpoints, requires_grad=True)
        residual, centered, mean = core.center_flow_residuals(velocity, base, torch.tensor([1, 0]))
        self.assertClose(centered.sum(dim=0), torch.zeros((2, 80), dtype=DTYPE))
        self.assertClose(mean, residual.mean(dim=0))
        self.assertClose(centered[0] - centered[2], residual[0] - residual[2])
        self.assertFalse(centered.requires_grad)
        self.assertTrue(torch.equal(centered[:, 1], torch.zeros((3, 80), dtype=DTYPE)))
        with self.assertRaises(ValueError):
            core.flow_path(endpoints, noise.expand(3, -1, -1), 0.4, sigma_min=0.1)

    def test_warmup_and_predicted_plans_are_explicitly_stopped(self):
        oracle = torch.ones((3, 2), dtype=DTYPE, requires_grad=True)
        predicted = torch.full((3, 2), 2.0, dtype=DTYPE, requires_grad=True)
        for warmup, expected in ((True, oracle), (False, predicted)):
            selected = core.choose_adapter_plan(oracle, predicted, warmup=warmup)
            self.assertTrue(torch.equal(selected, expected))
            self.assertFalse(selected.requires_grad)

    def make_adapter(self):
        return core.ZeroPreservingResidual(FixtureBranch(), width=4, output_dim=80).double()

    def test_exact_zero_gate_after_nonzero_weights_and_post_cfg_identity(self):
        adapter = self.make_adapter()
        self.assertIsNone(adapter.gate.bias)
        self.assertIsNone(adapter.output.bias)
        self.assertGreater(float(adapter.gate.weight.abs().sum()), 0.0)
        with torch.no_grad():
            adapter.output.weight.fill_(0.4)
        state = torch.randn((2, 3, 80), dtype=DTYPE)
        command = torch.zeros((2, 3, 2), dtype=DTYPE)
        mask = torch.tensor([1, 1, 0])
        residual = adapter(state, 0.5, torch.zeros_like(state), command, mask)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(core.add_after_cfg(state, residual, mask), state))

    def test_zero_output_initialization_has_non_dead_gradient(self):
        adapter = self.make_adapter()
        state = torch.ones((2, 2, 80), dtype=DTYPE)
        command = torch.ones((2, 2, 2), dtype=DTYPE)
        mask = torch.tensor([1, 1])
        output = adapter(state, 0.5, state * 0, command, mask)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        (output - 1).square().mean().backward()
        self.assertGreater(float(adapter.output.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(adapter.gate.weight.grad.abs().sum()), 0.0)
        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.output.weight.fill_(0.2)
        adapter(state, 0.5, state * 0, command, mask).sum().backward()
        self.assertGreater(float(adapter.gate.weight.grad.abs().sum()), 0.0)

    def test_post_cfg_addition_preserves_frozen_base_state_jacobian(self):
        state = torch.ones((2, 2, 80), dtype=DTYPE, requires_grad=True)
        base_velocity = 2 * state
        residual = torch.zeros_like(state)
        result = core.add_after_cfg(base_velocity, residual, torch.tensor([1, 1]))
        self.assertTrue(torch.equal(result, base_velocity))
        result.sum().backward()
        self.assertTrue(torch.equal(state.grad, torch.full_like(state, 2.0)))

    def test_masked_mse_and_exact_explicit_gamma_swap_formula(self):
        correct = torch.ones((2, 2, 80), dtype=DTYPE)
        correct[1, 0] = 2
        correct[:, 1] = 999
        swapped = torch.zeros_like(correct)
        swapped[0, 0], swapped[1, 0] = (2, 1)
        target = torch.zeros_like(correct)
        command = torch.ones((2, 2, 2), dtype=DTYPE)
        result = core.gcra_loss(correct, swapped, target, torch.tensor([1, 0]), command, -command, time=0.25, lambda_swap=0.4, margin=0.5, gamma=2.0)
        self.assertClose(result.correct_mse, tensor([1.0, 4.0]))
        self.assertClose(result.swapped_mse, tensor([4.0, 1.0]))
        self.assertClose(result.swap_hinge, tensor([0.0, 3.5]))
        self.assertAlmostEqual(float(result.loss), (1 + 4 + 0.4 * 0.75 ** 2 * 3.5) / 2)
        self.assertFalse(bool(result.equal_command.any()))
        with self.assertRaises(ValueError):
            core.masked_mse(correct, target, torch.tensor([0, 0]))

    def test_equal_command_hinge_is_retained_and_swap_gradient_cancels(self):
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.output.weight.fill_(0.2)
        state = torch.ones((3, 2, 80), dtype=DTYPE)
        command = torch.ones((3, 2, 2), dtype=DTYPE)
        mask = torch.tensor([1, 1])
        correct, swapped = adapter.forward_pair(state, 0.3, state * 0, command, command.clone(), mask)
        self.assertEqual(adapter.base_branch.calls, 1)
        result = core.gcra_loss(correct, swapped, torch.zeros_like(state), mask, command, command.clone(), time=0.3, lambda_swap=0.5, margin=0.25, gamma=2.0)
        self.assertTrue(bool(result.equal_command.all()))
        self.assertTrue(bool(result.no_swap_gradient_expected.all()))
        self.assertClose(result.swap_hinge, torch.full((3,), 0.25, dtype=DTYPE))
        result.weighted_swap.mean().backward()
        for parameter in adapter.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))
        self.assertGreater(float(result.weighted_swap.mean()), 0.0)

    def test_same_command_different_state_predictions_cannot_fake_swap(self):
        command = torch.zeros((2, 1, 2), dtype=DTYPE)
        target = torch.zeros((2, 1, 80), dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, 'same-state'):
            core.gcra_loss(target, torch.ones_like(target), target, torch.tensor([1]), command, command, time=0.5, lambda_swap=0.5, margin=0.2, gamma=2.0)

    def test_explicit_alternative_sampler_excludes_self_and_is_reproducible(self):
        first = core.sample_alternatives(7, generator=torch.Generator().manual_seed(FIXTURE_SEED))
        second = core.sample_alternatives(7, generator=torch.Generator().manual_seed(FIXTURE_SEED))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool((first != torch.arange(7)).all()))
        self.assertTrue(bool(((first >= 0) & (first < 7)).all()))
        self.assertTrue(torch.equal(core.sample_alternatives(2, generator=torch.Generator().manual_seed(1)), torch.tensor([1, 0])))

def math_exp(value):
    import math
    return math.exp(value)
if __name__ == '__main__':
    unittest.main()
