import unittest
import numpy as np
import scipy.optimize
import torch
from trace_tts.global_relaxation import boundary_weights, relax_plan, relax_plan_tensor

class GlobalRelaxationTests(unittest.TestCase):

    def setUp(self):
        self.plan = np.array([[0.8, -0.3], [-0.2, 0.8], [-0.7, -0.2], [0.1, -0.3]])
        self.omega = np.array([1.0, 0.0])
        self.boundaries = np.array([0.0, 1.0, 0.5])

    def solve(self, **kwargs):
        settings = dict(lambda_r=2.0, tau_b=0.7)
        settings.update(kwargs)
        return relax_plan(self.plan, self.omega, 0, self.boundaries, **settings)

    def test_boundary_weights(self):
        np.testing.assert_allclose(boundary_weights(self.boundaries, tau_b=0.7), np.exp(-0.7 * self.boundaries))

    def test_constraints(self):
        result = self.solve()
        np.testing.assert_allclose(result.sum(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(result[0], self.plan[0], atol=1e-10)
        self.assertTrue(np.all(result @ self.omega <= result[0] @ self.omega + 1e-10))

    def test_zero_and_disabled_smoothing_are_exact(self):
        np.testing.assert_array_equal(self.solve(lambda_r=0.0), self.plan)
        result = relax_plan(np.zeros_like(self.plan), self.omega, 0, self.boundaries, lambda_r=2.0, tau_b=0.7)
        np.testing.assert_array_equal(result, np.zeros_like(self.plan))

    def test_positive_homogeneity(self):
        expected = self.solve()
        for gain in (0.01, 0.7, 3.0, 100.0):
            result = relax_plan(gain * self.plan, self.omega, 0, self.boundaries, lambda_r=2.0, tau_b=0.7)
            np.testing.assert_allclose(result, gain * expected, rtol=1e-11, atol=1e-11)

    def test_objective_matches_full_constrained_optimization(self):
        weights = boundary_weights(self.boundaries, tau_b=0.7)

        def objective(vector):
            plan = vector.reshape(self.plan.shape)
            return np.square(plan - self.plan).sum() + 2.0 * (weights[:, None] * np.square(np.diff(plan, axis=0))).sum()
        constraints = [{'type': 'eq', 'fun': lambda vector: vector.reshape(self.plan.shape).sum(axis=0)}, {'type': 'eq', 'fun': lambda vector: vector.reshape(self.plan.shape)[0] - self.plan[0]}, {'type': 'ineq', 'fun': lambda vector: vector.reshape(self.plan.shape)[0, 0] - vector.reshape(self.plan.shape)[1:, 0]}]
        independent = scipy.optimize.minimize(objective, self.plan.ravel(), method='SLSQP', constraints=constraints, options={'ftol': 1e-12, 'maxiter': 200})
        self.assertTrue(independent.success, independent.message)
        np.testing.assert_allclose(self.solve(), independent.x.reshape(self.plan.shape), rtol=1e-05, atol=1e-06)
        self.assertLess(objective(self.solve().ravel()), objective(self.plan.ravel()))

    def test_nonuniform_boundaries(self):
        plan = np.array([[0.5], [0.5], [-3.0], [0.5], [0.5], [0.5], [0.5]])
        boundaries = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 30.0])
        result = relax_plan(plan, np.ones(1), 0, boundaries, lambda_r=10.0, tau_b=1.0)
        np.testing.assert_allclose(result.sum(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(result[0], plan[0], atol=1e-10)
        self.assertLessEqual(float(result.max()), 0.5 + 1e-10)
        uniform = relax_plan(plan, np.ones(1), 0, np.zeros(6), lambda_r=10.0, tau_b=1.0)
        self.assertFalse(np.allclose(result, uniform))

    def test_without_ordering_accepts_nonmaximal_target(self):
        plan = self.plan.copy()
        result = relax_plan(plan, self.omega, 2, self.boundaries, lambda_r=2.0, tau_b=0.7, enforce_ordering=False)
        np.testing.assert_allclose(result[2], plan[2], atol=1e-10)
        np.testing.assert_allclose(result.sum(axis=0), 0, atol=1e-10)
        with self.assertRaises(ValueError):
            relax_plan(plan, self.omega, 2, self.boundaries, lambda_r=2.0, tau_b=0.7)

    def test_two_words_fixed_by_equalities(self):
        plan = np.array([[0.6, -0.4], [-0.6, 0.4]])
        result = relax_plan(plan, self.omega, 0, np.array([0.0]), lambda_r=3.0, tau_b=1.0)
        np.testing.assert_allclose(result, plan, atol=1e-10)

    def test_zero_prominence_preserves_orthogonal_anchor(self):
        plan = np.array([[0.0, 1.0], [0.0, -0.6], [0.0, -0.4]])
        result = relax_plan(plan, self.omega, 0, np.zeros(2), lambda_r=3.0, tau_b=1.0)
        np.testing.assert_allclose(result[:, 0], 0, atol=1e-10)
        np.testing.assert_allclose(result[0], plan[0], atol=1e-10)
        np.testing.assert_allclose(result.sum(axis=0), 0, atol=1e-10)

    def test_tensor_wrapper_detaches_and_preserves_dtype(self):
        for dtype in (torch.float32, torch.float64):
            plan = torch.tensor(self.plan, dtype=dtype, requires_grad=True)
            result = relax_plan_tensor(plan, torch.tensor(self.omega, dtype=dtype), 0, torch.tensor(self.boundaries, dtype=dtype), lambda_r=2.0, tau_b=0.7)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, plan.device)
            self.assertFalse(result.requires_grad)
            torch.testing.assert_close(result[0], plan[0])
            torch.testing.assert_close(result.sum(dim=0), torch.zeros(2, dtype=dtype), atol=1e-06, rtol=0)

    def test_tensor_wrapper_rejects_invalid_rank_and_tolerance(self):
        with self.assertRaises(ValueError):
            relax_plan_tensor(torch.tensor(1.0), torch.ones(1), 0, torch.zeros(1), lambda_r=2.0, tau_b=0.7)
        for tolerance in (0, True, float('nan')):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                relax_plan_tensor(torch.tensor(self.plan), torch.tensor(self.omega), 0, torch.tensor(self.boundaries), lambda_r=2.0, tau_b=0.7, tolerance=tolerance)

    def test_invalid_inputs(self):
        for settings in ({'lambda_r': -1}, {'tau_b': float('nan')}, {'lambda_r': True}, {'tolerance': 0}, {'max_iterations': 0}, {'enforce_ordering': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.solve(**settings)
        with self.assertRaises(ValueError):
            relax_plan(self.plan + 1, self.omega, 0, self.boundaries, lambda_r=2.0, tau_b=0.7)
        with self.assertRaises(ValueError):
            relax_plan(self.plan, self.omega, True, self.boundaries, lambda_r=2.0, tau_b=0.7)
        with self.assertRaises(ValueError):
            relax_plan(self.plan, self.omega, 0, np.array([-1.0, 0.0, 0.0]), lambda_r=2.0, tau_b=0.7)
        with self.assertRaises(ValueError):
            relax_plan(self.plan, np.zeros(2), 0, self.boundaries, lambda_r=2.0, tau_b=0.7)
if __name__ == '__main__':
    unittest.main()
