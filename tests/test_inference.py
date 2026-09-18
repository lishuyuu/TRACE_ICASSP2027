import unittest
from types import SimpleNamespace
import torch
from trace_tts.backend_contract import NativeState, TransportedCondition
from trace_tts.inference import generate_trace, relax_training_plans

class SyntheticPlanner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('omega', torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
        self.register_buffer('q', torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0, 0.0]]))

    def forward(self, native, targets, speaker, valid):
        return SimpleNamespace(coordinates=self.q[None])

class SyntheticAdapter(torch.nn.Module):

    def forward(self, state, time, condition, command, valid):
        self.command = command.clone()
        return command.clone()

class SyntheticBackend:

    def __init__(self):
        self.waveform = object()
        self.calls = []

    def freeze(self):
        self.calls.append('freeze')

    def native(self, text, prompt, seed):
        self.calls.append('native')
        return NativeState(self.waveform, torch.ones(3, 2), torch.ones(2), torch.ones(3) / 3, torch.zeros(2), seed)

    def transport(self, native, clock):
        self.calls.append('transport')
        self.clock = clock
        return TransportedCondition(None, None, torch.arange(3), torch.ones(3) / 2, torch.ones(3), torch.ones(3, dtype=torch.bool), None)

    def velocity(self, state, time, prepared):
        return torch.ones_like(state)

    def integrate(self, native, prepared, field):
        self.calls.append('integrate')
        return field(torch.zeros(1, 3, 2), torch.tensor(0.5))

    def decode(self, state, prepared):
        self.calls.append('decode')
        return state

class InferenceTests(unittest.TestCase):

    def run_path(self, gain, backend=None, planner=None):
        backend = backend or SyntheticBackend()
        planner = planner or SyntheticPlanner().eval()
        adapter = SyntheticAdapter().eval()
        result = generate_trace(backend, text='synthetic', prompt=None, seed=1, target=0, planner=planner, adapter=adapter, alpha=gain, scales=torch.ones(5), lambda_r=0.0, tau_b=1.0)
        return (result, backend, adapter)

    def test_zero_gain_returns_exact_native_without_control_or_resynthesis(self):
        output, backend, adapter = self.run_path(0.0)
        self.assertIs(output, backend.waveform)
        self.assertEqual(backend.calls, ['freeze', 'native'])
        self.assertFalse(hasattr(adapter, 'command'))

    def test_single_gain_scales_both_plan_and_residual(self):
        a, _, adapter_a = self.run_path(1.0)
        b, _, adapter_b = self.run_path(2.0)
        torch.testing.assert_close(adapter_b.command, 2 * adapter_a.command)
        torch.testing.assert_close(b - 1, 4 * (a - 1))

    def test_train_mode_planner_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            self.run_path(1.0, planner=SyntheticPlanner())

    def test_training_plans_use_warmup_then_detached_prediction(self):
        planner = SyntheticPlanner()
        oracle = torch.stack([planner.q, planner.q.roll(1, 0)])
        predicted = (oracle * 2).requires_grad_()
        args = (oracle, predicted, torch.tensor([0, 1]), planner.omega, torch.zeros(2))
        warmup = relax_training_plans(*args, warmup=True, lambda_r=1.0, tau_b=1.0)
        main = relax_training_plans(*args, warmup=False, lambda_r=1.0, tau_b=1.0)
        torch.testing.assert_close(main, 2 * warmup)
        self.assertFalse(main.requires_grad)
if __name__ == '__main__':
    unittest.main()
