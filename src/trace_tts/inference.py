import torch
from . import trace_core as core
from .gains import validate_gain
from .global_relaxation import relax_plan_tensor as relax_plan

def _frozen(module, name):
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f'{name} must be a loaded torch module')
    if module.training or any((p.requires_grad for p in module.parameters())):
        raise ValueError(f'{name} must be frozen and in evaluation mode')

def relax_training_plans(oracle, prediction, targets, omega, boundary_strengths, *, warmup, lambda_r, tau_b):
    selected = core.choose_adapter_plan(oracle, prediction, warmup=warmup)
    if selected.ndim != 3 or selected.shape[0] < 2 or selected.shape[-1] != 5:
        raise ValueError('complete groups require coordinates [endpoints,words,5]')
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64 or targets.shape != (selected.shape[0],) or (targets.unique().numel() != targets.numel()):
        raise ValueError('one distinct int64 target index per rendition is required')
    return torch.stack([relax_plan(q, omega, int(k), boundary_strengths, lambda_r=lambda_r, tau_b=tau_b) for q, k in zip(selected, targets)])

@torch.no_grad()
def generate_trace(backend, *, text, prompt, seed, target, planner, adapter, alpha, scales, lambda_r, tau_b):
    gain = validate_gain(alpha)
    if type(seed) is not int or type(target) is not int or target < 0:
        raise ValueError('seed and target must be integer indices')
    backend.freeze()
    native = backend.native(text, prompt, seed)
    core._finite(native.word_states, 'native word states', 2)
    words = native.word_states.shape[0]
    core._index(target, words, 'target')
    if gain == 0:
        return native.waveform
    _frozen(planner, 'planner')
    _frozen(adapter, 'adapter')
    valid_words = torch.ones(1, words, dtype=torch.bool, device=native.word_states.device)
    targets = torch.tensor([target], dtype=torch.int64, device=native.word_states.device)
    predicted = planner(native.word_states[None], targets, native.speaker[None], valid_words).coordinates[0]
    if predicted.shape != (words, 5):
        raise ValueError('TRACE requires five acoustic coordinates per word')
    relaxed = relax_plan(predicted, planner.omega, target, native.boundary_strengths, lambda_r=lambda_r, tau_b=tau_b)
    cues = core.physical_cues(gain * relaxed, scales, (0, 1, 2, 3, 4))
    clock = core.duration_clock(native.word_durations, cues[:, 4])
    prepared = backend.transport(native, clock)
    command = core.frame_command(cues, prepared.word_index, prepared.nucleus_phase, prepared.voicing_gate, prepared.valid_frames)

    def field(state, time):
        core._finite(state, 'solver state', 3)
        if state.shape[0] != 1:
            raise ValueError('one requested rendition per inference call is required')
        velocity = backend.velocity(state, time, prepared)
        correction = adapter(state, time, prepared.adapter_condition, command[None], prepared.valid_frames)
        return core.add_after_cfg(velocity, gain * correction, prepared.valid_frames)
    generated = backend.integrate(native, prepared, field)
    return backend.decode(generated, prepared)
