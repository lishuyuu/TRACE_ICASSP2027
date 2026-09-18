from __future__ import annotations
import math
import torch
from torch import nn

def validate_gain(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or (not math.isfinite(value)) or (value < 0):
        raise ValueError('gain must be a finite nonnegative scalar')
    return float(value)

def scale_plan(plan: torch.Tensor, gain: float) -> torch.Tensor:
    gain = validate_gain(gain)
    if not isinstance(plan, torch.Tensor) or plan.ndim != 2 or min(plan.shape) < 1 or (not plan.is_floating_point()) or (not bool(torch.isfinite(plan).all())) or plan.requires_grad:
        raise ValueError('expected a finite stopped floating-point plan [words,cues]')
    result = plan.clone() if gain == 1.0 else plan * gain
    if not bool(torch.isfinite(result).all()):
        raise ValueError('scaled plan is nonfinite')
    return result

class ResidualGain(nn.Module):

    def __init__(self, adapter: nn.Module, gain: float):
        super().__init__()
        if not isinstance(adapter, nn.Module):
            raise TypeError('a real adapter module is required')
        if any((p.requires_grad for p in adapter.parameters())):
            raise ValueError('residual gain is inference-only; freeze the loaded adapter')
        self.adapter = adapter
        self.gain = validate_gain(gain)

    def forward(self, state, time, condition, command, valid_frames):
        if self.gain == 0:
            return torch.zeros_like(state)
        raw = self.adapter(state, time, condition, command, valid_frames)
        if not isinstance(raw, torch.Tensor) or raw.shape != state.shape or (not bool(torch.isfinite(raw).all())):
            raise ValueError('learned adapter returned malformed/nonfinite residual')
        result = raw if self.gain == 1.0 else self.gain * raw
        if not bool(torch.isfinite(result).all()):
            raise ValueError('scaled adapter residual is nonfinite')
        return result
