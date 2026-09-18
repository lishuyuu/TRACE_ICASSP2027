from __future__ import annotations
from typing import Any

class RendererCandidateError(RuntimeError):
    pass

def all_alternative_predictions(adapter: Any, state: Any, time_value: float, frame_condition: Any, commands: Any, valid_frames: Any) -> Any:
    torch = _torch_import()
    if state.ndim != 3 or commands.ndim != 3 or state.shape[:2] != commands.shape[:2] or (commands.shape[-1] != 2) or (state.shape[0] < 2):
        raise RendererCandidateError('all-alternative state/command geometry is invalid')
    if not hasattr(adapter, '_features') or not hasattr(adapter, '_project'):
        raise RendererCandidateError('adapter lacks the frozen zero-preserving projection seam')
    n, frames, features = state.shape
    hidden = adapter._features(state, time_value, frame_condition)
    if hidden.ndim != 3 or hidden.shape[:2] != (n, frames):
        raise RendererCandidateError('adapter feature geometry changed')
    width = hidden.shape[-1]
    cross_hidden = hidden[:, None].expand(n, n, frames, width).reshape(n * n, frames, width)
    cross_commands = commands[None].expand(n, n, frames, 2).reshape(n * n, frames, 2)
    prediction = adapter._project(cross_hidden, cross_commands, valid_frames).reshape(n, n, frames, features)
    if not bool(torch.isfinite(prediction).all()):
        raise RendererCandidateError('all-alternative residual prediction is non-finite')
    if bool(torch.count_nonzero(prediction[:, :, ~valid_frames])):
        raise RendererCandidateError('all-alternative residual leaked into the prompt prefix')
    return prediction

def _torch_import() -> Any:
    try:
        import torch
    except Exception as error:
        raise RendererCandidateError('PyTorch is required for renderer training') from error
    return torch
