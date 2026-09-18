from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from . import counterfactual as t5
from .trace_core import _positive_number

class SignedRendererError(RuntimeError):
    pass

@dataclass(frozen=True)
class SignedRendererLoss:
    loss: Any
    fit: Any
    signed: Any
    signed_cosine: Any
    signed_margin: Any
    correct_mse: Any
    pair_source: Any
    pair_target: Any
    pair_cosine: Any
    pair_projection: Any
    pair_magnitude_ratio: Any
    predictions: Any

def _masked_pair_mse(prediction: Any, target: Any, valid_frames: Any) -> Any:
    torch = _torch_import()
    if prediction.shape != target.shape or prediction.ndim < 3 or valid_frames.ndim != 1 or (prediction.shape[-2] != valid_frames.numel()) or (valid_frames.dtype != torch.bool) or (valid_frames.device != prediction.device) or (not bool(valid_frames.any())):
        raise SignedRendererError('masked pair MSE geometry is invalid')
    mask = valid_frames.to(prediction.dtype).reshape(*(1,) * (prediction.ndim - 2), valid_frames.numel(), 1)
    denominator = valid_frames.sum().to(prediction.dtype) * prediction.shape[-1]
    return ((prediction - target.detach()).square() * mask).sum(dim=(-2, -1)) / denominator

def _different_command_pairs(commands: Any, valid_frames: Any) -> tuple[Any, Any]:
    torch = _torch_import()
    n = commands.shape[0]
    valid = valid_frames[None, None, :, None]
    equal = ((commands[:, None] == commands[None, :]) | ~valid).all(dim=(-2, -1))
    different = ~torch.eye(n, dtype=torch.bool, device=commands.device) & ~equal
    source, target = torch.nonzero(different, as_tuple=True)
    if source.numel() == 0:
        raise SignedRendererError('complete group has no distinct-command ordered pair')
    if torch.unique(torch.stack((source, target), dim=1), dim=0).shape[0] != source.numel():
        raise SignedRendererError('distinct-command pair index contains duplicates')
    return (source, target)

def signed_renderer_loss(adapter: Any, state: Any, time_value: float, frame_condition: Any, commands: Any, centered_target: Any, endpoints: Any, valid_frames: Any, *, lambda_cf: float, lambda_p: float, margin: float, cosine_epsilon: float) -> SignedRendererLoss:
    torch = _torch_import()
    lambda_cf = _positive_number(lambda_cf, 'lambda_cf', allow_zero=True)
    lambda_p = _positive_number(lambda_p, 'lambda_p', allow_zero=True)
    margin = _positive_number(margin, 'margin')
    cosine_epsilon = _positive_number(cosine_epsilon, 'cosine_epsilon')
    if state.ndim != 3 or centered_target.shape != state.shape or endpoints.shape != state.shape or (commands.shape != (*state.shape[:2], 2)) or (valid_frames.shape != (state.shape[1],)) or (valid_frames.dtype != torch.bool) or (state.shape[0] < 2):
        raise SignedRendererError('Signed loss received malformed complete-group tensors')
    try:
        predictions = t5.all_alternative_predictions(adapter, state, time_value, frame_condition, commands, valid_frames)
    except t5.RendererCandidateError as error:
        raise SignedRendererError(str(error)) from error
    n = state.shape[0]
    index = torch.arange(n, device=state.device)
    correct = predictions[index, index]
    correct_mse = _masked_pair_mse(correct, centered_target, valid_frames)
    fit = correct_mse.mean()
    pair_source, pair_target = _different_command_pairs(commands, valid_frames)
    predicted_delta = (predictions[pair_source, pair_target] - predictions[pair_source, pair_source])[:, valid_frames].reshape(pair_source.numel(), -1)
    true_delta = (endpoints[pair_target] - endpoints[pair_source])[:, valid_frames].reshape(pair_source.numel(), -1).detach()
    target_rms = true_delta.square().mean(dim=1).sqrt()
    usable = torch.isfinite(target_rms) & (target_rms > 0)
    if not bool(usable.any()):
        raise SignedRendererError('no finite nonzero distinct-command target pair')
    pair_source, pair_target = (pair_source[usable], pair_target[usable])
    predicted_delta = predicted_delta[usable]
    true_delta = true_delta[usable]
    target_rms = target_rms[usable]
    normalized_prediction = predicted_delta / target_rms[:, None]
    normalized_target = true_delta / target_rms[:, None]
    cross_mean = (normalized_prediction * normalized_target).mean(dim=1)
    predicted_square_mean = normalized_prediction.square().mean(dim=1)
    target_square_mean = normalized_target.square().mean(dim=1)
    smooth_cosine = cross_mean / torch.sqrt((predicted_square_mean + cosine_epsilon ** 2) * (target_square_mean + cosine_epsilon ** 2))
    ordinary_cosine = cross_mean / torch.sqrt(predicted_square_mean.clamp_min(torch.finfo(state.dtype).tiny) * target_square_mean)
    projection = cross_mean
    magnitude = torch.sqrt(predicted_square_mean / target_square_mean)
    signed_cosine = (1.0 - smooth_cosine).mean()
    relative_deficit = torch.relu(1 - projection / margin)
    signed_margin = torch.nn.functional.huber_loss(relative_deficit, torch.zeros_like(relative_deficit), reduction='mean', delta=1.0)
    signed = signed_cosine + lambda_p * signed_margin
    total = fit + lambda_cf * signed
    finite_values = (total, fit, signed, signed_cosine, signed_margin, ordinary_cosine, projection, magnitude)
    if not all((bool(torch.isfinite(value).all()) for value in finite_values)):
        raise SignedRendererError('Signed loss contains nonfinite values')
    return SignedRendererLoss(total, fit, signed, signed_cosine, signed_margin, correct_mse, pair_source, pair_target, ordinary_cosine, projection, magnitude, predictions)

def _torch_import() -> Any:
    try:
        import torch
    except Exception as error:
        raise SignedRendererError('PyTorch is required for signed refinement') from error
    return torch
