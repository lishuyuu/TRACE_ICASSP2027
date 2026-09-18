from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from . import trace_core as core

def _dimension(value: int, name: str, minimum: int=1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an explicit integer >= {minimum}')
    return value

def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be an explicit positive finite scalar')
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)

def _dropout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(float(value))):
        raise TypeError('dropout must be an explicit finite scalar')
    if not 0 <= value < 1:
        raise ValueError('dropout must be in [0,1)')
    return float(value)

def _activation(name: str):
    if name == 'gelu':
        return F.gelu
    if name == 'relu':
        return F.relu
    if name == 'silu':
        return F.silu
    raise ValueError('activation must explicitly be gelu, relu or silu')

def sinusoidal_positions(length: int, width: int, *, base: float, reference: Tensor) -> Tensor:
    _dimension(length, 'position length')
    _dimension(width, 'position width')
    frequency_base = _positive(base, 'position base')
    if frequency_base <= 1:
        raise ValueError('position base must exceed one')
    core._finite(reference, 'position dtype/device reference')
    positions = torch.arange(length, dtype=reference.dtype, device=reference.device)
    frequencies = torch.exp(-math.log(frequency_base) * torch.arange(0, width, 2, dtype=reference.dtype, device=reference.device) / width)
    angles = positions[:, None] * frequencies[None, :]
    output = reference.new_zeros((length, width))
    output[:, 0::2] = torch.sin(angles)
    output[:, 1::2] = torch.cos(angles[:, :width // 2])
    return output

@dataclass(frozen=True)
class PlannerOutput:
    coordinates: Tensor
    b: Tensor
    u: Tensor
    valid_words: Tensor

class RelationalPlanner(nn.Module):

    def __init__(self, *, native_dim: int, speaker_dim: int, width: int, layers: int, heads: int, feedforward_dim: int, dropout: float, omega: Tensor, position_base: float, activation: str, norm_first: bool, norm_eps: float):
        super().__init__()
        self.native_dim = _dimension(native_dim, 'native_dim')
        self.speaker_dim = _dimension(speaker_dim, 'speaker_dim')
        self.width = _dimension(width, 'width')
        _dimension(layers, 'layers')
        _dimension(heads, 'heads')
        _dimension(feedforward_dim, 'feedforward_dim')
        probability = _dropout(dropout)
        if width % heads:
            raise ValueError('planner width must be divisible by attention heads')
        if type(norm_first) is not bool:
            raise TypeError('norm_first must be an explicit boolean')
        norm_epsilon = _positive(norm_eps, 'norm_eps')
        self.position_base = _positive(position_base, 'position_base')
        if self.position_base <= 1:
            raise ValueError('position_base must exceed one')
        activation_fn = _activation(activation)
        core._unit(omega)
        self.register_buffer('omega', omega.detach().clone())
        cues = omega.numel()
        options = {'device': omega.device, 'dtype': omega.dtype}
        self.native_projection = nn.Linear(native_dim, width, **options)
        self.speaker_projection = nn.Linear(speaker_dim, width, **options)
        self.target_role = nn.Embedding(2, width, **options)
        self.layers = nn.ModuleList((nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward_dim, dropout=probability, activation=activation_fn, batch_first=True, norm_first=norm_first, layer_norm_eps=norm_epsilon, **options) for _ in range(layers)))
        self.final_norm = nn.LayerNorm(width, eps=norm_epsilon, **options)
        self.b_head = nn.Linear(width, 1, **options)
        self.u_head = nn.Linear(width, cues, **options)

    def forward(self, native_word_states: Tensor, targets: Tensor, speaker_embedding: Tensor, valid_words: Tensor) -> PlannerOutput:
        native, speaker, valid, counts, positions = self._validated_inputs(native_word_states, speaker_embedding, valid_words)
        batch = native.shape[0]
        if not isinstance(targets, Tensor) or targets.dtype not in (torch.int32, torch.int64):
            raise TypeError('targets must be explicit hard integer word indices')
        if targets.shape != (batch,) or targets.device != valid.device:
            raise ValueError('one same-device hard target per endpoint is required')
        if not bool(((targets >= 0) & (targets < counts)).all()):
            raise ValueError('a target must index an actual word, never padding')
        role = positions[None, :] == targets[:, None]
        return self._forward_roles(native, speaker, valid, counts, role)

    def forward_target_mask(self, native_word_states: Tensor, target_mask: Tensor, speaker_embedding: Tensor, valid_words: Tensor) -> PlannerOutput:
        native, speaker, valid, counts, _ = self._validated_inputs(native_word_states, speaker_embedding, valid_words)
        batch, words = valid.shape
        mask = core._binary(target_mask, 'target mask', (batch, words)).to(device=valid.device, dtype=torch.bool)
        if bool((mask & ~valid).any()):
            raise ValueError('target mask may select actual words only')
        selected = mask.sum(dim=1)
        if bool((selected < 1).any()) or bool((selected >= counts).any()):
            raise ValueError('each target mask must select a nonempty proper word subset')
        for row in range(batch):
            indices = torch.nonzero(mask[row], as_tuple=False).flatten()
            if int(indices[-1] - indices[0] + 1) != int(indices.numel()):
                raise ValueError('multi-word target masks must be contiguous')
        return self._forward_roles(native, speaker, valid, counts, mask)

    def _validated_inputs(self, native_word_states: Tensor, speaker_embedding: Tensor, valid_words: Tensor):
        core._finite(native_word_states, 'supplied native word states', 3)
        core._finite(speaker_embedding, 'supplied speaker embedding', 2)
        core._same(native_word_states, speaker_embedding, self.omega)
        batch, words, native_dim = native_word_states.shape
        if native_dim != self.native_dim or speaker_embedding.shape != (batch, self.speaker_dim):
            raise ValueError('native/speaker dimensions do not match the explicit planner architecture')
        valid = core._binary(valid_words, 'valid words', (batch, words))
        if valid.device != native_word_states.device:
            raise ValueError('word mask must share the native-feature device')
        counts = valid.sum(dim=1)
        positions = torch.arange(words, device=valid.device)
        if bool((counts < 2).any()) or not torch.equal(valid, positions[None, :] < counts[:, None]):
            raise ValueError('valid words must be right-padded prefixes containing >=2 actual words')
        native = native_word_states.detach()
        speaker = speaker_embedding.detach()
        return (native, speaker, valid, counts, positions)

    def _forward_roles(self, native: Tensor, speaker: Tensor, valid: Tensor, counts: Tensor, target_mask: Tensor) -> PlannerOutput:
        batch, words, _ = native.shape
        role = target_mask.long()
        hidden = self.native_projection(native) + self.target_role(role) + self.speaker_projection(speaker)[:, None, :] + sinusoidal_positions(words, self.width, base=self.position_base, reference=native)[None]
        hidden = hidden.masked_fill(~valid[:, :, None], 0)
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=~valid)
            hidden = hidden.masked_fill(~valid[:, :, None], 0)
        hidden = self.final_norm(hidden).masked_fill(~valid[:, :, None], 0)
        b = self.b_head(hidden).squeeze(-1).masked_fill(~valid, 0)
        u = self.u_head(hidden).masked_fill(~valid[:, :, None], 0)
        plans = []
        for index, count in enumerate(counts.tolist()):
            plan = core.ordered_plan_mask(b[index, :count], u[index, :count], self.omega, target_mask[index, :count])
            plans.append(F.pad(plan, (0, 0, 0, words - count)))
        coordinates = torch.stack(plans)
        return PlannerOutput(coordinates, b, u, valid.detach().clone())

@dataclass(frozen=True)
class TargetFlagOutput:
    coordinates: Tensor
    raw_coordinates: Tensor
    valid_words: Tensor

class TargetFlagPlanner(nn.Module):

    def __init__(self, *, native_dim: int, speaker_dim: int, width: int, layers: int, heads: int, feedforward_dim: int, dropout: float, cues: int, position_base: float, activation: str, norm_first: bool, norm_eps: float):
        super().__init__()
        self.native_dim = _dimension(native_dim, 'native_dim')
        self.speaker_dim = _dimension(speaker_dim, 'speaker_dim')
        self.width = _dimension(width, 'width')
        self.cues = _dimension(cues, 'cues')
        _dimension(layers, 'layers')
        _dimension(heads, 'heads')
        _dimension(feedforward_dim, 'feedforward_dim')
        probability = _dropout(dropout)
        if width % heads:
            raise ValueError('target-flag width must be divisible by attention heads')
        if type(norm_first) is not bool:
            raise TypeError('norm_first must be an explicit boolean')
        epsilon = _positive(norm_eps, 'norm_eps')
        self.position_base = _positive(position_base, 'position_base')
        if self.position_base <= 1:
            raise ValueError('position_base must exceed one')
        activation_fn = _activation(activation)
        self.native_projection = nn.Linear(native_dim, width)
        self.speaker_projection = nn.Linear(speaker_dim, width)
        self.target_role = nn.Embedding(2, width)
        self.layers = nn.ModuleList((nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=feedforward_dim, dropout=probability, activation=activation_fn, batch_first=True, norm_first=norm_first, layer_norm_eps=epsilon) for _ in range(layers)))
        self.final_norm = nn.LayerNorm(width, eps=epsilon)
        self.coordinate_head = nn.Linear(width, cues)

    def forward(self, native_word_states: Tensor, targets: Tensor, speaker_embedding: Tensor, valid_words: Tensor) -> TargetFlagOutput:
        core._finite(native_word_states, 'supplied native word states', 3)
        core._finite(speaker_embedding, 'supplied speaker embedding', 2)
        batch, words, native_dim = native_word_states.shape
        if native_dim != self.native_dim or speaker_embedding.shape != (batch, self.speaker_dim):
            raise ValueError('native/speaker dimensions do not match the target-flag architecture')
        valid = core._binary(valid_words, 'valid words', (batch, words))
        if valid.device != native_word_states.device or speaker_embedding.device != native_word_states.device:
            raise ValueError('target-flag inputs and masks must share one device')
        counts = valid.sum(dim=1)
        positions = torch.arange(words, device=valid.device)
        if bool((counts < 2).any()) or not torch.equal(valid, positions[None, :] < counts[:, None]):
            raise ValueError('valid words must be right-padded prefixes containing >=2 actual words')
        if not isinstance(targets, Tensor) or targets.dtype not in (torch.int32, torch.int64):
            raise TypeError('targets must be explicit hard integer word indices')
        if targets.shape != (batch,) or targets.device != valid.device:
            raise ValueError('one same-device hard target per endpoint is required')
        if not bool(((targets >= 0) & (targets < counts)).all()):
            raise ValueError('a target must index an actual word, never padding')
        role = (positions[None, :] == targets[:, None]).long()
        native = native_word_states.detach()
        speaker = speaker_embedding.detach()
        hidden = self.native_projection(native) + self.target_role(role) + self.speaker_projection(speaker)[:, None, :] + sinusoidal_positions(words, self.width, base=self.position_base, reference=native)[None]
        hidden = hidden.masked_fill(~valid[:, :, None], 0)
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=~valid)
            hidden = hidden.masked_fill(~valid[:, :, None], 0)
        hidden = self.final_norm(hidden).masked_fill(~valid[:, :, None], 0)
        raw = self.coordinate_head(hidden).masked_fill(~valid[:, :, None], 0)
        mean = raw.sum(dim=1, keepdim=True) / counts.to(raw.dtype)[:, None, None]
        coordinates = (raw - mean).masked_fill(~valid[:, :, None], 0)
        core._finite(coordinates, 'target-flag centered coordinates', 3)
        return TargetFlagOutput(coordinates, raw, valid.detach().clone())

@dataclass(frozen=True)
class FrameCondition:
    native_frames: Tensor
    speaker: Tensor
    valid_frames: Tensor

class FrameResidualBackbone(nn.Module):

    def __init__(self, *, state_dim: int, native_dim: int, speaker_dim: int, width: int, layers: int, kernel_size: int, dropout: float, activation: str, norm_eps: float):
        super().__init__()
        self.state_dim = _dimension(state_dim, 'state_dim')
        self.native_dim = _dimension(native_dim, 'native_dim')
        self.speaker_dim = _dimension(speaker_dim, 'speaker_dim')
        self.width = _dimension(width, 'width')
        _dimension(layers, 'layers')
        _dimension(kernel_size, 'kernel_size')
        if kernel_size % 2 == 0:
            raise ValueError('temporal kernels must be odd for same-length symmetric padding')
        probability = _dropout(dropout)
        epsilon = _positive(norm_eps, 'norm_eps')
        self.activation = _activation(activation)
        self.state_projection = nn.Linear(state_dim, width)
        self.native_projection = nn.Linear(native_dim, width)
        self.speaker_projection = nn.Linear(speaker_dim, width)
        self.time_projection = nn.Linear(1, width)
        self.norms = nn.ModuleList((nn.LayerNorm(width, eps=epsilon) for _ in range(layers)))
        self.convolutions = nn.ModuleList((nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, stride=1) for _ in range(layers)))
        self.dropouts = nn.ModuleList((nn.Dropout(probability) for _ in range(layers)))
        self.final_norm = nn.LayerNorm(width, eps=epsilon)

    def forward(self, z: Tensor, time: Tensor | float, condition: FrameCondition) -> Tensor:
        core._finite(z, 'state z', 3)
        if z.shape[-1] != self.state_dim or not isinstance(condition, FrameCondition):
            raise ValueError('provide the configured state dimension and explicit FrameCondition')
        native = core._finite(condition.native_frames, 'shared native frames', 2)
        speaker = core._finite(condition.speaker, 'shared speaker', 1)
        core._same(z, native, speaker)
        if native.shape != (z.shape[1], self.native_dim) or speaker.shape != (self.speaker_dim,):
            raise ValueError('native frames/speaker must be shared, not endpoint-specific or time-resized')
        mask = core._flow_mask(z, condition.valid_frames)
        t = core._time(time, z)
        hidden = (self.state_projection(z) + self.native_projection(native.detach())[None] + self.speaker_projection(speaker.detach())[None, None] + self.time_projection(t.reshape(1))[None, None]) * mask
        for norm, convolution, dropout in zip(self.norms, self.convolutions, self.dropouts):
            branch = self.activation(norm(hidden)) * mask
            branch = convolution(branch.transpose(1, 2)).transpose(1, 2)
            hidden = (hidden + dropout(branch)) * mask
        result = self.final_norm(hidden) * mask
        core._finite(result, 'B_phi output', 3)
        return result

@dataclass(frozen=True)
class ObservedHuberLoss:
    loss: Tensor
    per_endpoint: Tensor
    observed_counts: Tensor
    normalization: str = 'observed coordinates per endpoint, then equal endpoint mean'

def observed_huber_loss(prediction: Tensor, target: Tensor, observed: Tensor, valid_words: Tensor, *, delta: float) -> ObservedHuberLoss:
    core._finite(prediction, 'planner prediction', 3)
    core._finite(target, 'oracle coordinate target', 3)
    core._same(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError('planner prediction and oracle target shapes must match exactly')
    mask = core._binary(observed, 'observation mask O', tuple(prediction.shape))
    valid = core._binary(valid_words, 'valid words', tuple(prediction.shape[:2]))
    if mask.device != prediction.device or valid.device != prediction.device:
        raise ValueError('explicit masks and predictions must share their device')
    if bool((mask & ~valid[:, :, None]).any()):
        raise ValueError('observed coordinates cannot reside on padded words')
    counts = mask.sum(dim=(1, 2))
    if bool((counts == 0).any()):
        raise ValueError('an endpoint has no observed supervision; it must not be silently dropped')
    threshold = _positive(delta, 'Huber delta')
    error = prediction - target.detach()
    absolute = error.abs()
    element = torch.where(absolute <= threshold, 0.5 * error.square(), threshold * (absolute - 0.5 * threshold))
    per_endpoint = (element * mask.to(element.dtype)).sum(dim=(1, 2)) / counts.to(element.dtype)
    core._finite(per_endpoint, 'observed Huber loss', 1)
    return ObservedHuberLoss(per_endpoint.mean(), per_endpoint, counts.detach())
