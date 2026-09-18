from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def _finite(value: Tensor, name: str, ndim: int | None=None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f'{name} must be an explicit floating tensor')
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions')
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be nonempty and finite, including masked cells')
    return value

def _same(reference: Tensor, *others: Tensor) -> None:
    if any((x.dtype != reference.dtype or x.device != reference.device for x in others)):
        raise ValueError('floating inputs must explicitly share dtype and device')

def _binary(mask: Tensor, name: str, shape: tuple[int, ...] | None=None) -> Tensor:
    if not isinstance(mask, Tensor) or mask.is_complex():
        raise TypeError(f'{name} must be an explicit binary tensor')
    if shape is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError(f'{name} has the wrong shape')
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f'{name} must contain only 0/1, not missing values')
    return mask.bool()

def _positive_number(value: float, name: str, *, allow_zero: bool=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f'{name} must be an explicit finite scalar')
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be {('nonnegative' if allow_zero else 'positive')}")
    return value

def _index(value: int, size: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < size:
        raise ValueError(f'{name} must be an integer in [0,{size})')
    return value

def _unit(omega: Tensor) -> Tensor:
    _finite(omega, 'omega', 1)
    tolerance = 64 * torch.finfo(omega.dtype).eps
    if abs(float(torch.linalg.vector_norm(omega).detach()) - 1.0) > tolerance:
        raise ValueError('omega must already have unit norm; no epsilon-denominator fallback')
    return omega

def _time(value: float | Tensor, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim != 0 or not bool(torch.isfinite(result)) or (not 0 <= float(result.detach()) <= 1):
        raise ValueError('one shared scalar flow time in [0,1] is required per group')
    return result

def masked_center(values: Tensor, mask: Tensor, *, dim: int=-1) -> Tensor:
    _finite(values, 'values')
    support = _binary(mask, 'mask')
    if support.device != values.device:
        raise ValueError('mask and values must share a device')
    if not -values.ndim <= dim < values.ndim:
        raise ValueError('centering dimension is invalid')
    try:
        support = torch.broadcast_to(support, values.shape)
    except RuntimeError as error:
        raise ValueError('mask does not broadcast to values') from error
    numeric = support.to(values.dtype)
    count = numeric.sum(dim=dim, keepdim=True)
    mean = (numeric * values).sum(dim=dim, keepdim=True) / count.clamp_min(1)
    return numeric * (values - mean)

def standardize_descriptors(completed: Tensor, support: Tensor, scales: Tensor) -> Tensor:
    _finite(completed, 'completed descriptors', 3)
    n, words, cues = completed.shape
    if n < 2 or words < 2:
        raise ValueError('a relational group needs >=2 renditions and >=2 words')
    _binary(support, 'support', (words, cues))
    _finite(scales, 'training scales', 1)
    _same(completed, scales)
    if scales.shape != (cues,) or not bool((scales > 0).all()):
        raise ValueError('one strictly positive, supplied training scale per cue is required')
    return masked_center(completed, support, dim=1) / scales

def group_center(values: Tensor) -> tuple[Tensor, Tensor]:
    _finite(values, 'group values')
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError('provide all >=2 group endpoints, not a single rendition')
    mean = values.mean(dim=0)
    return (values - mean, mean)

def normalize_direction(direction: Tensor, *, min_norm: float) -> Tensor:
    _finite(direction, 'cue direction', 1)
    minimum = _positive_number(min_norm, 'min_norm')
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm.detach()) <= minimum:
        raise ValueError('cue direction is zero/nearzero; no fallback direction is defined')
    omega = direction / norm
    _unit(omega)
    return omega

def estimate_direction_from_pairs(groups: Sequence[Tensor], target_words: Sequence[Sequence[int]], valid_pairs_by_cue: Sequence[Sequence[tuple[int, int, int]]], *, min_norm: float) -> tuple[Tensor, Tensor]:
    if not groups or len(groups) != len(target_words):
        raise ValueError('groups and target-word registries must be nonempty and match')
    reference = _finite(groups[0], 'group', 3)
    cues = reference.shape[-1]
    if len(valid_pairs_by_cue) != cues:
        raise ValueError('one nonempty explicit J_q is required for every retained cue')
    for group, targets in zip(groups, target_words):
        _finite(group, 'group', 3)
        _same(reference, group)
        if group.shape[0] < 2 or group.shape[1] < 2 or group.shape[2] != cues:
            raise ValueError('group shape does not match retained cues/endpoints')
        if len(targets) != group.shape[0] or len(set(targets)) != len(targets):
            raise ValueError('each endpoint must have one distinct registered target word')
        for target in targets:
            _index(target, group.shape[1], 'target word')
    coordinates = []
    for cue, pairs in enumerate(valid_pairs_by_cue):
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError('J_q must be an explicit nonempty set without duplicate triples')
        contrasts = []
        for g, i, j in pairs:
            _index(g, len(groups), 'group index')
            phi = groups[g].detach()
            _index(i, phi.shape[0], 'endpoint_i')
            _index(j, phi.shape[0], 'endpoint_j')
            if i == j:
                raise ValueError('an alternative pair must contain distinct endpoints')
            ti, tj = (target_words[g][i], target_words[g][j])
            contrasts.append(phi[i, ti, cue] - phi[j, ti, cue] - (phi[i, tj, cue] - phi[j, tj, cue]))
        coordinates.append(torch.stack(contrasts).mean())
    direction = torch.stack(coordinates)
    return (direction, normalize_direction(direction, min_norm=min_norm))

@dataclass(frozen=True)
class ProjectionResult:
    q: Tensor
    objective: float
    observed_error: float
    missing_regularizer: float
    frobenius_distortion: float
    min_target_gap: float
    centering_error: float
    stationarity_error: float
    complementarity_error: float
    dual_weights: Tensor
    solver: str = 'weighted equality projection + scipy.optimize.nnls dual'

def project_target_maximum(z: Tensor, observed: Tensor, omega: Tensor, target: int, *, lambda_mis: float, tolerance: float=1e-08) -> ProjectionResult:
    from scipy.optimize import nnls
    _finite(z, 'Z', 2)
    words, cues = z.shape
    if words < 2:
        raise ValueError('ordering projection requires >=2 words')
    mask = _binary(observed, 'O', tuple(z.shape))
    _unit(omega)
    if omega.shape != (cues,):
        raise ValueError('omega cue count does not match Z')
    _index(target, words, 'target')
    weight_missing = _positive_number(lambda_mis, 'lambda_mis')
    tol = _positive_number(tolerance, 'tolerance')
    data = z.detach().cpu().double().numpy()
    obs = mask.detach().cpu().double().numpy()
    direction = omega.detach().cpu().double().numpy()
    weights = (obs + weight_missing * (1 - obs)).reshape(-1)
    inverse = 1.0 / weights
    if not np.isfinite(inverse).all():
        raise ValueError('lambda_mis makes the weighted projection numerically singular')
    center = (obs * data).reshape(-1)
    equality = np.tile(np.eye(cues), (1, words))
    weighted_equality = equality * inverse[None, :]
    equality_gram = weighted_equality @ equality.T
    weighted_et = inverse[:, None] * equality.T
    q0 = center - weighted_et @ np.linalg.solve(equality_gram, equality @ center)
    tangent_inverse = np.diag(inverse) - weighted_et @ np.linalg.solve(equality_gram, weighted_equality)
    inequality = np.zeros((words - 1, words * cues))
    for row, other in enumerate((w for w in range(words) if w != target)):
        inequality[row, target * cues:(target + 1) * cues] = direction
        inequality[row, other * cues:(other + 1) * cues] = -direction
    gram = inequality @ tangent_inverse @ inequality.T
    gram = (gram + gram.T) / 2
    design = np.linalg.cholesky(gram).T
    rhs = np.linalg.solve(design.T, -(inequality @ q0))
    multipliers, _ = nnls(design, rhs)
    solution = q0 + tangent_inverse @ inequality.T @ multipliers
    gradient = weights * (solution - center)
    equality_dual = np.linalg.solve(equality @ equality.T, equality @ (inequality.T @ multipliers - gradient))
    stationarity = gradient - inequality.T @ multipliers + equality.T @ equality_dual
    gaps = inequality @ solution
    centering_error = float(np.max(np.abs(equality @ solution)))
    stationarity_error = float(np.max(np.abs(stationarity)))
    complementarity_error = float(np.max(np.abs(multipliers * gaps)))
    scale = 1 + float(np.max(np.abs(solution))) + float(np.max(np.abs(gradient)))
    if not np.isfinite(solution).all() or float(gaps.min()) < -tol * scale or centering_error > tol * scale or (stationarity_error > tol * scale) or (complementarity_error > tol * scale * (1 + float(multipliers.max()))):
        raise RuntimeError('Eq3 solver failed its feasibility/KKT checks; no target returned')
    result = solution.reshape(words, cues)
    observed_error = float(np.sum((obs * (result - data)) ** 2))
    missing_penalty = weight_missing * float(np.sum(((1 - obs) * result) ** 2))
    return ProjectionResult(torch.from_numpy(result.copy()), observed_error + missing_penalty, observed_error, missing_penalty, float(np.linalg.norm(result - data)), float(gaps.min()), centering_error, stationarity_error, complementarity_error, torch.from_numpy(multipliers.copy()))

def ordered_plan(b: Tensor, u: Tensor, omega: Tensor, target: int) -> Tensor:
    if not isinstance(target, int) or isinstance(target, bool):
        raise TypeError('target must be one explicit integer word index')
    words = b.shape[0] if isinstance(b, Tensor) and b.ndim == 1 else 0
    _index(target, words, 'target')
    mask = torch.zeros(words, dtype=torch.bool, device=b.device)
    mask[target] = True
    return ordered_plan_mask(b, u, omega, mask)

def ordered_plan_mask(b: Tensor, u: Tensor, omega: Tensor, target_mask: Tensor) -> Tensor:
    _finite(b, 'b', 1)
    _finite(u, 'U', 2)
    _unit(omega)
    _same(b, u, omega)
    words, cues = u.shape
    if words < 2 or b.shape != (words,) or omega.shape != (cues,):
        raise ValueError('planner b[W], U[W,D], omega[D] must match with W>=2')
    selector = _binary(target_mask, 'target_mask', (words,)).to(device=b.device, dtype=torch.bool)
    count = int(selector.sum())
    if count < 1 or count >= words:
        raise ValueError('target mask must select a nonempty proper subset of words')
    g = torch.where(selector, torch.zeros_like(b), -F.softplus(b))
    centered_g, centered_u = (g - g.mean(), u - u.mean(dim=0))
    perpendicular = torch.eye(cues, dtype=u.dtype, device=u.device) - torch.outer(omega, omega)
    result = centered_g[:, None] * omega[None, :] + centered_u @ perpendicular
    _finite(result, 'ordered plan', 2)
    return result

def physical_cues(plan: Tensor, scales: Tensor, cue_indices: Sequence[int]) -> Tensor:
    _finite(plan, 'plan', 2)
    _finite(scales, 'scales', 1)
    _same(plan, scales)
    if len(cue_indices) != plan.shape[1] or scales.shape != (plan.shape[1],):
        raise ValueError('active cue map and scales must match plan columns')
    if len(set(cue_indices)) != len(cue_indices) or not bool((scales > 0).all()):
        raise ValueError('cue indices must be unique and scales positive')
    for index in cue_indices:
        _index(index, 5, 'physical cue index')
    indices = torch.tensor(cue_indices, dtype=torch.long, device=plan.device)
    return plan.new_zeros((plan.shape[0], 5)).index_copy(1, indices, plan * scales)

def duration_clock(native_composition: Tensor, duration_delta: Tensor) -> Tensor:
    _finite(native_composition, 'native composition', 1)
    _finite(duration_delta, 'physical duration delta', 1)
    _same(native_composition, duration_delta)
    if native_composition.shape != duration_delta.shape or not bool((native_composition > 0).all()):
        raise ValueError('native composition must be positive and match the word command')
    tol = 64 * torch.finfo(native_composition.dtype).eps
    if abs(float(native_composition.sum().detach()) - 1) > tol:
        raise ValueError('native composition must already sum to one')
    if not bool(torch.count_nonzero(duration_delta)):
        return native_composition.clone()
    result = torch.softmax(torch.log(native_composition) + duration_delta, dim=0)
    if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError('clock numerically lost positive support; no hidden duration floor')
    return result

def frame_command(cues: Tensor, word_index: Tensor, nucleus_phase: Tensor, voicing_gate: Tensor, valid_frames: Tensor) -> Tensor:
    _finite(cues, 'physical cues', 2)
    _finite(nucleus_phase, 'nucleus phase', 1)
    _finite(voicing_gate, 'voicing gate', 1)
    _same(cues, nucleus_phase, voicing_gate)
    count = nucleus_phase.numel()
    valid = _binary(valid_frames, 'valid_frames', (count,))
    if cues.shape[1] != 5 or voicing_gate.shape != (count,):
        raise ValueError('physical cues[W,5] and scalar per-frame geometry are required')
    if not isinstance(word_index, Tensor) or word_index.dtype not in (torch.int32, torch.int64):
        raise TypeError('word_index must explicitly contain integer word indices')
    if word_index.shape != (count,) or any((x.device != cues.device for x in (word_index, valid))):
        raise ValueError('shared geometry shape/device mismatch')
    if not bool(((nucleus_phase >= 0) & (nucleus_phase <= 1)).all()):
        raise ValueError('nucleus phase must be supplied in [0,1]')
    if not bool(((voicing_gate >= 0) & (voicing_gate <= 1)).all()):
        raise ValueError('voicing gate must be supplied in [0,1], without threshold inference')
    if not bool(((word_index[valid] >= 0) & (word_index[valid] < cues.shape[0])).all()):
        raise ValueError('valid frames need in-range word ownership')
    safe_words = torch.where(valid, word_index, torch.zeros_like(word_index)).long()
    positions = 2 * nucleus_phase - 1
    legendre = torch.stack((torch.ones_like(positions), positions, (3 * positions.square() - 1) / 2), dim=-1)
    pitch = (cues[safe_words, :3] * legendre).sum(dim=-1) * voicing_gate
    energy = cues[safe_words, 3]
    return torch.stack((pitch, energy), dim=-1) * valid[:, None].to(cues.dtype)

def flow_path(endpoints: Tensor, shared_noise: Tensor, time: float | Tensor, *, sigma_min: float) -> tuple[Tensor, Tensor]:
    _finite(endpoints, 'transported endpoints', 3)
    _finite(shared_noise, 'shared noise', 2)
    _same(endpoints, shared_noise)
    if endpoints.shape[0] < 2 or shared_noise.shape != endpoints.shape[1:]:
        raise ValueError('complete group endpoints must share one same-shape noise tensor')
    sigma = _positive_number(sigma_min, 'sigma_min', allow_zero=True)
    if sigma > 1:
        raise ValueError('sigma_min must be in [0,1]')
    t = _time(time, endpoints)
    alpha = 1 - (1 - sigma) * t
    return (alpha * shared_noise + t * endpoints, endpoints - (1 - sigma) * shared_noise)

def _flow_mask(values: Tensor, valid_frames: Tensor) -> Tensor:
    _finite(values, 'flow tensor', 3)
    valid = _binary(valid_frames, 'shared valid_frames', (values.shape[1],))
    if valid.device != values.device or not bool(valid.any()):
        raise ValueError('a nonempty valid clock mask on the same device is required')
    return valid[None, :, None].to(values.dtype)

def center_flow_residuals(true_velocity: Tensor, frozen_cfg_velocity: Tensor, valid_frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mask = _flow_mask(true_velocity, valid_frames)
    _finite(frozen_cfg_velocity, 'frozen CFG velocity', 3)
    _same(true_velocity, frozen_cfg_velocity)
    if true_velocity.shape != frozen_cfg_velocity.shape or true_velocity.shape[0] < 2:
        raise ValueError('all complete group endpoints and their CFG velocities must match')
    residual = (true_velocity.detach() - frozen_cfg_velocity.detach()) * mask
    centered, mean = group_center(residual)
    return (residual, centered, mean)

def choose_adapter_plan(oracle: Tensor, prediction: Tensor, *, warmup: bool) -> Tensor:
    _finite(oracle, 'oracle plan')
    _finite(prediction, 'predicted plan')
    _same(oracle, prediction)
    if type(warmup) is not bool or oracle.shape != prediction.shape:
        raise ValueError('warmup must be explicit and oracle/predicted plans must match')
    return (oracle if warmup else prediction).detach()

class ZeroPreservingResidual(nn.Module):

    def __init__(self, base_branch: nn.Module, *, width: int, output_dim: int):
        super().__init__()
        if not isinstance(base_branch, nn.Module) or type(width) is not int or width < 1:
            raise ValueError('provide B_phi and a positive adapter width')
        if type(output_dim) is not int or output_dim < 1:
            raise ValueError('output dimension must be positive')
        self.base_branch = base_branch
        self.width, self.output_dim = (width, output_dim)
        self.gate = nn.Linear(2, width, bias=False)
        self.output = nn.Linear(width, output_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def _features(self, z: Tensor, time: Tensor | float, condition) -> Tensor:
        _finite(z, 'state z', 3)
        t = _time(time, z)
        features = self.base_branch(z, t, condition)
        _finite(features, 'B_phi features', 3)
        _same(z, features)
        if features.shape != (*z.shape[:-1], self.width):
            raise ValueError('B_phi must return one width-dimensional feature per state frame')
        return features

    def _project(self, features: Tensor, command: Tensor, valid_frames: Tensor) -> Tensor:
        _finite(command, 'frame command', 3)
        _same(features, command)
        if command.shape != (*features.shape[:-1], 2):
            raise ValueError("command[N,T,2] must match B_phi's exact state clock")
        mask = _flow_mask(features, valid_frames)
        return self.output(features * torch.tanh(self.gate(command))) * mask

    def forward(self, z, time, condition, command, valid_frames):
        return self._project(self._features(z, time, condition), command, valid_frames)

    def forward_pair(self, z, time, condition, command, swapped_command, valid_frames):
        features = self._features(z, time, condition)
        return (self._project(features, command, valid_frames), self._project(features, swapped_command, valid_frames))

def add_after_cfg(frozen_cfg_velocity: Tensor, residual: Tensor, valid_frames: Tensor) -> Tensor:
    _finite(frozen_cfg_velocity, 'CFG velocity', 3)
    _finite(residual, 'adapter residual', 3)
    _same(frozen_cfg_velocity, residual)
    if frozen_cfg_velocity.shape != residual.shape:
        raise ValueError('residual must be added to the same CFG field shape')
    return frozen_cfg_velocity + _flow_mask(residual, valid_frames) * residual

def masked_mse(prediction: Tensor, target: Tensor, valid_frames: Tensor) -> Tensor:
    mask = _flow_mask(prediction, valid_frames)
    _finite(target, 'stopped residual target', 3)
    _same(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError('prediction and residual target must have identical shapes')
    denominator = mask.sum() * prediction.shape[-1]
    return ((prediction - target.detach()).square() * mask).sum(dim=(1, 2)) / denominator

@dataclass(frozen=True)
class GCRALoss:
    loss: Tensor
    correct_mse: Tensor
    swapped_mse: Tensor
    swap_hinge: Tensor
    swap_weight: Tensor
    weighted_swap: Tensor
    equal_command: Tensor
    no_swap_gradient_expected: Tensor

def gcra_loss(correct_prediction: Tensor, swapped_prediction: Tensor, target: Tensor, valid_frames: Tensor, correct_command: Tensor, swapped_command: Tensor, *, time: float | Tensor, lambda_swap: float, margin: float, gamma: float) -> GCRALoss:
    correct = masked_mse(correct_prediction, target, valid_frames)
    swapped = masked_mse(swapped_prediction, target, valid_frames)
    if correct_prediction.shape != swapped_prediction.shape or len(correct) < 2:
        raise ValueError('same-state predictions for all >=2 group endpoints are required')
    _same(correct_prediction, swapped_prediction)
    _finite(correct_command, 'correct command', 3)
    _finite(swapped_command, 'swapped command', 3)
    _same(correct_prediction, correct_command, swapped_command)
    expected = (*correct_prediction.shape[:-1], 2)
    if correct_command.shape != expected or swapped_command.shape != expected:
        raise ValueError("both commands must share the residual group's exact [N,T,2] clock")
    lam = _positive_number(lambda_swap, 'lambda_swap', allow_zero=True)
    mu = _positive_number(margin, 'margin', allow_zero=True)
    exponent = _positive_number(gamma, 'gamma')
    t = _time(time, correct_prediction)
    valid = _binary(valid_frames, 'valid_frames', (correct_prediction.shape[1],))
    equal = ((correct_command == swapped_command) | ~valid[None, :, None]).all(dim=(1, 2))
    if bool(equal.any()) and (not torch.equal(correct_prediction[equal][:, valid], swapped_prediction[equal][:, valid])):
        raise ValueError('equal commands produced different valid predictions: same-state swap contract violated')
    hinge = F.relu(mu + correct - swapped)
    weight = lam * (1 - t).pow(exponent)
    weighted = weight * hinge
    return GCRALoss((correct + weighted).mean(), correct, swapped, hinge, weight, weighted, equal.detach(), equal.detach().clone())

def sample_alternatives(group_size: int, *, generator: torch.Generator) -> Tensor:
    if type(group_size) is not int or group_size < 2 or (not isinstance(generator, torch.Generator)):
        raise ValueError('provide a complete group size>=2 and an explicit random generator')
    draw = torch.randint(group_size - 1, (group_size,), generator=generator, device=generator.device)
    indices = torch.arange(group_size, device=draw.device)
    return draw + (draw >= indices).long()
