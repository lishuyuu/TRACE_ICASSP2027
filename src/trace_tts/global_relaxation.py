from __future__ import annotations
import math
import numbers
import numpy as np
import scipy.linalg
import torch
from torch import Tensor

def _nonnegative_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a finite nonnegative number')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return value

def boundary_weights(boundary_strengths: np.ndarray, *, tau_b: float) -> np.ndarray:
    tau_b = _nonnegative_scalar(tau_b, 'tau_b')
    strengths = np.asarray(boundary_strengths, dtype=np.float64)
    if strengths.ndim != 1 or not np.isfinite(strengths).all() or (strengths < 0).any():
        raise ValueError('boundary_strengths must be a finite nonnegative vector')
    with np.errstate(over='ignore'):
        return np.exp(-tau_b * strengths)

def _bounded_solution(hessian: np.ndarray, linear: np.ndarray, anchor: float, tolerance: float, max_iterations: int) -> np.ndarray:
    size = linear.size
    if anchor < 0:
        raise ValueError('a centered target-maximal plan requires nonnegative target prominence')
    if anchor == 0:
        return np.zeros(size, dtype=np.float64)
    current = np.full(size, -anchor / size, dtype=np.float64)
    active = np.zeros(size, dtype=bool)
    for _ in range(max_iterations):
        free = ~active
        if not free.any():
            raise RuntimeError('global relaxation has no free coordinate')
        free_hessian = hessian[np.ix_(free, free)]
        rhs = linear[free] - hessian[np.ix_(free, active)].sum(axis=1) * anchor
        solved = scipy.linalg.solve(free_hessian, np.column_stack((rhs, np.ones(free.sum()))), assume_a='pos')
        free_total = -anchor * (1 + active.sum())
        multiplier = (solved[:, 0].sum() - free_total) / solved[:, 1].sum()
        candidate = np.full(size, anchor, dtype=np.float64)
        candidate[free] = solved[:, 0] - multiplier * solved[:, 1]
        violating = free & (candidate > anchor + tolerance)
        if violating.any():
            direction = candidate - current
            indices = np.flatnonzero(violating)
            ratios = (anchor - current[indices]) / direction[indices]
            first = int(np.argmin(ratios))
            step = float(np.clip(ratios[first], 0.0, 1.0))
            current += step * direction
            hit = int(indices[first])
            current[hit] = anchor
            active[hit] = True
            continue
        current = candidate
        stationarity = hessian @ current - linear + multiplier
        release = active & (stationarity > tolerance)
        if release.any():
            indices = np.flatnonzero(release)
            active[indices[np.argmax(stationarity[indices])]] = False
            continue
        if abs(current.sum() + anchor) > 10 * tolerance or (current > anchor + 10 * tolerance).any() or np.max(np.abs(stationarity[free])) > 10 * tolerance:
            raise RuntimeError('global relaxation did not satisfy its optimality conditions')
        return current
    raise RuntimeError('global relaxation exceeded max_iterations')

def relax_plan(plan: np.ndarray, omega: np.ndarray, target: int, boundary_strengths: np.ndarray, *, lambda_r: float, tau_b: float, enforce_ordering: bool=True, tolerance: float=1e-09, max_iterations: int=1000) -> np.ndarray:
    lambda_r = _nonnegative_scalar(lambda_r, 'lambda_r')
    tolerance = _nonnegative_scalar(tolerance, 'tolerance')
    if tolerance == 0:
        raise ValueError('tolerance must be positive')
    if type(enforce_ordering) is not bool:
        raise ValueError('enforce_ordering must be boolean')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, numbers.Integral) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    source = np.asarray(plan, dtype=np.float64)
    direction = np.asarray(omega, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 1 or (not np.isfinite(source).all()):
        raise ValueError('plan must be a finite [words, cues] matrix with at least two words')
    if direction.shape != (source.shape[1],) or not np.isfinite(direction).all():
        raise ValueError('omega must have one finite entry per cue')
    norm = scipy.linalg.norm(direction)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('omega must be nonzero')
    direction = direction / norm
    if isinstance(target, bool) or not isinstance(target, numbers.Integral) or (not 0 <= target < source.shape[0]):
        raise ValueError('target must be a valid integer word index')
    weights = boundary_weights(boundary_strengths, tau_b=tau_b)
    if weights.shape != (source.shape[0] - 1,):
        raise ValueError('one boundary strength is required between consecutive words')
    scale = float(np.abs(source).max())
    normalized = source / scale if scale else source.copy()
    if np.abs(normalized.sum(axis=0)).max() > 10 * tolerance:
        raise ValueError('plan must be column-centered')
    prominence = normalized @ direction
    if enforce_ordering and (prominence > prominence[target] + 10 * tolerance).any():
        raise ValueError('plan must be target-maximal when enforce_ordering is enabled')
    if scale == 0 or lambda_r == 0 or (not weights.any()):
        return source.copy()
    words = source.shape[0]
    differences = np.zeros((words - 1, words), dtype=np.float64)
    indices = np.arange(words - 1)
    differences[indices, indices] = -1
    differences[indices, indices + 1] = 1
    laplacian = differences.T @ (weights[:, None] * differences)
    hessian = np.eye(words) + lambda_r * laplacian
    if not np.isfinite(hessian).all():
        raise ValueError('lambda_r produces a nonfinite quadratic system')
    matrix_scale = float(np.abs(hessian).max())
    hessian /= matrix_scale
    linear = normalized / matrix_scale
    equalities = np.zeros((2, words), dtype=np.float64)
    equalities[0] = 1
    equalities[1, target] = 1
    values = np.vstack((np.zeros(source.shape[1]), normalized[target]))
    solved = scipy.linalg.solve(hessian, np.column_stack((linear, equalities.T)), assume_a='pos')
    unconstrained = solved[:, :source.shape[1]]
    response = solved[:, source.shape[1]:]
    multipliers = scipy.linalg.solve(equalities @ response, equalities @ unconstrained - values, assume_a='pos')
    result = unconstrained - response @ multipliers
    if enforce_ordering:
        free = np.arange(words) != target
        anchor = float(prominence[target])
        bounded = _bounded_solution(hessian[np.ix_(free, free)], (linear @ direction)[free] - hessian[free, target] * anchor, anchor, tolerance, int(max_iterations))
        relaxed_prominence = np.empty(words, dtype=np.float64)
        relaxed_prominence[target] = anchor
        relaxed_prominence[free] = bounded
        result += np.outer(relaxed_prominence - result @ direction, direction)
    if not np.isfinite(result).all() or np.abs(result.sum(axis=0)).max() > 20 * tolerance or np.abs(result[target] - normalized[target]).max() > 20 * tolerance or (enforce_ordering and np.max(result @ direction - result[target] @ direction) > 20 * tolerance):
        raise RuntimeError('global relaxation did not satisfy its constraints')
    return result * scale

def relax_plan_tensor(plan: Tensor, omega: Tensor, target: int, boundary_strengths: Tensor, *, lambda_r: float, tau_b: float, enforce_ordering: bool=True, tolerance: float=1e-09, max_iterations: int=1000) -> Tensor:
    tensors = (plan, omega, boundary_strengths)
    if not all((isinstance(value, Tensor) for value in tensors)):
        raise ValueError('plan, omega and boundary_strengths must be tensors')
    if plan.ndim != 2:
        raise ValueError('plan must be a [words, cues] matrix')
    if plan.dtype not in (torch.float32, torch.float64):
        raise ValueError('plan must use float32 or float64')
    if any((value.dtype != plan.dtype or value.device != plan.device for value in tensors)):
        raise ValueError('all inputs must have the same floating dtype and device')
    tolerance = _nonnegative_scalar(tolerance, 'tolerance')
    if tolerance == 0:
        raise ValueError('tolerance must be positive')
    numerical_tolerance = max(tolerance, 4 * torch.finfo(plan.dtype).eps * max(1, plan.shape[0]))
    result = relax_plan(plan.detach().cpu().double().numpy(), omega.detach().cpu().double().numpy(), target, boundary_strengths.detach().cpu().double().numpy(), lambda_r=lambda_r, tau_b=tau_b, enforce_ordering=enforce_ordering, tolerance=numerical_tolerance, max_iterations=max_iterations)
    return torch.as_tensor(result, dtype=plan.dtype, device=plan.device)
