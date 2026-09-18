from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from . import trace_core as core

@dataclass(frozen=True)
class EqualityProjection:
    q: Tensor
    objective: float
    observed_error: float
    missing_regularizer: float
    centering_error: float
    stationarity_error: float

def project_no_ordering(z: Tensor, observed: Tensor, *, lambda_mis: float) -> EqualityProjection:
    core._finite(z, 'Z', 2)
    if z.shape[0] < 2:
        raise ValueError('zero-sum word coordinates require >=2 words')
    mask = core._binary(observed, 'O', tuple(z.shape))
    lam = core._positive_number(lambda_mis, 'lambda_mis')
    data = z.detach().to(device='cpu', dtype=torch.float64)
    obs = mask.detach().to(device='cpu', dtype=torch.float64)
    weights = obs + lam * (1 - obs)
    inverse = weights.reciprocal()
    if not bool(torch.isfinite(inverse).all()):
        raise ValueError('lambda_mis makes equality projection singular')
    center = obs * data
    lagrange = center.sum(dim=0) / inverse.sum(dim=0)
    q = center - inverse * lagrange
    observed_error = (obs * (q - data)).square().sum()
    missing_error = lam * ((1 - obs) * q).square().sum()
    stationarity = weights * (q - center) + lagrange
    if not bool(torch.isfinite(q).all()):
        raise RuntimeError('non-finite equality target')
    return EqualityProjection(q=q, objective=float(observed_error + missing_error), observed_error=float(observed_error), missing_regularizer=float(missing_error), centering_error=float(q.sum(dim=0).abs().max()), stationarity_error=float(stationarity.abs().max()))

def unordered_plan(u: Tensor) -> Tensor:
    core._finite(u, 'U', 2)
    if u.shape[0] < 2:
        raise ValueError('planner requires >=2 words')
    return u - u.mean(dim=0, keepdim=True)

def target_local_plan(plan: Tensor, target_mask: Tensor) -> Tensor:
    core._finite(plan, 'plan', 2)
    selected = core._binary(target_mask, 'target_mask', (plan.shape[0],))
    if selected.device != plan.device:
        raise ValueError('target mask and plan must share device')
    if not 0 < int(selected.sum()) < plan.shape[0]:
        raise ValueError('target mask must be a nonempty proper subset')
    return plan * selected[:, None].to(plan.dtype)
