from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
BUNDLE_SCHEMA = 'trace-mp-cae-dpe-style-bundle-v2'
METHOD_ID = 'mp_cae_dpe_style'
SYSTEM_ID = 'mp_cae_dpe_adapt'
EXPECTED_CUES = ('target_log_f0', 'target_log_rms')
ARRAY_NAMES = {'feature_mean', 'components', 'latent_scale', 'target_mean', 'target_scale', 'probe'}

class MPCAEContractError(ValueError):
    pass
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

def _finite_array(value: ArrayLike, name: str, *, ndim: int) -> FloatArray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.number):
        raise MPCAEContractError(f'{name} must be a numeric array with ndim={ndim}')
    result = np.asarray(array, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise MPCAEContractError(f'{name} must be nonempty and finite')
    return result

def _text_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple((str(value) for value in values))
    if not result or any((not value for value in result)):
        raise MPCAEContractError(f'{name} must contain nonempty strings')
    return result

def _group_weights(group_ids: Sequence[str]) -> FloatArray:
    groups = _text_tuple(group_ids, 'group_ids')
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    weights /= weights.sum()
    return weights

@dataclass(frozen=True)
class MPCAETrainingSet:
    features: FloatArray
    acoustic_targets: FloatArray
    fit_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    group_weights: FloatArray
    excluded_fit_ids: tuple[str, ...]

@dataclass(frozen=True)
class MPCAEModel:
    feature_mean: FloatArray
    components: FloatArray
    latent_scale: FloatArray
    target_mean: FloatArray
    target_scale: FloatArray
    probe: FloatArray
    ridge: float
    trust_radius: float
    cue_names: tuple[str, ...]
    fit_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    explained_variance: float
    hook: str
    method_id: str = METHOD_ID
    system_id: str = SYSTEM_ID

    @property
    def pca_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.components.shape[1])

def training_set_from_observations(features: ArrayLike, acoustic_targets: ArrayLike, *, fit_ids: Sequence[str], group_ids: Sequence[str], excluded_fit_ids: Sequence[str]=()) -> MPCAETrainingSet:
    x = _finite_array(features, 'features', ndim=2)
    y = _finite_array(acoustic_targets, 'acoustic_targets', ndim=2)
    ids = _text_tuple(fit_ids, 'fit_ids')
    groups = _text_tuple(group_ids, 'group_ids')
    excluded = tuple((str(value) for value in excluded_fit_ids))
    if x.shape[0] != y.shape[0] or x.shape[0] != len(ids) or len(ids) != len(groups):
        raise MPCAEContractError('features, targets, fit_ids, and group_ids must align')
    if len(ids) != len(set(ids)):
        raise MPCAEContractError('fit_ids must be unique')
    if any((not value for value in excluded)) or len(excluded) != len(set(excluded)):
        raise MPCAEContractError('excluded_fit_ids must be unique nonempty strings')
    if set(ids) & set(excluded):
        raise MPCAEContractError('retained and excluded fit identities overlap')
    if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
        raise MPCAEContractError('fitting needs at least two rows and nonzero widths')
    if len(set(groups)) < 2:
        raise MPCAEContractError('fitting needs at least two TRAIN groups')
    return MPCAETrainingSet(features=x, acoustic_targets=y, fit_ids=ids, group_ids=groups, group_weights=_group_weights(groups), excluded_fit_ids=excluded)

def pca_variance_profile(features: ArrayLike, group_ids: Sequence[str], candidates: Sequence[int]) -> dict[int, float]:
    x = _finite_array(features, 'features', ndim=2)
    groups = _text_tuple(group_ids, 'group_ids')
    if len(groups) != x.shape[0]:
        raise MPCAEContractError('group_ids must align with PCA rows')
    dims = tuple((int(value) for value in candidates))
    if not dims or any((value <= 0 or value > min(x.shape) for value in dims)) or tuple(sorted(set(dims))) != dims:
        raise MPCAEContractError('PCA candidates must be increasing unique feasible dimensions')
    weights = _group_weights(groups)
    mean = np.sum(weights[:, None] * x, axis=0)
    weighted = np.sqrt(weights[:, None]) * (x - mean)
    singular = np.linalg.svd(weighted, compute_uv=False)
    total = float(np.sum(singular ** 2))
    if not np.isfinite(total) or total <= np.finfo(np.float64).eps:
        raise MPCAEContractError('TRAIN activation variance is zero or non-finite')
    return {dim: float(np.sum(singular[:dim] ** 2) / total) for dim in dims}

def _validate_model(model: MPCAEModel) -> None:
    if not isinstance(model, MPCAEModel):
        raise MPCAEContractError('model must be an MPCAEModel')
    mean = _finite_array(model.feature_mean, 'model.feature_mean', ndim=1)
    components = _finite_array(model.components, 'model.components', ndim=2)
    latent_scale = _finite_array(model.latent_scale, 'model.latent_scale', ndim=1)
    target_mean = _finite_array(model.target_mean, 'model.target_mean', ndim=1)
    target_scale = _finite_array(model.target_scale, 'model.target_scale', ndim=1)
    probe = _finite_array(model.probe, 'model.probe', ndim=2)
    if components.shape[1] != mean.size or components.shape[0] != latent_scale.size:
        raise MPCAEContractError('model PCA geometry is inconsistent')
    if probe.shape != (components.shape[0], target_mean.size):
        raise MPCAEContractError('model probe geometry is inconsistent')
    if target_scale.shape != target_mean.shape or np.any(target_scale <= 0):
        raise MPCAEContractError('model target scales must be positive and aligned')
    if np.any(latent_scale <= 0):
        raise MPCAEContractError('model latent scales must be positive')
    gram = components @ components.T
    if not np.allclose(gram, np.eye(components.shape[0]), rtol=0.0, atol=1e-08):
        raise MPCAEContractError('PCA components must remain row-orthonormal')
    if not isinstance(model.hook, str) or not model.hook or model.method_id != METHOD_ID or (model.system_id != SYSTEM_ID) or (tuple(model.cue_names) != EXPECTED_CUES):
        raise MPCAEContractError('model identity or cue ordering changed')
    if not np.isfinite(model.ridge) or model.ridge <= 0 or (not np.isfinite(model.trust_radius)) or (model.trust_radius <= 0) or (not 0.0 < model.explained_variance <= 1.0):
        raise MPCAEContractError('model scalar configuration is invalid')
    if len(model.fit_ids) != len(model.group_ids) or not model.fit_ids:
        raise MPCAEContractError('model TRAIN provenance is incomplete')

def fit_mp_cae_model(features: ArrayLike, acoustic_targets: ArrayLike, *, fit_ids: Sequence[str], group_ids: Sequence[str], pca_dim: int, ridge: float, trust_radius: float, hook: str, cue_names: Sequence[str]=EXPECTED_CUES) -> MPCAEModel:
    x = _finite_array(features, 'features', ndim=2)
    y = _finite_array(acoustic_targets, 'acoustic_targets', ndim=2)
    ids = _text_tuple(fit_ids, 'fit_ids')
    groups = _text_tuple(group_ids, 'group_ids')
    cues = _text_tuple(cue_names, 'cue_names')
    if x.shape[0] != y.shape[0] or x.shape[0] != len(ids) or len(ids) != len(groups) or (y.shape[1] != len(cues)):
        raise MPCAEContractError('fit arrays and identities do not align')
    if len(ids) != len(set(ids)):
        raise MPCAEContractError('fit_ids must remain unique')
    if type(pca_dim) is not int or pca_dim <= 0 or pca_dim > min(x.shape):
        raise MPCAEContractError('pca_dim is infeasible')
    if not np.isfinite(ridge) or ridge <= 0:
        raise MPCAEContractError('ridge must be finite and positive')
    if not np.isfinite(trust_radius) or trust_radius <= 0:
        raise MPCAEContractError('trust_radius must be finite and positive')
    weights = _group_weights(groups)
    feature_mean = np.sum(weights[:, None] * x, axis=0)
    centered = x - feature_mean
    _u, singular, vt = np.linalg.svd(np.sqrt(weights[:, None]) * centered, full_matrices=False)
    total = float(np.sum(singular ** 2))
    if total <= np.finfo(np.float64).eps or singular[pca_dim - 1] <= np.finfo(np.float64).eps:
        raise MPCAEContractError('TRAIN activation matrix lacks the requested PCA rank')
    components = vt[:pca_dim].copy()
    explained = float(np.sum(singular[:pca_dim] ** 2) / total)
    latent = centered @ components.T
    latent_scale = np.sqrt(np.sum(weights[:, None] * latent ** 2, axis=0))
    if np.any(latent_scale <= np.finfo(np.float64).eps):
        raise MPCAEContractError('a retained PCA coordinate has zero TRAIN scale')
    target_mean = np.sum(weights[:, None] * y, axis=0)
    target_centered = y - target_mean
    target_scale = np.sqrt(np.sum(weights[:, None] * target_centered ** 2, axis=0))
    if np.any(target_scale <= np.finfo(np.float64).eps):
        raise MPCAEContractError('every acoustic cue must vary across TRAIN groups')
    standardized = target_centered / target_scale
    regression_weights = weights * x.shape[0]
    weighted_latent = latent * regression_weights[:, None]
    gram = latent.T @ weighted_latent
    rhs = weighted_latent.T @ standardized
    probe = np.linalg.solve(gram + float(ridge) * np.eye(pca_dim), rhs)
    if not np.all(np.isfinite(probe)) or np.linalg.matrix_rank(probe) < y.shape[1]:
        raise MPCAEContractError('joint acoustic probe is non-finite or cue-rank deficient')
    model = MPCAEModel(feature_mean=feature_mean, components=components, latent_scale=latent_scale, target_mean=target_mean, target_scale=target_scale, probe=probe, ridge=float(ridge), trust_radius=float(trust_radius), cue_names=cues, fit_ids=ids, group_ids=groups, explained_variance=explained, hook=hook)
    _validate_model(model)
    return model

def predict_target_cues(features: ArrayLike, model: MPCAEModel) -> FloatArray:
    _validate_model(model)
    x = _finite_array(features, 'features', ndim=2)
    if x.shape[1] != model.feature_dim:
        raise MPCAEContractError('feature width differs from the fitted bundle')
    latent = (x - model.feature_mean) @ model.components.T
    return model.target_mean + latent @ model.probe * model.target_scale

def _target_words(target_words: Iterable[int], word_count: int) -> tuple[int, ...]:
    raw = tuple(target_words)
    if not raw or any((type(value) is not int for value in raw)) or tuple(sorted(set(raw))) != raw or (raw[0] < 0) or (raw[-1] >= word_count) or (raw[-1] - raw[0] + 1 != len(raw)) or (len(raw) >= word_count):
        raise MPCAEContractError('target_words must be a contiguous proper word span')
    return raw

def _target_pooled_mu_feature(mu: FloatArray, owners: IntArray, targets: tuple[int, ...]) -> tuple[FloatArray, NDArray[np.bool_]]:
    valid = owners >= 0
    if not np.any(valid):
        raise MPCAEContractError('word_index contains no valid target-speech frames')
    word_count = int(owners[valid].max()) + 1
    if set(owners[valid].tolist()) != set(range(word_count)):
        raise MPCAEContractError('every target-speech word must own at least one frame')
    if targets != _target_words(targets, word_count):
        raise MPCAEContractError('target span differs from validated word geometry')
    target_set = np.asarray(targets, dtype=np.int64)
    target_mask = valid & np.isin(owners, target_set)
    target_pooled = np.stack([mu[:, owners == word].mean(axis=1) for word in targets]).mean(axis=0)
    return (target_pooled, target_mask)

def apply_mp_cae_target_mu(mu: ArrayLike, word_index: ArrayLike, target_words: Sequence[int], model: MPCAEModel, *, pitch_semitones: float, energy_db: float, trust_radius: float) -> tuple[FloatArray, dict[str, Any]]:
    _validate_model(model)
    source = _finite_array(mu, 'mu', ndim=2)
    owners_raw = np.asarray(word_index)
    if owners_raw.ndim != 1 or not np.issubdtype(owners_raw.dtype, np.integer):
        raise MPCAEContractError('word_index must be a one-dimensional integer array')
    owners = owners_raw.astype(np.int64, copy=False)
    if source.shape != (model.feature_dim, owners.size):
        raise MPCAEContractError('mu must have [fitted feature, word_index frame] geometry')
    if any((not np.isfinite(value) for value in (pitch_semitones, energy_db, trust_radius))):
        raise MPCAEContractError('physical controls and trust_radius must be finite')
    if trust_radius <= 0 or not np.isclose(trust_radius, model.trust_radius, rtol=0.0, atol=1e-12):
        raise MPCAEContractError('runtime trust_radius differs from the frozen bundle')
    valid = owners >= 0
    word_count = int(owners[valid].max()) + 1 if np.any(valid) else 0
    targets = _target_words(target_words, word_count)
    target_feature, target_mask = _target_pooled_mu_feature(source, owners, targets)
    requested_log = np.asarray([float(pitch_semitones) * np.log(2.0) / 12.0, float(energy_db) * np.log(10.0) / 20.0], dtype=np.float64)
    request_standardized = requested_log / model.target_scale
    latent_before = (target_feature - model.feature_mean) @ model.components.T
    predicted_before = model.target_mean + latent_before @ model.probe * model.target_scale
    effective = model.latent_scale[:, None] * model.probe
    if np.linalg.matrix_rank(effective) < len(EXPECTED_CUES):
        raise MPCAEContractError('frozen probe cannot jointly realize both acoustic cues')
    standardized_delta = np.linalg.pinv(effective.T, rcond=1e-10) @ request_standardized
    requested_norm = float(np.linalg.norm(standardized_delta))
    clipped = requested_norm > trust_radius
    if clipped:
        standardized_delta *= float(trust_radius / requested_norm)
    applied_norm = float(np.linalg.norm(standardized_delta))
    latent_delta = standardized_delta * model.latent_scale
    feature_delta = model.components.T @ latent_delta
    edited = source.copy()
    if np.any(feature_delta != 0.0):
        edited[:, target_mask] += feature_delta[:, None]
    edited[:, ~target_mask] = source[:, ~target_mask]
    if not np.all(np.isfinite(edited)):
        raise MPCAEContractError('counterfactual edit produced non-finite CFM values')
    if not np.array_equal(edited[:, ~target_mask], source[:, ~target_mask]):
        raise MPCAEContractError('counterfactual edit changed a non-target frame')
    achieved_standardized = latent_delta @ model.probe
    achieved_log = achieved_standardized * model.target_scale
    predicted_after = predicted_before + achieved_log
    diagnostics = {'method_id': METHOD_ID, 'control_hook': model.hook, 'pca_dim': model.pca_dim, 'trust_radius': float(trust_radius), 'requested_latent_norm': requested_norm, 'applied_latent_norm': applied_norm, 'trust_clipped': bool(clipped), 'requested_pitch_semitones': float(pitch_semitones), 'requested_energy_db': float(energy_db), 'achieved_probe_pitch_semitones': float(achieved_log[0] * 12.0 / np.log(2.0)), 'achieved_probe_energy_db': float(achieved_log[1] * 20.0 / np.log(10.0)), 'predicted_target_log_cues_before': predicted_before.tolist(), 'predicted_target_log_cues_after': predicted_after.tolist(), 'target_frame_count': int(np.count_nonzero(target_mask)), 'non_target_frame_count': int(np.count_nonzero(~target_mask)), 'non_target_exact': True, 'minimum_norm_joint_solution': True}
    return (edited, diagnostics)
