import numpy as np

class RelationMeasurementError(RuntimeError):
    pass

def endpoint_prominence(raw, observed, support, scales, direction):
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed)
    support = np.asarray(support)
    scales = np.asarray(scales, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise RelationMeasurementError('Expected word-by-cue endpoint descriptors')
    if observed.dtype != np.bool_ or support.dtype != np.bool_:
        raise RelationMeasurementError('Observation and support masks must be boolean')
    if observed.shape != raw.shape or support.shape != raw.shape:
        raise RelationMeasurementError('Endpoint descriptor masks do not match')
    if scales.shape != (raw.shape[1],) or direction.shape != scales.shape:
        raise RelationMeasurementError('Training scales and direction do not match cues')
    if np.isinf(raw).any() or not np.array_equal(observed, np.isfinite(raw)) or (not np.isfinite(scales).all()) or np.any(scales <= 0) or (not np.isfinite(direction).all()):
        raise RelationMeasurementError('Endpoint descriptors or training calibration are invalid')
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=1e-06, atol=1e-08):
        raise RelationMeasurementError('Training direction must have unit norm')
    if np.any(support & ~observed):
        raise RelationMeasurementError('A supported endpoint coordinate is unavailable')
    if np.any(~np.any(support, axis=1)):
        raise RelationMeasurementError('Every scored word requires supported coordinates')
    counts = support.sum(axis=0)
    selected = np.where(support, raw, 0.0)
    means = selected.sum(axis=0) / np.maximum(counts, 1)
    phi = np.where(support, selected - means[None, :], 0.0) / scales[None, :]
    scores = phi @ direction
    if not np.isfinite(scores).all():
        raise RelationMeasurementError('Endpoint prominence is non-finite')
    return scores
