from __future__ import annotations
import numpy as np

class PhoneClockError(ValueError):
    pass

def _array(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PhoneClockError(f'{name} must be a NumPy array')
    return value

def validate_phone_clock(phone_ids: np.ndarray, word_indices: np.ndarray, boundaries: np.ndarray) -> int:
    ids = _array(phone_ids, 'phone_ids')
    words = _array(word_indices, 'word_indices')
    edges = _array(boundaries, 'boundaries')
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in 'USiu':
        raise PhoneClockError('phone_ids must be nonempty 1-D strings or integers')
    if ids.dtype.kind in 'US' and np.any(np.char.str_len(ids) == 0) or (ids.dtype.kind in 'iu' and np.any(ids < 0)):
        raise PhoneClockError('phone identities must not be empty or negative')
    if words.shape != ids.shape or words.dtype.kind not in 'iu':
        raise PhoneClockError('word_indices must be integer and match phone_ids')
    if words[0] != 0 or np.any(words[1:] < words[:-1]) or np.any(words[1:].astype(np.float64) - words[:-1].astype(np.float64) > 1):
        raise PhoneClockError('word indices must be ordered and contiguous from zero')
    if edges.shape != (ids.size + 1,) or edges.dtype.kind not in 'fiu':
        raise PhoneClockError('boundaries must contain one numeric edge per phone plus one')
    numeric = edges.astype(np.float64)
    if not np.isfinite(numeric).all() or numeric[0] != 0 or np.any(np.diff(numeric) <= 0):
        raise PhoneClockError('phone edges must start at zero and strictly increase')
    if not float(numeric[-1]).is_integer():
        raise PhoneClockError('the final edge must equal a whole mel-frame count')
    return int(numeric[-1])

def _paired_clocks(src_ids, src_words, src_edges, dst_ids, dst_words, dst_edges):
    source_frames = validate_phone_clock(src_ids, src_words, src_edges)
    target_frames = validate_phone_clock(dst_ids, dst_words, dst_edges)
    if not np.array_equal(src_ids, dst_ids) or not np.array_equal(src_words, dst_words):
        raise PhoneClockError('source/target phone identities and word membership must match')
    return (source_frames, target_frames)

def _same_clock(source: np.ndarray, target: np.ndarray) -> bool:
    return source.dtype == target.dtype and source.tobytes() == target.tobytes()

def shared_phone_boundaries(phone_ids: np.ndarray, word_indices: np.ndarray, native_edges: np.ndarray, word_pi: np.ndarray) -> np.ndarray:
    frames = validate_phone_clock(phone_ids, word_indices, native_edges)
    pi = _array(word_pi, 'word_pi')
    word_count = int(word_indices[-1]) + 1
    if pi.shape != (word_count,) or pi.dtype.kind not in 'fiu':
        raise PhoneClockError('word_pi must have exactly one numeric value per word')
    pi = pi.astype(np.float64)
    mass = float(pi.sum())
    if not np.isfinite(pi).all() or np.any(pi <= 0) or abs(mass - 1.0) > 2e-06:
        raise PhoneClockError('word_pi must be finite, strictly positive and sum to one')
    edges = native_edges.astype(np.float64)
    starts = np.r_[0, np.flatnonzero(word_indices[1:] != word_indices[:-1]) + 1]
    ends = np.r_[starts[1:], phone_ids.size]
    native_totals = edges[ends] - edges[starts]
    if np.array_equal(pi, native_totals / frames):
        return native_edges.copy()
    totals = frames * (pi / mass)
    word_edges = np.r_[0.0, np.cumsum(totals)]
    word_edges[-1] = float(frames)
    result = np.empty(phone_ids.size + 1, dtype=np.float64)
    for word, (begin, end) in enumerate(zip(starts, ends)):
        fractions = (edges[begin:end + 1] - edges[begin]) / native_totals[word]
        result[begin:end + 1] = word_edges[word] + (word_edges[word + 1] - word_edges[word]) * fractions
        result[begin], result[end] = (word_edges[word], word_edges[word + 1])
    validate_phone_clock(phone_ids, word_indices, result)
    return result

def map_phone_positions(positions: np.ndarray, src_ids: np.ndarray, src_words: np.ndarray, src_edges: np.ndarray, dst_ids: np.ndarray, dst_words: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    _, target_frames = _paired_clocks(src_ids, src_words, src_edges, dst_ids, dst_words, dst_edges)
    points = _array(positions, 'positions')
    if points.ndim != 1 or points.dtype.kind not in 'fiu' or (not np.isfinite(points).all()):
        raise PhoneClockError('positions must be a finite numeric vector')
    if np.any(points < 0) or np.any(points > target_frames):
        raise PhoneClockError('positions lie outside the target clock')
    if _same_clock(src_edges, dst_edges):
        return points.copy()
    source, target = (src_edges.astype(np.float64), dst_edges.astype(np.float64))
    owner = np.searchsorted(target[1:-1], points, side='right')
    phase = (points - target[owner]) / (target[owner + 1] - target[owner])
    return source[owner] + phase * (source[owner + 1] - source[owner])

def warp_phone_features(features: np.ndarray, src_ids: np.ndarray, src_words: np.ndarray, src_edges: np.ndarray, dst_ids: np.ndarray, dst_words: np.ndarray, dst_edges: np.ndarray) -> tuple[np.ndarray, dict]:
    source_frames, target_frames = _paired_clocks(src_ids, src_words, src_edges, dst_ids, dst_words, dst_edges)
    values = _array(features, 'features')
    if values.ndim < 1 or values.shape[0] != source_frames or values.size == 0 or (values.dtype.kind != 'f') or (not np.isfinite(values).all()):
        raise PhoneClockError('features must be finite floating [T_source,...]')
    source_centers = np.arange(source_frames, dtype=np.float64) + 0.5
    target_centers = np.arange(target_frames, dtype=np.float64) + 0.5
    mapped = map_phone_positions(target_centers, src_ids, src_words, src_edges, dst_ids, dst_words, dst_edges)
    source_owner = np.searchsorted(src_edges[1:-1], source_centers, side='right')
    target_owner = np.searchsorted(dst_edges[1:-1], target_centers, side='right')
    source_counts = np.bincount(source_owner, minlength=src_ids.size)
    target_counts = np.bincount(target_owner, minlength=dst_ids.size)
    identity = source_frames == target_frames and _same_clock(src_edges, dst_edges)
    diagnostics = {'identity': identity, 'source_frames': source_frames, 'target_frames': target_frames, 'source_sample_positions': mapped, 'target_phone_index': target_owner, 'source_phone_frame_counts': source_counts, 'target_phone_frame_counts': target_counts, 'source_subframe_phone_indices': np.flatnonzero(np.diff(src_edges.astype(np.float64)) < 1), 'target_subframe_phone_indices': np.flatnonzero(np.diff(dst_edges.astype(np.float64)) < 1), 'source_zero_center_phone_indices': np.flatnonzero(source_counts == 0), 'target_zero_center_phone_indices': np.flatnonzero(target_counts == 0), 'sampling': 'source-frame-center-linear; utterance-end-clamp'}
    if identity:
        return (values.copy(), diagnostics)
    positions = np.clip(mapped - 0.5, 0, source_frames - 1)
    left = np.floor(positions).astype(np.intp)
    right = np.minimum(left + 1, source_frames - 1)
    fraction = (positions - left).reshape((target_frames,) + (1,) * (values.ndim - 1))
    warped = values[left] * (1 - fraction) + values[right] * fraction
    if not np.isfinite(warped).all():
        raise PhoneClockError('interpolation produced nonfinite features')
    return (warped.astype(values.dtype), diagnostics)
__all__ = ['PhoneClockError', 'validate_phone_clock', 'shared_phone_boundaries', 'map_phone_positions', 'warp_phone_features']
