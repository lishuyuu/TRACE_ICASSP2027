from __future__ import annotations
import math
import numpy as np
import torch
from torch import Tensor
from .trace_phone_clock import map_phone_positions

def finite_tensor(value, name, *, ndim=None, dtype=None):
    if not isinstance(value, Tensor) or (ndim is not None and value.ndim != ndim):
        raise ValueError(f'{name} has wrong tensor rank/type')
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f'{name} must have dtype {dtype}')
    if not value.numel() or not value.is_floating_point() or (not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be nonempty finite floating data')
    return value

def shared_native_geometry(native, shared_edges, *, device, require_native_identity=False):
    alignment = native['alignment']
    ids = np.asarray(alignment['phone_ids'])
    words = alignment['phone_word_indices'].numpy()
    source_edges = alignment['clock_edges'].numpy()
    if require_native_identity:
        if not np.array_equal(np.asarray(shared_edges), source_edges):
            raise RuntimeError('native pi0 must reuse the exact frozen phone clock')
        cached = native['frame_geometry']
        return {name: cached[name].to(device) for name in ('word_index', 'nucleus_phase', 'voicing_gate')}
    frames = int(source_edges[-1])
    target_frames = int(shared_edges[-1])
    if target_frames != frames:
        raise ValueError('shared clock must preserve native total frame count')
    centers = np.arange(target_frames, dtype=np.float64) + 0.5
    phone = np.searchsorted(shared_edges[1:-1], centers, side='right')
    word_index = words[phone]
    source_position = map_phone_positions(centers, ids, words, source_edges, ids, words, np.asarray(shared_edges))
    duration = alignment.get('duration_sec')
    if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
        raise ValueError('native alignment duration missing/nonpositive')
    seconds = source_position * float(duration) / frames
    descriptor = native.get('descriptor', {})
    f0_times = finite_tensor(descriptor.get('f0_times_sec'), 'native F0 times', ndim=1, dtype=torch.float64).numpy()
    f0_value_tensor = descriptor.get('f0_hz')
    if not isinstance(f0_value_tensor, Tensor) or f0_value_tensor.dtype != torch.float64 or f0_value_tensor.ndim != 1 or (not f0_value_tensor.numel()):
        raise ValueError('native F0 values must be a nonempty float64 vector')
    f0_values = f0_value_tensor.numpy()
    if f0_times.shape != f0_values.shape or np.any(np.diff(f0_times) <= 0):
        raise ValueError('native WORLD F0 grid is missing/nonincreasing')
    right = np.searchsorted(f0_times, seconds, side='left').clip(0, len(f0_times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(seconds - f0_times[left] <= f0_times[right] - seconds, left, right)
    nucleus_phone = alignment.get('nucleus_phone_indices')
    fractions = alignment.get('nucleus_clock_fractions')
    word_count = int(words[-1]) + 1
    if not isinstance(nucleus_phone, Tensor) or nucleus_phone.dtype != torch.int64 or nucleus_phone.shape != (word_count,) or (not isinstance(fractions, Tensor)) or (fractions.dtype != torch.float64) or (fractions.shape != (word_count, 2)):
        raise ValueError('native nucleus phone/fraction geometry missing')
    phase = np.zeros(target_frames, dtype=np.float32)
    gate = np.zeros(target_frames, dtype=np.float32)
    for word, phone_index in enumerate(nucleus_phone.tolist()):
        if phone_index < 0:
            if not torch.isnan(fractions[word]).all():
                raise ValueError('nucleus-free word has nonmissing fractions')
            continue
        if not 0 <= phone_index < len(ids) or words[phone_index] != word:
            raise ValueError('nucleus phone does not belong to its word')
        fraction = fractions[word].numpy()
        if not np.isfinite(fraction).all() or not 0 <= fraction[0] < fraction[1] <= 1:
            raise ValueError('native nucleus fractions invalid')
        begin = shared_edges[phone_index] + fraction[0] * (shared_edges[phone_index + 1] - shared_edges[phone_index])
        end = shared_edges[phone_index] + fraction[1] * (shared_edges[phone_index + 1] - shared_edges[phone_index])
        inside = (phone == phone_index) & (centers >= begin) & (centers < end)
        phase[inside] = ((centers[inside] - begin) / (end - begin)).astype(np.float32)
        voiced = np.isfinite(f0_values[nearest]) & (f0_values[nearest] > 0)
        gate[inside & voiced] = 1
    result = {'word_index': torch.from_numpy(word_index.astype(np.int64)).to(device), 'nucleus_phase': torch.from_numpy(phase).to(device), 'voicing_gate': torch.from_numpy(gate).to(device)}
    return result
