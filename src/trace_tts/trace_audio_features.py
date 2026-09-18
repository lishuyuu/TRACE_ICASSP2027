from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Sequence
import numpy as np
from .trace_phone_clock import validate_phone_clock
CUES = ('p0', 'p1', 'p2', 'E', 'd')
SR = 24000
SCALE_FLOOR = 1e-06
DIRECTION_MIN_NORM = 1e-08
WORD_RE = re.compile("[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
VOWELS = frozenset('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split())
CONSONANTS = frozenset('B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH'.split())
SILENCES = frozenset(('', 'sil', 'sp', '<eps>'))
TIME_TOL = 5e-07

class FeatureError(ValueError):
    pass

@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str

@dataclass(frozen=True)
class TextGrid:
    duration_sec: float
    tiers: dict[str, tuple[Interval, ...]]

@dataclass(frozen=True)
class PauseAllocation:
    start: float
    end: float
    left_phone: int | None
    right_phone: int | None
    left_seconds: float
    right_seconds: float

@dataclass(frozen=True)
class Alignment:
    words: tuple[str, ...]
    raw_word_intervals: tuple[Interval, ...]
    raw_phone_intervals: tuple[Interval, ...]
    word_intervals: tuple[Interval, ...]
    phone_ids: np.ndarray
    phone_word_indices: np.ndarray
    speech_phone_intervals: np.ndarray
    clock_edges_seconds: np.ndarray
    clock_edges: np.ndarray
    word_pi: np.ndarray
    raw_word_durations: np.ndarray
    speech_phone_durations_by_word: np.ndarray
    nucleus_phone_indices: np.ndarray
    nucleus_intervals: np.ndarray
    pause_allocations: tuple[PauseAllocation, ...]
    pause_seconds: float
    duration_sec: float
    mel_frames: int

@dataclass(frozen=True)
class Descriptor:
    words: tuple[str, ...]
    values: np.ndarray
    observed: np.ndarray
    raw_missing: tuple[tuple[str | None, ...], ...]
    pitch_sample_count: np.ndarray

@dataclass(frozen=True)
class GroupFeatures:
    group_id: str
    corpus: str
    split: str
    words: tuple[str, ...]
    target_words: np.ndarray
    raw: np.ndarray
    observed: np.ndarray
    raw_missing: tuple
    support: np.ndarray
    completed: np.ndarray
    centered: np.ndarray
    O: np.ndarray
    valid_pairs_by_cue: tuple[tuple[tuple[int, int], ...], ...]

@dataclass(frozen=True)
class TrainingInputs:
    group_ids: tuple[str, ...]
    cue_indices: tuple[int, ...]
    cue_names: tuple[str, ...]
    scales: np.ndarray
    all_scales: np.ndarray
    scale_group_counts: np.ndarray
    unavailable_reasons: tuple[str | None, ...]
    phi: tuple[np.ndarray, ...]
    z: tuple[np.ndarray, ...]
    O: tuple[np.ndarray, ...]
    support: tuple[np.ndarray, ...]
    target_words: tuple[np.ndarray, ...]
    valid_pairs_by_cue: tuple[tuple[tuple[int, int, int], ...], ...]
    dbar: np.ndarray
    omega: np.ndarray

def canonical_words(text: str) -> tuple[str, ...]:
    return tuple(WORD_RE.findall(text.replace('’', "'").casefold()))

def _field(body: str, name: str, *, quoted: bool=False):
    value = '("(?:""|[^"])*")' if quoted else '([^\\r\\n]+)'
    matches = re.findall('(?m)^\\s*' + re.escape(name) + '\\s*=\\s*' + value + '\\s*$', body)
    if len(matches) != 1:
        raise FeatureError(f'TextGrid requires exactly one {name!r} field')
    raw = matches[0].strip()
    if quoted:
        return raw[1:-1].replace('""', '"')
    try:
        result = float(raw)
    except ValueError as exc:
        raise FeatureError(f'invalid numeric TextGrid field {name}') from exc
    if not np.isfinite(result):
        raise FeatureError(f'nonfinite TextGrid field {name}')
    return result

def parse_textgrid(text: str) -> TextGrid:
    if not isinstance(text, str) or not re.search('Object class\\s*=\\s*"TextGrid"', text):
        raise FeatureError('expected a long-format TextGrid string')
    chunks = re.split('(?m)^\\s*item\\s*\\[\\s*(\\d+)\\s*\\]\\s*:\\s*$', text)
    header = chunks[0]
    duration = _field(header, 'xmax')
    if _field(header, 'xmin') != 0 or duration <= 0:
        raise FeatureError('TextGrid must begin at zero and have positive duration')
    tier_count = _field(header, 'size')
    if tier_count != len(chunks[1::2]):
        raise FeatureError('TextGrid tier count mismatch')
    tiers, seen_names = ({}, set())
    for expected_index, (index, body) in enumerate(zip(chunks[1::2], chunks[2::2]), 1):
        if int(index) != expected_index:
            raise FeatureError('TextGrid item indices are not contiguous')
        segments = re.split('(?m)^\\s*intervals\\s*\\[\\s*(\\d+)\\s*\\]\\s*:\\s*$', body)
        meta = segments[0]
        name = _field(meta, 'name', quoted=True)
        if name in seen_names:
            raise FeatureError('duplicate TextGrid tier name')
        seen_names.add(name)
        if _field(meta, 'class', quoted=True) != 'IntervalTier':
            continue
        if _field(meta, 'xmin') != 0 or abs(_field(meta, 'xmax') - duration) > TIME_TOL:
            raise FeatureError('tier/global time bounds disagree')
        if _field(meta, 'intervals: size') != len(segments[1::2]):
            raise FeatureError('TextGrid interval count mismatch')
        intervals = []
        for j, (interval_index, fields) in enumerate(zip(segments[1::2], segments[2::2]), 1):
            if int(interval_index) != j:
                raise FeatureError('TextGrid interval indices are not contiguous')
            intervals.append(Interval(_field(fields, 'xmin'), _field(fields, 'xmax'), _field(fields, 'text', quoted=True)))
        _validate_intervals(intervals, duration, name)
        tiers[name] = tuple(intervals)
    return TextGrid(duration, tiers)

def _validate_intervals(intervals, duration: float, name: str):
    previous_end = 0.0
    for x in intervals:
        if not np.isfinite([x.start, x.end]).all() or x.start < 0 or x.end <= x.start or (x.end > duration + TIME_TOL) or (x.start < previous_end - TIME_TOL) or (not isinstance(x.label, str)):
            raise FeatureError(f'{name}: nonpositive, overlapping or out-of-range interval')
        previous_end = x.end

def _phone_id(label: str) -> str | None:
    value = label.strip()
    if value.casefold() in SILENCES:
        return None
    value = value.upper()
    match = re.fullmatch('([A-Z]+)([012]?)', value)
    if not match or not (match[1] in VOWELS or (match[1] in CONSONANTS and (not match[2]))):
        raise FeatureError(f'OOV/noise/non-ARPA phone: {label!r}')
    return value

def build_alignment(word_intervals: Sequence[Interval], phone_intervals: Sequence[Interval], expected_words: Sequence[str] | str, duration_sec: float, mel_frames: int) -> Alignment:
    if not np.isfinite(duration_sec) or duration_sec <= 0 or isinstance(mel_frames, bool) or (not isinstance(mel_frames, (int, np.integer))) or (mel_frames <= 0):
        raise FeatureError('positive utterance duration and actual integer mel length required')
    if isinstance(expected_words, str):
        words = canonical_words(expected_words)
    else:
        normalized = [canonical_words(w) for w in expected_words]
        if any((len(w) != 1 for w in normalized)):
            raise FeatureError('each expected word must be one registry lexical unit')
        words = tuple((w[0] for w in normalized))
    if not words:
        raise FeatureError('empty expected words')
    raw_words, raw_phones = (tuple(word_intervals), tuple(phone_intervals))
    _validate_intervals(raw_words, duration_sec, 'words')
    _validate_intervals(raw_phones, duration_sec, 'phones')
    lexical, labels = ([], [])
    for item in raw_words:
        if item.label.strip().casefold() in SILENCES:
            continue
        tokens = canonical_words(item.label)
        if len(tokens) != 1:
            raise FeatureError('MFA word interval is not one lexical unit')
        lexical.append(item)
        labels.append(tokens[0])
    if tuple(labels) != words:
        raise FeatureError(f'MFA word identity mismatch: expected {words}, got {tuple(labels)}')
    ids, ownership, speech = ([], [], [])
    for item in raw_phones:
        identity = _phone_id(item.label)
        if identity is None:
            continue
        owners = [w for w, interval in enumerate(lexical) if item.start >= interval.start - TIME_TOL and item.end <= interval.end + TIME_TOL]
        if len(owners) != 1:
            raise FeatureError('speech phone lacks a unique containing actual word interval')
        ids.append(identity)
        ownership.append(owners[0])
        speech.append((item.start, item.end))
    if not ids or set(ownership) != set(range(len(words))):
        raise FeatureError('every word must contain actual speech phones')
    intervals = np.asarray(speech, dtype=np.float64)
    owners = np.asarray(ownership, dtype=np.int64)
    edges = np.concatenate(([0.0], (intervals[:-1, 1] + intervals[1:, 0]) / 2, [duration_sec]))
    phone_ids = np.asarray(ids)
    frame_edges = edges * (int(mel_frames) / duration_sec)
    frame_edges[-1] = int(mel_frames)
    validate_phone_clock(phone_ids, owners, frame_edges)
    if (intervals[:, 0] < edges[:-1] - TIME_TOL).any() or (intervals[:, 1] > edges[1:] + TIME_TOL).any():
        raise FeatureError('allocation does not contain the original speech phone')
    allocations = []
    if intervals[0, 0] > 0:
        allocations.append(PauseAllocation(0.0, float(intervals[0, 0]), None, 0, 0.0, float(intervals[0, 0])))
    for j, (end, start) in enumerate(zip(intervals[:-1, 1], intervals[1:, 0])):
        if start < end - TIME_TOL:
            raise FeatureError('speech phones overlap')
        if start > end:
            half = float((start - end) / 2)
            allocations.append(PauseAllocation(float(end), float(start), j, j + 1, half, half))
    if intervals[-1, 1] < duration_sec:
        allocations.append(PauseAllocation(float(intervals[-1, 1]), duration_sec, len(ids) - 1, None, float(duration_sec - intervals[-1, 1]), 0.0))
    pi = np.bincount(owners, weights=np.diff(edges), minlength=len(words)) / duration_sec
    if not np.isfinite(pi).all() or (pi <= 0).any() or abs(float(pi.sum()) - 1) > 1e-12:
        raise FeatureError('invalid clock-accounted word composition')
    nuclei = np.full(len(words), -1, dtype=np.int64)
    nucleus_intervals = np.full((len(words), 2), np.nan)
    for w in range(len(words)):
        vowels = [j for j, phone in enumerate(ids) if owners[j] == w and phone.rstrip('012') in VOWELS]
        primary = [j for j in vowels if ids[j].endswith('1')]
        if vowels:
            selected = primary[0] if primary else max(vowels, key=lambda j: (intervals[j, 1] - intervals[j, 0], -j))
            nuclei[w] = selected
            nucleus_intervals[w] = intervals[selected]
    return Alignment(words, raw_words, raw_phones, tuple(lexical), phone_ids, owners, intervals, edges, frame_edges, pi, np.asarray([w.end - w.start for w in lexical]), np.bincount(owners, weights=np.diff(intervals, axis=1)[:, 0], minlength=len(words)), nuclei, nucleus_intervals, tuple(allocations), sum((a.left_seconds + a.right_seconds for a in allocations)), duration_sec, int(mel_frames))

def alignment_from_textgrid(textgrid_text: str, expected_words: Sequence[str] | str, mel_frames: int, *, duration_sec: float | None=None, word_tier: str='words', phone_tier: str='phones') -> Alignment:
    grid = parse_textgrid(textgrid_text)
    if duration_sec is not None and abs(grid.duration_sec - duration_sec) > 1 / SR:
        raise FeatureError('TextGrid/audio duration mismatch')
    if word_tier not in grid.tiers or phone_tier not in grid.tiers or word_tier == phone_tier:
        raise FeatureError('explicit word/phone IntervalTier names are absent or identical')
    return build_alignment(grid.tiers[word_tier], grid.tiers[phone_tier], expected_words, grid.duration_sec if duration_sec is None else duration_sec, mel_frames)

def require_matching_phones(alignments: Sequence[Alignment]) -> None:
    if len(alignments) < 2:
        raise FeatureError('phone matching requires at least two alignments')
    first = alignments[0]
    for item in alignments:
        validate_phone_clock(item.phone_ids, item.phone_word_indices, item.clock_edges)
        if item.words != first.words or not np.array_equal(item.phone_ids, first.phone_ids) or (not np.array_equal(item.phone_word_indices, first.phone_word_indices)):
            raise FeatureError('non-silence phone sequence or word ownership mismatch')

def _audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    value = np.asarray(audio)
    if sample_rate != SR or value.ndim != 1 or (not value.size) or (value.dtype.kind != 'f') or (not np.isfinite(value).all()):
        raise FeatureError('finite mono floating-point 24kHz waveform required; no hidden resampling')
    return np.ascontiguousarray(value, dtype=np.float64)

def world_f0(audio: np.ndarray, sample_rate: int=SR) -> tuple[np.ndarray, np.ndarray]:
    waveform = _audio(audio, sample_rate)
    import pyworld
    initial, times = pyworld.dio(waveform, sample_rate, f0_floor=50.0, f0_ceil=600.0, frame_period=5.0)
    refined = pyworld.stonemask(waveform, initial, times, sample_rate)
    return (np.asarray(times, dtype=np.float64), np.asarray(refined, dtype=np.float64))

def describe(audio: np.ndarray, sample_rate: int, alignment: Alignment, *, f0_times: np.ndarray, f0: np.ndarray) -> Descriptor:
    waveform = _audio(audio, sample_rate)
    if abs(len(waveform) / sample_rate - alignment.duration_sec) > 1 / sample_rate:
        raise FeatureError('alignment/waveform duration mismatch')
    times, pitch = (np.asarray(f0_times, dtype=np.float64), np.asarray(f0, dtype=np.float64))
    if times.ndim != 1 or pitch.shape != times.shape or (not times.size) or (not np.isfinite(times).all()) or (np.diff(times) <= 0).any() or (times[0] < 0) or (times[-1] > alignment.duration_sec + 1 / sample_rate):
        raise FeatureError('invalid F0 time sequence')
    w_count = len(alignment.words)
    values = np.full((w_count, 5), np.nan)
    reasons = [[None] * 5 for _ in range(w_count)]
    counts = np.zeros(w_count, dtype=np.int64)
    for w, interval in enumerate(alignment.word_intervals):
        start, end = (int(np.ceil(interval.start * sample_rate)), int(np.ceil(interval.end * sample_rate)))
        samples = waveform[start:min(end, len(waveform))]
        if not samples.size:
            raise FeatureError(f'word {w} has no actual waveform samples')
        with np.errstate(over='ignore', invalid='ignore'):
            values[w, 3] = np.log(np.sqrt(np.mean(samples * samples)) + 1e-08)
        if not np.isfinite(values[w, 3]):
            raise FeatureError(f'word {w} energy is nonfinite')
        if alignment.nucleus_phone_indices[w] < 0:
            reasons[w][:3] = ['no_arpa_vowel_nucleus'] * 3
            continue
        begin, finish = alignment.nucleus_intervals[w]
        valid = (times >= begin) & (times < finish) & np.isfinite(pitch) & (pitch > 0)
        counts[w] = np.count_nonzero(valid)
        if counts[w] < 5:
            reasons[w][:3] = ['fewer_than_five_voiced_nucleus_samples'] * 3
            continue
        phase = 2 * (times[valid] - begin) / (finish - begin) - 1
        design = np.column_stack((np.ones_like(phase), phase, (3 * phase * phase - 1) / 2))
        coefficients, _, rank, _ = np.linalg.lstsq(design, np.log(pitch[valid]), rcond=None)
        if rank != 3 or not np.isfinite(coefficients).all():
            reasons[w][:3] = ['rank_deficient_or_nonfinite_pitch_fit'] * 3
        else:
            values[w, :3] = coefficients
    if not np.isfinite(alignment.word_pi).all() or (alignment.word_pi <= 0).any():
        raise FeatureError('duration composition is unavailable')
    log_pi = np.log(alignment.word_pi)
    values[:, 4] = log_pi - log_pi.mean()
    return Descriptor(alignment.words, values, np.isfinite(values), tuple((tuple(r) for r in reasons)), counts)

def prepare_group(descriptors: Sequence[Descriptor], targets: Sequence[int], *, group_id: str, corpus: str, split: str) -> GroupFeatures:
    if len(descriptors) < 2 or not group_id or (not corpus) or (not split):
        raise FeatureError('registered group with >=2 endpoints required')
    words = descriptors[0].words
    if len(words) < 2:
        raise FeatureError('a relational group needs >=2 words')
    target = np.asarray(targets)
    if target.dtype.kind not in 'iu' or target.shape != (len(descriptors),) or len(set(target.tolist())) != len(target) or (target < 0).any() or (target >= len(words)).any():
        raise FeatureError('each endpoint needs one distinct in-range target word')
    for item in descriptors:
        if item.words != words or item.values.shape != (len(words), 5) or item.observed.shape != item.values.shape or (item.observed.dtype.kind != 'b') or (not np.array_equal(item.observed, np.isfinite(item.values))) or (not item.observed[:, 3:].all()) or np.isinf(item.values).any() or (np.shape(item.raw_missing) != item.values.shape):
            raise FeatureError('descriptor identity/mask mismatch or missing actual E/d')
    raw = np.stack([d.values for d in descriptors]).copy()
    observed = np.stack([d.observed for d in descriptors]).copy()
    support = observed.sum(axis=0) >= 2
    completed = np.zeros_like(raw)
    for w, q in zip(*np.nonzero(support)):
        measured = observed[:, w, q]
        neutral = raw[measured, w, q].mean()
        completed[:, w, q] = np.where(measured, raw[:, w, q], neutral)
    denom = np.maximum(support.sum(axis=0), 1)
    means = completed.sum(axis=1) / denom
    centered = (completed - means[:, None, :]) * support[None, :, :]
    O = observed & support[None, :, :]
    valid_pairs = []
    for q in range(5):
        valid_pairs.append(tuple(((i, j) for i in range(len(target)) for j in range(i + 1, len(target)) if O[i, target[i], q] and O[i, target[j], q] and O[j, target[i], q] and O[j, target[j], q])))
    return GroupFeatures(group_id, corpus, split, words, target.astype(np.int64), raw, observed, tuple((d.raw_missing for d in descriptors)), support, completed, centered, O, tuple(valid_pairs))

def fit_training_inputs(groups: Sequence[GroupFeatures]) -> TrainingInputs:
    if not groups or len({g.group_id for g in groups}) != len(groups) or any((g.corpus.casefold() != 'expresso' or g.split.casefold() != 'train' for g in groups)):
        raise FeatureError('scales/omega require distinct admitted Expresso TRAIN groups only')
    scales = np.full(5, np.nan)
    contributing = np.zeros(5, dtype=np.int64)
    reasons = [None] * 5
    all_pairs = [tuple(((g, i, j) for g, item in enumerate(groups) for i, j in item.valid_pairs_by_cue[q])) for q in range(5)]
    for q in range(5):
        squared = [float(np.mean(g.centered[:, :, q][g.O[:, :, q]] ** 2)) for g in groups if g.O[:, :, q].any()]
        contributing[q] = len(squared)
        candidate = float(np.sqrt(np.mean(squared))) if squared else np.nan
        if not np.isfinite(candidate) or candidate <= SCALE_FLOOR:
            reasons[q] = 'unavailable_scale_at_or_below_1e-6'
        else:
            scales[q] = candidate
        if not all_pairs[q]:
            reasons[q] = 'no_valid_observed_target_pair' if reasons[q] is None else reasons[q] + ';no_valid_observed_target_pair'
    active = tuple((q for q in range(5) if reasons[q] is None))
    if active != tuple(range(5)):
        raise FeatureError('all five cues require an observed target pair and positive TRAIN scale')
    indices = list(active)
    phi = tuple((g.centered[:, :, indices] / scales[indices] for g in groups))
    z = tuple((p - p.mean(axis=0, keepdims=True) for p in phi))
    pairs = tuple((all_pairs[q] for q in active))
    dbar = np.asarray([np.mean([phi[g][i, groups[g].target_words[i], q] - phi[g][j, groups[g].target_words[i], q] - (phi[g][i, groups[g].target_words[j], q] - phi[g][j, groups[g].target_words[j], q]) for g, i, j in jq]) for q, jq in enumerate(pairs)])
    norm = float(np.linalg.norm(dbar))
    if not np.isfinite(dbar).all() or not np.isfinite(norm) or norm <= DIRECTION_MIN_NORM:
        raise FeatureError('Eq2 direction unavailable: finite norm must exceed 1e-8')
    return TrainingInputs(tuple((g.group_id for g in groups)), active, tuple((CUES[q] for q in active)), scales[indices], scales, contributing, tuple(reasons), phi, z, tuple((g.O[:, :, indices] for g in groups)), tuple((g.support[:, indices] for g in groups)), tuple((g.target_words.copy() for g in groups)), pairs, dbar, dbar / norm)
