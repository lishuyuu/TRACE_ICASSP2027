import math
import statistics

def finite(value):
    return isinstance(value, (float, int)) and (not isinstance(value, bool)) and math.isfinite(value)

def complete_mean(values):
    values = list(values)
    if any((v is not None and (not isinstance(v, (int, float))) for v in values)):
        raise ValueError('Mean inputs must be numeric values, binary outcomes, or None')
    if not values or any((v is None or not math.isfinite(v) for v in values)):
        return None
    return sum(values) / len(values)

def pair_contrast(stressed_indices, target, competitor):
    if type(target) is not int or type(competitor) is not int or target == competitor or (min(target, competitor) < 0):
        raise ValueError('Distinct nonnegative target indices required')
    if stressed_indices is None:
        return None
    if not isinstance(stressed_indices, (list, tuple)) or any((type(i) is not int or i < 0 for i in stressed_indices)):
        raise ValueError('Detector word indices must be nonnegative integers')
    if len(set(stressed_indices)) != len(stressed_indices):
        raise ValueError('Detector word indices must be unique, not a word-level 0/1 label vector')
    return target in stressed_indices and competitor not in stressed_indices

def cast_pair(stressed_a, stressed_b, scores_a, scores_b, target_a, target_b):
    a = pair_contrast(stressed_a, target_a, target_b)
    b = pair_contrast(stressed_b, target_b, target_a)
    correct = None if a is None or b is None else a and b
    margin = None
    if scores_a is not None and scores_b is not None:
        if len(scores_a) != len(scores_b) or max(target_a, target_b) >= len(scores_a):
            raise ValueError('Prominence vectors do not share the target word map')
        if any((indices is not None and any((i >= len(scores_a) for i in indices)) for indices in (stressed_a, stressed_b))):
            raise ValueError('Detector index is outside the supplied word map')
        vals = [scores_a[target_a], scores_a[target_b], scores_b[target_b], scores_b[target_a]]
        if all((finite(v) for v in vals)):
            margin = (vals[0] - vals[1] + vals[2] - vals[3]) / 2
    return {'pair_correct': correct, 'contrast_a': a, 'contrast_b': b, 'delta_P': margin}

def rank_correct(word_scores, target):
    if word_scores is None:
        return None
    if len(word_scores) < 2 or type(target) is not int or (not 0 <= target < len(word_scores)):
        raise ValueError('Invalid target or word count')
    if not all((finite(v) for v in word_scores)):
        return None
    return word_scores[target] > max((v for i, v in enumerate(word_scores) if i != target))

def switching_outcomes(cast_pairs):
    counts = [0, 0, 0]
    missing = not cast_pairs
    for row in cast_pairs:
        a, b = (row['contrast_a'], row['contrast_b'])
        correct = row['pair_correct']
        if any((value is not None and type(value) is not bool for value in (a, b, correct))):
            raise ValueError('Switching outcomes require binary direction observations')
        expected = None if a is None or b is None else a and b
        if correct is not expected:
            raise ValueError('Pair-Correct differs from its two direction observations')
        if expected is None:
            missing = True
        else:
            counts[int(a) + int(b)] += 1
    if missing:
        return {'f0': None, 'f1': None, 'f2': None}
    return {f'f{i}': 100 * count / len(cast_pairs) for i, count in enumerate(counts)}

def aggregate_cell(cast_pairs, expresso_endpoints, utmos_endpoints, *, expected_pairs=113, expected_expresso=30):
    if len(cast_pairs) != expected_pairs or len(expresso_endpoints) != expected_expresso or len(utmos_endpoints) != 2 * expected_pairs:
        raise ValueError('Incomplete registered evaluation cell')

    def percent(values):
        value = complete_mean(values)
        return None if value is None else 100 * value
    return {'Pair-Correct': percent((r['pair_correct'] for r in cast_pairs)), 'Pair-Contrast': percent((r[k] for r in cast_pairs for k in ('contrast_a', 'contrast_b'))), 'delta_P': complete_mean((r['delta_P'] for r in cast_pairs)), 'RankAcc': percent(expresso_endpoints), 'UTMOS': complete_mean(utmos_endpoints)}

def aggregate_seeds(cells, *, seeds=(2703, 2704, 2705), prompts):
    if len(seeds) != 3 or any((type(s) is not int for s in seeds)) or len(set(seeds)) != 3:
        raise ValueError('Three distinct integer training seeds required')
    if len(prompts) != 4 or len(set(prompts)) != 4:
        raise ValueError('Four distinct fixed prompts required')
    expected = {(s, p) for s in seeds for p in prompts}
    if set(cells) != expected:
        raise ValueError('Seed/prompt grid differs from protocol')
    means = [complete_mean((cells[s, p] for p in prompts)) for s in seeds]
    if any((v is None for v in means)):
        return {'mean': None, 'sample_sd': None}
    return {'mean': statistics.mean(means), 'sample_sd': statistics.stdev(means)}
