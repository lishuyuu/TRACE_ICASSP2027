import math

def select_gain(dev_rankacc, *, candidates, split):
    if split != 'dev':
        raise ValueError('Only Expresso dev may choose gain')
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Supply unique, predeclared gain candidates')
    if set(dev_rankacc) != set(candidates):
        raise ValueError('Every predeclared candidate must have a development value')
    if any((isinstance(g, bool) or not isinstance(g, (int, float)) or (not math.isfinite(g)) or (g <= 0) or isinstance(dev_rankacc[g], bool) or (not isinstance(dev_rankacc[g], (int, float))) or (not math.isfinite(dev_rankacc[g])) or (not 0 <= dev_rankacc[g] <= 100) for g in candidates)):
        raise ValueError('Invalid gain or RankAcc; no silent candidate exclusion')
    return min(candidates, key=lambda g: (-dev_rankacc[g], g))
