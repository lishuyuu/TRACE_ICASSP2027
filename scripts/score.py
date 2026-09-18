import argparse
import json
from pathlib import Path
from trace_tts.metrics import cast_pair, rank_correct, aggregate_cell

def main():
    parser = argparse.ArgumentParser(description='Aggregate the five endpoint and pair metrics')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    cast = payload['cast']
    expresso = payload['expresso']
    if len({r['pair_id'] for r in cast}) != len(cast) or len({r['endpoint_id'] for r in expresso}) != len(expresso):
        raise ValueError('Duplicate observation identity')
    pairs = [cast_pair(r['stressed_a'], r['stressed_b'], r['prominence_a'], r['prominence_b'], r['target_a'], r['target_b']) for r in cast]
    ranks = [rank_correct(r['word_scores'], r['target']) for r in expresso]
    utmos = [r[k] for r in cast for k in ('utmos_a', 'utmos_b')]
    result = aggregate_cell(pairs, ranks, utmos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
if __name__ == '__main__':
    main()
