import json
from pathlib import Path

def main():
    root = Path(__file__).absolute().parents[1]
    identifiers, groups = ({}, {})
    for split, count in [('train', 68), ('dev', 8), ('test', 8)]:
        rows = json.loads((root / f'data/splits/expresso_{split}.json').read_text(encoding='utf-8'))
        groups[split] = {r['group_id'] for r in rows}
        identifiers[split] = {e['source_id'] for r in rows for e in r['reference_endpoints']}
        if len(rows) != count or len(groups[split]) != count or None in identifiers[split]:
            raise ValueError('Invalid split identity/count')
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if groups[a] & groups[b] or identifiers[a] & identifiers[b]:
            raise ValueError('Source/group IDs overlap across splits')
    print('Source IDs and group split memberships are consistent.')
if __name__ == '__main__':
    main()
