import csv
import hashlib
import json
from pathlib import Path

CLASSES = ['idle', 'swing_bucket', 'load_bucket', 'dump', 'move']
HYPOTHESES = {'zero_closed': (0, 1), 'one_closed': (-1, 0),
              'zero_half_open': (0, 0), 'one_half_open': (-1, -1)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def label_hypotheses(rows, n):
    """No tie breaking: conflicting labels are -2, missing labels are -1."""
    import numpy as np
    outputs, stats = [], {}
    for name, (ds, de) in HYPOTHESES.items():
        labels = np.full(n, -1, dtype=np.int16)
        out_of_bounds = 0
        for s, e, c in rows:
            start, end = s + ds, e + de
            out_of_bounds += max(0, -start) + max(0, end - n)
            start, end = max(0, start), min(n, end)
            part = labels[start:end]
            conflict = (part != -1) & (part != c)
            part[part == -1] = c
            part[conflict] = -2
        stats[name] = {'unlabeled': int((labels == -1).sum()),
                       'conflict': int((labels == -2).sum()),
                       'out_of_bounds_assignments': out_of_bounds}
        outputs.append(labels)
    stack = np.stack(outputs)
    consensus = np.where((stack == stack[0]).all(axis=0) & (stack[0] >= 0), stack[0], -1)
    return consensus, stats


def spans(mask):
    import numpy as np
    d = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))
