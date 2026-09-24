#!/usr/bin/env python3
"""Post-hoc frame evaluation; labels are inaccessible to inference decisions."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from earthmoving.common import CLASSES, write_json, write_csv, sha256
from earthmoving.metrics import classification_metrics

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--derived', type=Path, default=Path('derived/earthmoving_v1'))
    a = p.parse_args()
    cfg = json.loads((a.inference / 'inference_config.json').read_text())
    truth = np.load(a.derived / f'{cfg["video_id"]}_consensus.npy')
    with (a.inference / 'action_segments.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert cfg['equipment_ids'] == ['1'], 'Current GT evaluation supports only the one annotated excavator'
    predicted = np.full(len(truth), -1, dtype=int)
    for r in rows:
        s, e = int(r['start_frame']), int(r['end_frame'])
        assert 0 <= s < e <= len(truth) and (predicted[s:e] == -1).all()
        predicted[s:e] = CLASSES.index(r['pred_action'])
    mask = (truth >= 0) & (predicted >= 0)
    metrics, classes, cm = classification_metrics(truth[mask], predicted[mask])
    result = {'unit': 'frame, post-hoc consensus labels only', 'bbox_source': cfg['bbox_source'],
              'metrics': metrics, 'all_frames': len(truth), 'valid_truth_frames': int((truth >= 0).sum()),
              'evaluated_frames': int(mask.sum()), 'valid_truth_without_prediction': int(((truth >= 0) & (predicted < 0)).sum()),
              'ignored_truth_frames': int((truth < 0).sum()),
              'coverage_of_valid_truth': float(mask.sum() / (truth >= 0).sum()),
              'includes_temporal_segment_metrics': False,
              'prediction_sha256': sha256(a.inference / 'action_segments.csv')}
    write_json(a.inference / 'frame_evaluation.json', result)
    write_csv(a.inference / 'frame_class_metrics.csv', classes)
    write_json(a.inference / 'frame_confusion_matrix.json', cm.tolist())
    print(result)
