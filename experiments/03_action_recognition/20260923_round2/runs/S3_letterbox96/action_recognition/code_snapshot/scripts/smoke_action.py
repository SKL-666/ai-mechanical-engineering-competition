#!/usr/bin/env python3
"""Read every indexed sample, exercise 5-class logits, export a clearly untrained CSV."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from earthmoving.common import write_json, write_csv, CLASSES
from earthmoving.data import EarthmovingClips
from earthmoving.model import TinyActionCNN

root = Path(__file__).resolve().parents[1]
torch.manual_seed(42)
torch.set_num_threads(4)
d = root / 'derived/earthmoving_v1'
out = root / 'experiments/01_dataset'
counts, examples = {}, []
model = TinyActionCNN().to('mps').eval()
for split in ['train', 'val', 'test']:
    data = EarthmovingClips(root, d, split)
    for i in range(len(data)):
        x, y, _ = data[i]
        assert x.shape == (3, 1, 96, 96) and torch.isfinite(x).all()
    counts[split] = len(data)
    if split == 'train':
        for label in range(5):
            i = next(i for i, r in enumerate(data.rows) if r['label'] == label)
            x, y, _ = data[i]
            with torch.inference_mode():
                scores = model(x.unsqueeze(0).to('mps')).softmax(-1).cpu()
            assert scores.shape == (1, 5) and torch.isclose(scores.sum(), torch.tensor(1.0))
            c, pred = scores[0].max(-1)
            r = data.rows[i]
            examples.append({'video_id': r['video_id'], 'start_time': r['start_frame'] / 25,
                             'end_time': r['end_frame'] / 25, 'true_action': CLASSES[y],
                             'pred_action': CLASSES[int(pred)], 'confidence': float(c)})
train = EarthmovingClips(root, d, 'train', training=True)
for i in range(len(train)):
    train[i]
write_csv(out / 'smoke_UNTRAINED_train_examples.csv', examples)
write_json(out / 'smoke.json', {'all_classification_clips_read': counts, 'all_training_windows_read': len(train),
                              'logits_shape': [1, 5], 'mps_forward': 'pass',
                              'csv': 'smoke_UNTRAINED_train_examples.csv', 'csv_purpose': 'untrained format check, not model evaluation'})
print('Smoke passed', counts, 'training windows', len(train))
