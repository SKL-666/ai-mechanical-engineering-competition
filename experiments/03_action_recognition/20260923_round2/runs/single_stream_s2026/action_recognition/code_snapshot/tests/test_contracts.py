import unittest
from pathlib import Path
import json
import numpy as np
import torch
from earthmoving.common import label_hypotheses, read_jsonl
from earthmoving.data import sample_frames, EarthmovingClips
from earthmoving.metrics import classification_metrics
from earthmoving.model import TinyActionCNN
from scripts.infer_continuous import merge_segments, hold_end, reject_label_access


class Contracts(unittest.TestCase):
    def test_continuous_forbids_truth_but_allows_images(self):
        for path in ['Data/Labels/1/excavator.txt', 'derived/clips.jsonl', 'derived/windows.jsonl',
                     'derived/104154_consensus.npy', b'derived/raw_intervals.jsonl']:
            with self.assertRaises(RuntimeError):
                reject_label_access('open', (path, 'r'))
        reject_label_access('open', ('Data/frames_ce/1/1_I00000.jpg', 'r'))

    def test_hold_prediction_stops_at_first_track_gap(self):
        boxes = {('v', '1', t): (0, 0, 1, 1) for t in [63, 64, 66, 67]}
        self.assertEqual(hold_end(boxes, 'v', '1', 63, 68), 65)

    def test_ambiguous_endpoints_never_fill_idle(self):
        labels, stats = label_hypotheses([[1, 5, 2], [6, 10, 3]], 12)
        self.assertEqual(labels.tolist(), [-1, 2, 2, 2, -1, -1, 3, 3, 3, -1, -1, -1])
        self.assertEqual(stats['one_closed']['unlabeled'], 2)

    def test_conflicting_shared_endpoint_is_ignored(self):
        labels, _ = label_hypotheses([[1, 5, 0], [5, 10, 3]], 12)
        self.assertTrue((labels[4:6] == -1).all())

    def test_short_clip_and_single_frame_do_not_escape(self):
        self.assertEqual(sample_frames(8, 9, 8).tolist(), [8] * 8)
        self.assertEqual(sample_frames(8, 12, 1).tolist(), [9])
        with self.assertRaises(ValueError):
            sample_frames(8, 8, 1)

    def test_macro_always_keeps_five_classes(self):
        m, c, cm = classification_metrics([0, 1], [0, 0])
        self.assertAlmostEqual(m['accuracy'], .5)
        self.assertAlmostEqual(m['macro_f1'], (2 / 3) / 5)
        self.assertEqual([r['support'] for r in c], [1, 1, 0, 0, 0])

    def test_no_segment_merge_across_gap_or_device(self):
        def r(s, e, device='1'):
            return dict(video_id='v', equipment_id=device, start_frame=s, end_frame=e,
                        start_time=s / 25, end_time=e / 25, pred_action='idle', confidence=.5)
        result = merge_segments([r(0, 3), r(3, 5), r(6, 8), r(8, 9, '2')])
        self.assertEqual([(r['start_frame'], r['end_frame']) for r in result], [(0, 5), (6, 8), (8, 9)])

    def test_local_data_split_and_model(self):
        root = Path(__file__).resolve().parents[1]
        d = root / 'derived/earthmoving_v1'
        split = json.loads((d / 'split.json').read_text())['groups']
        self.assertEqual(len(set(sum(split.values(), []))), 10)
        for name in split:
            data = EarthmovingClips(root, d, name, frames=8, size=64)
            self.assertEqual({r['label'] for r in data.rows}, set(range(5)))
            self.assertTrue(all(r['video_id'] in split[name] for r in data.rows))
            for c in range(5):
                idx = next(i for i, r in enumerate(data.rows) if r['label'] == c)
                x, y, _ = data[idx]
                self.assertEqual(x.shape, (3, 8, 64, 64))
                self.assertTrue(torch.isfinite(x).all())
                self.assertEqual(TinyActionCNN()(x.unsqueeze(0)).shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
