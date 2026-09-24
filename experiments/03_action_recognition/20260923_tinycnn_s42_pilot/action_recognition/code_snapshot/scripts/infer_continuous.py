#!/usr/bin/env python3
"""Label-free causal fixed-grid inference. Never reads an action label or action index."""
import argparse
import csv
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from earthmoving.common import CLASSES, read_jsonl, sha256, spans, write_csv, write_json
from earthmoving.data import box_lookup, load_clip
from earthmoving.model import TinyActionCNN
from scripts.train_action import sync


def merge_segments(rows):
    segments = []
    for r in rows:
        if (segments and all(segments[-1][k] == r[k] for k in ['video_id', 'equipment_id', 'pred_action'])
                and segments[-1]['end_frame'] == r['start_frame']):
            old = segments[-1]
            a = old['end_frame'] - old['start_frame']
            b = r['end_frame'] - r['start_frame']
            old['confidence'] = (a * old['confidence'] + b * r['confidence']) / (a + b)
            old['end_frame'], old['end_time'] = r['end_frame'], r['end_time']
        else:
            segments.append(dict(r))
    return segments


def infer(a):
    root = Path(__file__).resolve().parents[1]
    a.out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['config']
    torch.set_num_threads(cfg['torch_num_threads'])
    torch.use_deterministic_algorithms(cfg['deterministic_algorithms'])
    if a.tracks:
        with a.tracks.open() as f:
            rows = list(csv.DictReader(f))
        source = 'external_tracks_unverified_origin'
        trackfile = a.tracks
    else:
        trackfile = a.xml_gt_prototype / 'boxes.jsonl'
        rows = read_jsonl(trackfile)
        source = 'xml_gt_prototype_NOT_team2_integration'
    rows = [r for r in rows if str(r['video_id']) == a.video]
    boxes = box_lookup(rows)
    ids = sorted({str(r['equipment_id']) for r in rows})
    assert ids, 'No equipment tracks for requested video'
    paths = sorted((root / 'Data/frames_ce' / a.video).glob('*.jpg'))
    numbers = [int(p.stem.split('_I')[1]) for p in paths]
    assert numbers == list(range(len(paths))), 'Image numbering must be contiguous and 0-based'
    n, window, stride, fps = len(paths), cfg['window_frames'], cfg['stride_frames'], cfg['fps']
    model = TinyActionCNN().to(cfg['device'])
    model.load_state_dict(saved['state_dict']); model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros(1, 3, cfg['frames'], cfg['input_size'], cfg['input_size'], device=cfg['device']))
    sync(cfg['device'])
    started = time.perf_counter()
    predictions, windows, gaps = [], [], []
    for eid in ids:
        covered = np.zeros(n, dtype=bool)
        for end in range(window, n + 1, stride):
            start, emit = end - window, end - 1
            stop = min(emit + stride, n)
            # Entire track history must exist; do not bridge tracking gaps or guess boxes.
            if any((a.video, eid, t) not in boxes for t in range(start, end)):
                continue
            row = {'video_id': a.video, 'equipment_id': eid, 'start_frame': start, 'end_frame': end}
            x = load_clip(root, row, boxes, cfg['frames'], cfg['input_size']).unsqueeze(0).to(cfg['device'])
            with torch.inference_mode():
                probs = model(x).softmax(-1)[0].cpu()
            conf, pred = probs.max(-1)
            output = {'video_id': a.video, 'equipment_id': eid, 'start_frame': emit, 'end_frame': stop,
                      'start_time': emit / fps, 'end_time': stop / fps,
                      'pred_action': CLASSES[int(pred)], 'confidence': float(conf)}
            predictions.append(output)
            windows.append({**output, 'input_start_frame': start, 'input_end_frame': end,
                            'available_at_frame': emit, 'bbox_source': source})
            covered[emit:stop] = True
        for start, end in spans(~covered):
            gaps.append({'video_id': a.video, 'equipment_id': eid, 'start_frame': start, 'end_frame': end,
                         'reason': 'warmup_missing_track_or_uncovered_tail'})
    sync(cfg['device'])
    seconds = time.perf_counter() - started
    fields = ['video_id', 'equipment_id', 'start_frame', 'end_frame', 'start_time', 'end_time', 'pred_action', 'confidence']
    write_csv(a.out / 'action_segments.csv', merge_segments(predictions), fields)
    write_csv(a.out / 'window_predictions.csv', windows, fields + ['input_start_frame', 'input_end_frame', 'available_at_frame', 'bbox_source'])
    write_csv(a.out / 'output_gaps.csv', gaps, ['video_id', 'equipment_id', 'start_frame', 'end_frame', 'reason'])
    write_json(a.out / 'inference_config.json', {
        'checkpoint_sha256': sha256(a.checkpoint), 'tracks_sha256': sha256(trackfile),
        'video_id': a.video, 'equipment_ids': ids, 'frames': n, 'bbox_source': source,
        'window_frames': window, 'stride_frames': stride, 'sampled_frames_per_window': cfg['frames'],
        'fps': fps, 'fps_basis': cfg['fps_basis'], 'action_labels_read': False,
        'boundary_trigger': 'fixed global frame grid only', 'causal': True,
        'output_semantics': 'At input end-1, hold prediction for next stride frames; never backdate to input window start',
        'smoothing': 'none; only merge contiguous equal predicted classes, duration-weighted confidence',
        'warmup_frames': window - 1, 'decision_interval_frames': stride,
        'future_frames_used_relative_to_available_at': 0,
        'single_frame_center_sample_age_frames': window // 2 if cfg['frames'] == 1 else None,
        'run_mode': 'offline replay of causal decisions; not a wall-clock live stream',
        'windows_predicted': len(windows), 'wall_seconds': seconds,
        'sampled_input_frames_per_second': len(windows) * cfg['frames'] / seconds,
        'timed_scope': 'track completeness lookup + JPEG/crop/resize + transfer + forward + softmax + collect, batch=1; no detector/tracker',
        'command': ' '.join(sys.argv)})
    print({'video': a.video, 'windows': len(windows), 'segments': len(merge_segments(predictions)), 'seconds': seconds})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--out', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--tracks', type=Path, help='CSV: video_id,equipment_id,frame,x1,y1,x2,y2')
    group.add_argument('--xml-gt-prototype', type=Path, help='Explicitly use derived XML GT boxes, not real team-2 predictions')
    infer(p.parse_args())
