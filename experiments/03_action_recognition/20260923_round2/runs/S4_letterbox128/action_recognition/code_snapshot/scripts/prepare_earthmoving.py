#!/usr/bin/env python3
"""Freeze a video-disjoint local split BEFORE constructing any sample windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from earthmoving.common import read_jsonl, write_jsonl, write_json, sha256, spans


def prepare(d, window, stride):
    audit = json.loads((d / 'audit.json').read_text())
    raw = read_jsonl(d / 'raw_intervals.jsonl')
    videos = sorted(audit['videos'])
    rng = random.Random(42)
    split = None
    for attempt in range(10000):
        candidate = videos.copy()
        rng.shuffle(candidate)
        groups = {'train': sorted(candidate[:6]), 'val': sorted(candidate[6:8]), 'test': sorted(candidate[8:])}
        supports = {name: Counter(r['label'] for r in raw if r['video_id'] in vs) for name, vs in groups.items()}
        if all(set(c) == set(range(5)) for c in supports.values()) and supports['train'][4] >= 2:
            split = {'version': 'earthmoving_video_v1', 'status': 'locally_frozen_pending_team_ratification',
                     'seed': 42, 'selection': 'first seeded 6/2/2 shuffle with 5-class support and >=2 train Move intervals',
                     'attempt_1based': attempt + 1, 'groups': groups,
                     'raw_interval_support': {k: [v[i] for i in range(5)] for k, v in supports.items()},
                     'raw_intervals_sha256': sha256(d / 'raw_intervals.jsonl'),
                     'source_manifest_sha256': sha256(d / 'source_manifest.jsonl'),
                     'scene_group_status': 'video-disjoint only; 10 videos from 6 projects; project identity unresolved'}
            break
    assert split is not None
    splitfile = d / 'split.json'
    if splitfile.exists():
        assert json.loads(splitfile.read_text()) == split, 'Frozen split differs; create a new version explicitly'
    else:
        write_json(splitfile, split)
    # From here on, every derivative inherits its video's already-frozen membership.
    membership = {v: name for name, vs in split['groups'].items() for v in vs}
    assert len(membership) == len(videos)
    # Exclude ALL occurrences of cross-video identical frames, independently of split.
    # This dataset contains two duplicated first-frame pairs; retain raw originals.
    duplicate_frames = {(v, int(t)) for group in audit['cross_video_exact_duplicates'] for v, t in group['frames']}
    boxes = read_jsonl(d / 'boxes.jsonl')
    valid = {v: np.zeros(audit['videos'][v]['frames'], dtype=bool) for v in videos}
    equipment = {}
    for box in boxes:
        valid[box['video_id']][box['frame']] = True
        equipment[box['video_id']] = box['equipment_id']
    for v, t in duplicate_frames:
        valid[v][t] = False
    labels = {v: np.load(d / f'{v}_consensus.npy') for v in videos}
    clips, windows, discarded = [], [], []
    for row in raw:
        v, c = row['video_id'], row['label']
        # Intersect the SAME raw interval under all four interpretations: [raw_start, raw_end-1).
        lo, hi = max(0, row['raw_start']), min(len(labels[v]), row['raw_end'] - 1)
        runs = spans((labels[v][lo:hi] == c) & valid[v][lo:hi])
        if not runs:
            discarded.append(row)
            continue
        a, b = max(runs, key=lambda x: (x[1] - x[0], -x[0]))
        a, b = a + lo, b + lo
        length = min(window, b - a)
        start = a + (b - a - length) // 2
        base = {**row, 'equipment_id': equipment[v], 'split': membership[v], 'fps': 25,
                'fps_basis': 'publisher_search_excerpt', 'bbox_source': 'xml_gt',
                'frame_dir': f'Data/frames_ce/{v}', 'core_start': a, 'core_end': b,
                'alignment': 'four_hypothesis_consensus_v1'}
        clips.append({**base, 'sample_id': f'{v}_r{row["raw_line"]:03d}_clip',
                      'start_frame': start, 'end_frame': start + length})
        starts = list(range(a, b - window + 1, stride)) if b - a >= window else [a]
        for s in starts:
            windows.append({**base, 'sample_id': f'{v}_r{row["raw_line"]:03d}_f{s:05d}',
                            'start_frame': s, 'end_frame': min(s + window, b)})
    for rows in [clips, windows]:
        assert len({r['sample_id'] for r in rows}) == len(rows)
        for r in rows:
            assert membership[r['video_id']] == r['split']
            assert (labels[r['video_id']][r['start_frame']:r['end_frame']] == r['label']).all()
            assert not any((r['video_id'], t) in duplicate_frames for t in range(r['start_frame'], r['end_frame']))
    write_jsonl(d / 'clips.jsonl', clips)
    write_jsonl(d / 'windows.jsonl', windows)
    write_jsonl(d / 'discarded_intervals.jsonl', discarded)
    summary = {'window_frames': window, 'stride_frames': stride, 'split_sha256': sha256(splitfile),
               'index_sha256': {n: sha256(d / n) for n in ['clips.jsonl', 'windows.jsonl', 'boxes.jsonl']},
               'classification_unit': 'one center window per original interval, longest valid consensus core',
               'train_sampling': 'fixed windows within interval core; class weights from raw training intervals',
               'discarded_intervals': len(discarded), 'cross_split_video_overlap': 0,
               'cross_video_duplicate_raw_frames_excluded': sorted(duplicate_frames),
               'cross_split_exact_duplicate_frames': 0,
               'coverage': {name: {'clips': len([r for r in clips if r['split'] == name]),
                                   'windows': len([r for r in windows if r['split'] == name]),
                                   'clip_class_support': [sum(r['split'] == name and r['label'] == c for r in clips) for c in range(5)]}
                            for name in ['train', 'val', 'test']}}
    assert all(all(v['clip_class_support']) for v in summary['coverage'].values()), 'Lost class after filtering'
    write_json(d / 'index_summary.json', summary)
    print(json.dumps({'split': split, 'index': summary}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--derived', type=Path, default=Path('derived/earthmoving_v1'))
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--stride', type=int, default=32)
    a = p.parse_args()
    assert a.window > 0 and a.stride > 0
    prepare(a.derived, a.window, a.stride)
