#!/usr/bin/env python3
"""Verify raw preservation and delivered predictions after a completed pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from earthmoving.common import read_jsonl, sha256, write_json


def verify(root, derived, run):
    source = read_jsonl(derived / 'source_manifest.jsonl')
    def unchanged(r):
        p = root / r['path']
        return p.exists() and p.stat().st_size == r['bytes'] and sha256(p) == r['sha256']
    with ThreadPoolExecutor(max_workers=6) as pool:
        checks = list(pool.map(unchanged, source))
    changed = [r['path'] for r, ok in zip(source, checks) if not ok]
    assert not changed, f'Original sources changed: {changed}'
    expected = {r['path'] for r in source if r['path'].startswith(('Data/', '指导/'))}
    current = {p.relative_to(root).as_posix() for d in ['Data', '指导'] for p in (root / d).rglob('*') if p.is_file()}
    assert expected == current, 'Source file inventory changed'
    out = run / 'action_recognition'
    cfg = json.loads((out / 'config.yaml').read_text())
    assert cfg['split_sha256'] == sha256(derived / 'split.json')
    for name, digest in cfg['input_hashes'].items():
        assert sha256(derived / name) == digest
    details = read_jsonl(out / 'prediction_details.jsonl')
    test = [r for r in read_jsonl(derived / 'clips.jsonl') if r['split'] == 'test']
    assert len(details) == len(test) == 170
    assert {r['sample_id'] for r in details} == {r['sample_id'] for r in test}
    assert (out / 'predictions.csv').read_bytes() == (run / 'reevaluation/predictions.csv').read_bytes()
    summary = json.loads((out / 'run_summary.json').read_text())
    assert sha256(out / 'best.pt') == summary['checkpoint_sha256']
    continuous = {}
    for p in sorted(run.glob('continuous_xml_gt_*')):
        meta = json.loads((p / 'inference_config.json').read_text())
        with (p / 'window_predictions.csv').open() as f:
            windows = list(csv.DictReader(f))
        for r in windows:
            assert int(r['input_end_frame']) - int(r['input_start_frame']) == 64
            assert int(r['input_start_frame']) % 32 == 0
            assert int(r['input_end_frame']) - 1 == int(r['available_at_frame']) == int(r['start_frame'])
        assert meta['action_labels_read'] is False and meta['action_truth_access_guard']
        continuous[meta['video_id']] = {'windows': len(windows), 'causal_grid_verified': True}
    result = {'raw_files_sha256_verified_unchanged': len(source), 'raw_inventory_unchanged': True,
              'includes_original_zip_and_guidance': True, 'checkpoint_reload_predictions_byte_identical': True,
              'test_prediction_count': len(details), 'continuous': continuous,
              'source_manifest_sha256': sha256(derived / 'source_manifest.jsonl')}
    fixture = run / 'csv_interface_fixture/action_segments.csv'
    if fixture.exists():
        assert fixture.read_bytes() == (run / 'continuous_xml_gt_104154/action_segments.csv').read_bytes()
        result['external_csv_adapter_xml_fixture_matches_direct_xml'] = True
        result['external_csv_fixture_is_real_team2_tracks'] = False
    write_json(root / 'experiments/01_dataset/final_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--derived', type=Path, default=Path('derived/earthmoving_v1'))
    p.add_argument('--run', type=Path, default=Path('experiments/03_action_recognition/20260923_tinycnn_s42_pilot'))
    a = p.parse_args()
    verify(Path(__file__).resolve().parents[1], a.derived, a.run)
