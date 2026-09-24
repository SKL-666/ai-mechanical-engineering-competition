#!/usr/bin/env python3
"""Read-only data-path smoke check before any formal joint training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_preflight.data import ROOT, PRE, RathanStillCandidates, TemporalCandidates, verify_indexes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_tensor(sample, expected_frames: int) -> dict:
    x, label, sample_id = sample
    assert x.shape == (3, expected_frames, 128, 128), (sample_id, x.shape)
    assert x.dtype == torch.float32 and bool(torch.isfinite(x).all())
    assert 0 <= float(x.min()) <= float(x.max()) <= 1
    assert label in range(5)
    return {'sample_id': sample_id, 'class_id': label, 'shape': list(x.shape)}


def main() -> None:
    manifest, caches = verify_indexes()
    train = [json.loads(line) for line in (PRE / 'train_candidates.jsonl').read_text().splitlines()]
    val = [json.loads(line) for line in (PRE / 'frozen_val_reference.jsonl').read_text().splitlines()]
    assert len(train) == 628 and len(val) == 85
    assert Counter(row['source'] for row in train) == {'earthmoving': 224, 'kit': 304, 'rathan': 100}
    assert {row['source'] for row in val} == {'earthmoving'}
    assert not ({r['source_group'] for r in train} & {r['source_group'] for r in val})
    assert {r['source_group'] for r in train if r['source'] == 'rathan'} == {'rathan_source_prefix:BV6'}
    assert {r['source_group'] for r in train if r['source'] == 'kit'} == {'kit_origin_unresolved_ALL'}
    assert all(r['label'] == 0 and not r['temporal_supervision'] for r in train if r['source'] == 'rathan')
    assert not any(r['label'] in {0, 4} for r in train if r['source'] == 'kit')
    assert len(caches['records']) == 404
    for record in caches['records']:
        path = ROOT / record['cache_path']
        if not path.is_file() or sha256(path) != record['sha256']:
            raise RuntimeError(f'Cache hash mismatch: {record["sample_id"]}')
    temporal = TemporalCandidates('train_candidate')
    validation = TemporalCandidates('frozen_val_reference')
    stills = RathanStillCandidates()
    assert len(temporal) == 528 and len(validation) == 85 and len(stills) == 100
    selected = []
    for source in ('earthmoving', 'kit'):
        for label in sorted({r['label'] for r in temporal.rows if r['source'] == source}):
            index = next(i for i, r in enumerate(temporal.rows) if r['source'] == source and r['label'] == label)
            selected.append(check_tensor(temporal[index], 8))
    selected.append(check_tensor(validation[0], 8))
    selected.append(check_tensor(stills[0], 1))
    selected.append(check_tensor(stills[-1], 1))
    result = {
        'status': 'passed_data_path_only_no_model_or_optimizer_run',
        'temporal_train_candidate_rows': len(temporal),
        'static_aux_candidate_rows': len(stills),
        'frozen_validation_reference_rows': len(validation),
        'cache_records_hash_checked': len(caches['records']),
        'sampled_reads': selected,
        'candidate_manifest_sha256': sha256(PRE / 'candidate_manifest.json'),
        'cache_manifest_sha256': sha256(PRE / 'cache_manifest.json'),
        'frozen_split_sha256': manifest['original_split_sha256'],
        'test_index_sha256': manifest['original_test_index_sha256'],
        'test_samples_read': 0,
        'training_steps': 0,
    }
    (PRE / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'temporal_train_candidate_rows',
                                               'static_aux_candidate_rows',
                                               'frozen_validation_reference_rows',
                                               'cache_records_hash_checked')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
