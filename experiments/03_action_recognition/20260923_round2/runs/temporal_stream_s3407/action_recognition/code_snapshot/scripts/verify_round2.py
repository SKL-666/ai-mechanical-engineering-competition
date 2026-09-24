#!/usr/bin/env python3
"""Independent artifact, index, freeze-order and optional original-file verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from earthmoving.common import sha256,read_jsonl,write_json
from earthmoving.round2_data import ROOT,V1,D2,verify_inputs

SUITE=ROOT/'experiments/03_action_recognition/20260923_round2'


def verify(originals):
    manifest=verify_inputs()
    expected=read_jsonl(V1/'clips.jsonl')
    boxframes={(r['video_id'],r['equipment_id'],r['frame']) for r in read_jsonl(V1/'boxes.jsonl')}
    labels={v:np.load(V1/f'{v}_consensus.npy') for vv in manifest['split'].values() for v in vv}
    used_frames={subset:set() for subset in manifest['split']}
    for subset,videos in manifest['split'].items():
        clips=read_jsonl(D2/f'{subset}_clips.jsonl')
        assert clips==[r for r in expected if r['split']==subset]
        for row in read_jsonl(D2/f'{subset}_continuous.jsonl'):
            v=row['video_id'];t=row['target_frame']
            assert v in videos and t==row['end_frame']-1 and row['start_frame']%8==0
            assert row['end_frame']-row['start_frame']==64
            assert labels[v][t]==row['label']>=0
            assert all((v,row['equipment_id'],f) in boxframes for f in range(row['start_frame'],row['end_frame']))
            used_frames[subset].update((v,f) for f in range(row['start_frame'],row['end_frame']))
        for r in clips:
            used_frames[subset].update((r['video_id'],f) for f in range(r['core_start'],r['core_end']))
    duplicates=json.loads((V1/'audit.json').read_text())['cross_video_exact_duplicates']
    for group in duplicates:
        touched={subset for subset,frames in used_frames.items() if any((v,t) in frames for v,t in group['frames'])}
        assert len(touched)<=1,'Duplicate RGB frame used across split in round2'
    frozen=json.loads((SUITE/'freeze.json').read_text())
    assert frozen['protocol_sha256']==sha256(ROOT/'configs/round2_protocol.json')
    events=read_jsonl(SUITE/'events.jsonl')
    first_freeze=next(i for i,e in enumerate(events) if e['event']=='protocol_frozen_before_test')
    first_test=next(i for i,e in enumerate(events) if e['event']=='test_evaluation_start')
    assert first_freeze<first_test
    with (SUITE/'final_per_seed.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==6
    hashes=[];timing={}
    for row in rows:
        out=SUITE/'runs'/row['run']/'action_recognition'
        cfg=json.loads((out/'config.yaml').read_text());run=json.loads((out/'run_summary.json').read_text())
        assert cfg['seed']==int(row['seed']) and cfg['epochs']==40 and run['epochs_completed']==40
        assert cfg['split_sha256']==manifest['parent_split_sha256']
        assert cfg['training_mode']==frozen['selected_training_mode']
        h=sha256(out/'best.pt');assert h==run['checkpoint_sha256'];hashes.append(h)
        preds=read_jsonl(out/'prediction_details.jsonl')
        assert len(preds)==170
        for a,b in zip(preds,[r for r in expected if r['split']=='test']):
            assert all(a[k]==b[k] for k in b)
            assert a['sampled_frames'][-1]==a['end_frame']-1
        with (out/'class_metrics.csv').open() as f:cls=list(csv.DictReader(f))
        assert [int(r['support']) for r in cls]==[10,80,44,35,1]
        for v in manifest['split']['test']:
            p=SUITE/'runs'/row['run']/f'test_continuous_raw_{v}'
            meta=json.loads((p/'inference.json').read_text())
            assert meta['no_action_truth_access_guard'] and meta['checkpoint_sha256']==h
            for r in read_jsonl(p/'probabilities.jsonl'):
                assert set(r)=={'video_id','equipment_id','start_frame','end_frame','available_at_frame','probabilities'}
                assert r['available_at_frame']==r['end_frame']-1 and r['start_frame']%8==0
                prob=np.array(r['probabilities']);assert prob.shape==(5,) and np.isfinite(prob).all() and abs(prob.sum()-1)<1e-5
            pp=json.loads((p/'selected_output/postprocess.json').read_text())
            setting=frozen['postprocess'][row['model']]
            assert pp['stride']==setting['stride'] and pp['alpha']==setting['alpha']
        timing[row['run']]=json.loads((out/'timing.json').read_text())
    assert len(set(hashes))==6,'Final weights unexpectedly reused'
    result={'v1_and_round2_index_hashes_unchanged':True,'v1_clip_evaluation_units_identical':True,
            'no_cross_split_exact_duplicate_frames_in_round2_inputs':True,
            'all_continuous_targets_verified':True,'test_started_after_freeze':True,
            'six_final_runs_and_distinct_checkpoints':True,'test_support':[10,80,44,35,1],
            'all_inference_probability_files_label_free_and_on_fixed_grid':True,
            'postprocess_matches_pretest_freeze':True,'originals_verified':False}
    if originals:
        source=read_jsonl(V1/'source_manifest.jsonl')
        def same(r):
            p=ROOT/r['path']
            return p.exists() and p.stat().st_size==r['bytes'] and sha256(p)==r['sha256']
        with ThreadPoolExecutor(max_workers=6) as pool:checks=list(pool.map(same,source))
        assert all(checks),[r['path'] for r,x in zip(source,checks) if not x]
        known={r['path'] for r in source if r['path'].startswith(('Data/','指导/'))}
        actual={p.relative_to(ROOT).as_posix() for d in ['Data','指导'] for p in (ROOT/d).rglob('*') if p.is_file()}
        assert known==actual
        result.update(originals_verified=True,original_file_count=len(source),raw_inventory_unchanged=True)
    write_json(SUITE/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--originals',action='store_true');a=p.parse_args();verify(a.originals)
