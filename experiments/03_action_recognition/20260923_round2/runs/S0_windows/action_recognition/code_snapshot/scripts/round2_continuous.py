#!/usr/bin/env python3
"""Prediction-only postprocessing, followed by explicitly separate masked GT evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from earthmoving.common import CLASSES,read_jsonl,write_json,write_csv,spans,sha256
from earthmoving.round2_data import V1
from earthmoving.round2_metrics import temporal_metrics
from scripts.infer_continuous import merge_segments


def render_predictions(meta,rows,stride,alpha):
    assert stride in [8,16] and 0<alpha<=1
    n=meta['frame_count'];outputs=[];dense={}
    for eid in meta['equipment_ids']:
        y=np.full(n,-1,dtype=int);previous=None;last_emit=None
        track=set(meta['valid_tracks'][eid])
        for r in rows:
            if r['equipment_id']!=eid or (r['available_at_frame']-63)%stride:continue
            emit=r['available_at_frame'];stop=min(n,emit+stride)
            stop=next((t for t in range(emit,stop) if t not in track),stop)
            prob=np.asarray(r['probabilities'])
            # Reset after every missing decision or track gap.
            if previous is None or last_emit+stride!=emit:smoothed=prob
            else:smoothed=alpha*prob+(1-alpha)*previous
            pred=int(smoothed.argmax())
            y[emit:stop]=pred
            outputs.append({'video_id':meta['video_id'],'equipment_id':eid,'start_frame':emit,'end_frame':stop,
                            'start_time':emit/25,'end_time':stop/25,'pred_action':CLASSES[pred],
                            'confidence':float(smoothed[pred])})
            previous=smoothed if stop==min(n,emit+stride) else None
            last_emit=emit
        dense[eid]=y
    return merge_segments(outputs),dense


def process(path,stride,alpha,out=None):
    meta=json.loads((path/'inference.json').read_text());rows=read_jsonl(path/'probabilities.jsonl')
    segments,dense=render_predictions(meta,rows,stride,alpha)
    if out:
        out.mkdir(parents=True,exist_ok=False)
        write_csv(out/'action_segments.csv',segments,['video_id','equipment_id','start_frame','end_frame','start_time','end_time','pred_action','confidence'])
        gaps=[{'equipment_id':eid,'start_frame':a,'end_frame':b,'status':'warmup_or_track_gap'} for eid,y in dense.items() for a,b in spans(y<0)]
        write_csv(out/'output_gaps.csv',gaps,['equipment_id','start_frame','end_frame','status'])
        write_json(out/'postprocess.json',{'stride':stride,'alpha':alpha,'smoothing':'causal EMA of class probabilities; reset on gaps',
                   'nominal_ema_mean_age_frames':(1-alpha)/alpha*stride,'mean_age_is_not_bound_on_decision_delay':True,
                   'warmup_frames':63,'latest_sample_age_frames':0,'probabilities_sha256':sha256(path/'probabilities.jsonl'),
                   'segments_created_before_reading_truth':True,'source':meta['source']})
    # Only now load truth. It never changes any returned/exported segment.
    assert meta['equipment_ids']==['1'],'GT evaluator supports only dataset excavator ID=1'
    truth=np.load(V1/f'{meta["video_id"]}_consensus.npy')
    result=temporal_metrics(truth,dense['1'])
    result.update(video_id=meta['video_id'],stride=stride,alpha=alpha,source=meta['source'])
    if out:
        write_json(out/'continuous_metrics.json',result)
        write_csv(out/'class_metrics.csv',result['class_metrics'])
    return result


def pooled(results):
    from earthmoving.metrics import classification_metrics
    cm=sum((np.asarray(r['confusion']) for r in results),np.zeros((5,5),dtype=int))
    tp=cm.diagonal();pr=np.divide(tp,cm.sum(0),out=np.zeros(5),where=cm.sum(0)>0)
    re=np.divide(tp,cm.sum(1),out=np.zeros(5),where=cm.sum(1)>0)
    f=np.divide(2*pr*re,pr+re,out=np.zeros(5),where=pr+re>0)
    counts={t:sum((np.array(r['segment_counts'][t]) for r in results),np.zeros(3,dtype=int)) for t in ['0.1','0.25','0.5']}
    return {'accuracy':float(tp.sum()/max(1,cm.sum())),'precision':float(pr.mean()),'recall':float(re.mean()),
            'macro_f1':float(f.mean()),'evaluated_frames':int(cm.sum()),
            'coverage':sum(r['evaluated_frames'] for r in results)/sum(r['valid_truth_frames'] for r in results),
            'edit_video_mean':float(np.mean([r['edit'] for r in results])),
            **{f'segment_f1_at_{t}':float(2*v[0]/max(1,2*v[0]+v[1]+v[2])) for t,v in counts.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inference',type=Path,required=True);p.add_argument('--stride',type=int,required=True)
    p.add_argument('--alpha',type=float,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(process(a.inference,a.stride,a.alpha,a.out)))
