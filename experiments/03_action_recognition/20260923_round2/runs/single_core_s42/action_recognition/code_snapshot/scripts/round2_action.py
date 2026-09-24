#!/usr/bin/env python3
"""Separate train/clip-eval/label-free-infer commands for round 2."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from earthmoving.common import CLASSES, sha256, write_json, write_csv, write_jsonl, read_jsonl, spans
from earthmoving.data import box_lookup
from earthmoving.round2_data import ROOT,V1,D2,Round2Dataset,count_groups,verify_inputs,load_input,frame_ids,source_image
from earthmoving.round2_model import ActionNet
from earthmoving.metrics import classification_metrics,save_confusion
from scripts.train_action import evaluate,sync


def init(cfg):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    return ActionNet(cfg['model']=='temporal').to(cfg['device'])


def save_predictions(model,data,cfg,out):
    result = evaluate(model,DataLoader(data,batch_size=cfg['batch_size'],shuffle=False,num_workers=0),cfg['device'])
    metrics,classes,cm,values,timing=result
    write_csv(out/'metrics.csv',[{'model':cfg['model'],**metrics,'fps':timing['input_frames_per_second']}])
    write_csv(out/'class_metrics.csv',classes)
    write_json(out/'confusion_matrix.json',cm.tolist());save_confusion(out/'confusion_matrix.png',cm)
    timing['raw_image_lru_cache_entries']=256
    timing['batch_size']=cfg['batch_size'];timing['sampled_frames_per_clip']=cfg['frames']
    write_json(out/'timing.json',timing)
    rows,details=[],[]
    for i,y,p,confidence in values:
        r=data.rows[i]
        rows.append({'video_id':r['video_id'],'start_time':r['start_frame']/25,'end_time':r['end_frame']/25,
                     'true_action':CLASSES[y],'pred_action':CLASSES[p],'confidence':confidence})
        details.append({**r,'true_id':y,'pred_id':p,'confidence':confidence,'sampled_frames':frame_ids(r,cfg['frames']).tolist()})
    write_csv(out/'predictions.csv',rows);write_jsonl(out/'prediction_details.jsonl',details)
    per_video=[]
    for v in sorted({r['video_id'] for r in details}):
        rr=[r for r in details if r['video_id']==v]
        mm,cc,_=classification_metrics([r['true_id'] for r in rr],[r['pred_id'] for r in rr])
        per_video.append({'video_id':v,'clips':len(rr),**mm})
    write_csv(out/'per_video_metrics.csv',per_video)
    return metrics


def train(cfg,out):
    manifest=verify_inputs()
    out.mkdir(parents=True,exist_ok=False)
    model=init(cfg)
    cfg={**cfg,'classes':CLASSES,'pretrained':False,'data':'Earthmoving only',
         'split_sha256':manifest['parent_split_sha256'],'index_manifest_sha256':sha256(D2/'manifest.json'),
         'command':' '.join(sys.argv),'fps':25,'fps_basis':'publisher search excerpt; local cadence not independently verified',
         'alignment':'four_hypothesis_consensus_v1','code_commit':None,'deterministic_algorithms':True}
    write_json(out/'config.yaml',cfg)
    code=[]
    for folder in ['earthmoving','scripts','tests','configs']:
        for p in (ROOT/folder).glob('*'):
            if p.is_file() and p.suffix in ['.py','.json','.txt']:
                dest=out/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dest);code.append({'path':str(p.relative_to(ROOT)),'sha256':sha256(p)})
    write_json(out/'code_manifest.json',code)
    shutil.copy2(ROOT/'experiments/01_dataset/environment.json',out/'environment.json')
    shutil.copy2(ROOT/'configs/environment.lock.txt',out/'environment.lock.txt')
    shutil.copy2(V1/'split.json',out/'split.json')
    data=Round2Dataset('train',cfg,True);val=Round2Dataset('val',cfg)
    counts=np.array(count_groups(data))
    weights=1/np.sqrt(counts) if cfg['weighted'] else np.ones(5)
    weights/=weights.mean()
    lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=cfg['device']))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=1e-5)
    loader=DataLoader(data,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    vl=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    logs,exposures=[],[];best=-1;started=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        data.set_epoch(epoch)
        ec=Counter((r['video_id'],r['raw_line'],r['label']) for r,_ in data.schedule)
        exposures.extend({'epoch':epoch,'video_id':v,'raw_line':line,'class_id':c,'draws':n} for (v,line,c),n in sorted(ec.items()))
        model.train();total=0;n=0
        for x,y,_ in loader:
            x=x.to(cfg['device']);y=y.to(cfg['device'])
            opt.zero_grad(set_to_none=True);logits=model(x)
            assert logits.shape==(len(y),5)
            loss=lossfn(logits,y);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            total+=loss.item()*len(y);n+=len(y)
        score,_,_,_,_=evaluate(model,vl,cfg['device'])
        logs.append({'epoch':epoch,'train_loss_batch_weighted_mean':total/n,'lr':opt.param_groups[0]['lr'],
                     **{'val_'+k:v for k,v in score.items()},'elapsed_seconds':time.perf_counter()-started})
        write_csv(out/'training_log.csv',logs)
        if score['macro_f1']>best:
            best=score['macro_f1'];selected_epoch=epoch
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,
                        'selected_epoch':epoch,'val_macro_f1':best},out/'best.pt')
        scheduler.step()
        if epoch%5==0:print(json.dumps({'run':out.parent.name,**logs[-1]}),flush=True)
    duration=time.perf_counter()-started
    write_csv(out/'sampling_exposure.csv',exposures)
    write_json(out/'sampling_summary.json',{'groups_per_class':counts.tolist(),'loss_weights':weights.tolist(),
              'draws_per_epoch':cfg['samples_per_epoch'],'sampling':cfg['sampling'],
              'training_mode':cfg['training_mode'],'all_original_train_intervals':224,
              'available_source_rows':len(data.rows),'groups':len(data.groups)})
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location='cpu')['state_dict'])
    vo=out/'validation';vo.mkdir()
    metrics=save_predictions(model,val,cfg,vo)
    assert abs(metrics['macro_f1']-best)<1e-10
    summary={'val_metrics':metrics,'selected_epoch':selected_epoch,'training_seconds':duration,
             'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha256(out/'best.pt'),
             'test_evaluated':False,'epochs_completed':cfg['epochs']}
    write_json(out/'run_summary.json',summary)
    (out/'README.md').write_text('# Round 2 training run\n\n本目录从零训练，仅使用Earthmoving train；每轮以同一85个val片段Macro-F1选择最早最佳权重。测试尚未执行，验证结果在validation/，不能冒充测试指标。\n\n采样曝光、配置、训练日志、代码/环境及权重哈希已保存。原始数据及v1划分未变。TXT端点仍不明确，只用保守一致目标；时间列按有来源的25FPS条件化。\n')
    print(json.dumps({'complete':str(out),'validation':best}),flush=True)


def evaluate_checkpoint(checkpoint,out,subset):
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu');cfg=saved['config']
    assert sha256(D2/'manifest.json')==cfg['index_manifest_sha256'];verify_inputs()
    model=init(cfg);model.load_state_dict(saved['state_dict'])
    out.mkdir(parents=True,exist_ok=False)
    metrics=save_predictions(model,Round2Dataset(subset,cfg),cfg,out)
    write_json(out/'evaluation.json',{'subset':subset,'checkpoint_sha256':sha256(checkpoint),'metrics':metrics,
                                    'config':cfg,'command':' '.join(sys.argv)})
    print(metrics)


def guard_truth(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if ('/Labels/' in path or '_consensus.npy' in path or
                path.endswith(('_clips.jsonl','_windows.jsonl','_continuous.jsonl','raw_intervals.jsonl','clips.jsonl','windows.jsonl'))):
            raise RuntimeError('Continuous inference attempted to read action truth: '+path)


def infer(checkpoint,out,video,tracks=None):
    # Separate process: this guard persists and cannot accidentally be disabled by an evaluator.
    sys.addaudithook(guard_truth)
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu');cfg=saved['config']
    model=init(cfg);model.load_state_dict(saved['state_dict']);model.eval()
    if tracks:
        with open(tracks) as f:rr=list(csv.DictReader(f))
        source='external_csv_origin_unverified'
    else:
        tracks=V1/'boxes.jsonl';rr=read_jsonl(tracks);source='XML_GT_PROTOTYPE_NOT_TEAM2'
    rr=[r for r in rr if str(r['video_id'])==video]
    boxes=box_lookup(rr);ids=sorted({str(r['equipment_id']) for r in rr})
    assert ids,'No requested excavator tracks'
    files=sorted((ROOT/'Data/frames_ce'/video).glob('*.jpg'))
    assert [int(p.stem.split('_I')[1]) for p in files]==list(range(len(files)))
    n=len(files);out.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        for _ in range(3):model(torch.zeros(1,3,cfg['frames'],cfg['input_size'],cfg['input_size'],device=cfg['device']))
    sync(cfg['device']);started=time.perf_counter();rows=[]
    for eid in ids:
        for end in range(64,n+1,8):
            if any((video,eid,t) not in boxes for t in range(end-64,end)):continue
            r={'video_id':video,'equipment_id':eid,'start_frame':end-64,'end_frame':end}
            x=load_input(r,boxes,cfg).unsqueeze(0).to(cfg['device'])
            with torch.inference_mode():prob=model(x).softmax(-1)[0].cpu().tolist()
            rows.append({**r,'available_at_frame':end-1,'probabilities':prob})
    sync(cfg['device']);elapsed=time.perf_counter()-started
    write_jsonl(out/'probabilities.jsonl',rows)
    # Track availability is permitted; it never supplies an action boundary or class.
    write_json(out/'inference.json',{'video_id':video,'equipment_ids':ids,'frame_count':n,'source':source,
                'checkpoint_sha256':sha256(checkpoint),'tracks_sha256':sha256(tracks),'config':cfg,
                'window_frames':64,'base_stride':8,'latest_sample_age_frames':0,'warmup_frames':63,
                'no_action_truth_access_guard':'active; action label/index reads raise',
                'causal_offline_replay':True,'seconds':elapsed,'windows':len(rows),'batch_size':1,
                'sampled_input_frames_per_second':len(rows)*cfg['frames']/elapsed,
                'timed_scope':'track checks, JPEG/cache, crop/resize, transfer, forward/softmax and output collection; excludes detector/tracker and file writing',
                'valid_tracks':{eid:[t for t in range(n) if (video,eid,t) in boxes] for eid in ids},
                'command':' '.join(sys.argv)})
    print(json.dumps({'inferred':video,'windows':len(rows),'seconds':elapsed}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--config',type=Path,required=True);t.add_argument('--out',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',type=Path,required=True);e.add_argument('--out',type=Path,required=True);e.add_argument('--subset',choices=['val','test'],required=True)
    i=sub.add_parser('infer');i.add_argument('--checkpoint',type=Path,required=True);i.add_argument('--out',type=Path,required=True);i.add_argument('--video',required=True);i.add_argument('--tracks',type=Path)
    a=p.parse_args()
    if a.command=='train':train(json.loads(a.config.read_text()),a.out)
    elif a.command=='evaluate':evaluate_checkpoint(a.checkpoint,a.out,a.subset)
    else:infer(a.checkpoint,a.out,a.video,a.tracks)
