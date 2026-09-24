#!/usr/bin/env python3
"""Execute the predeclared finite protocol; all adaptive decisions use validation only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from earthmoving.common import write_json,write_csv,sha256
from earthmoving.round2_data import ROOT,D2,V1,prepare,verify_inputs
from scripts.round2_continuous import process,pooled

PY=ROOT/'.venv/bin/python'
SUITE=ROOT/'experiments/03_action_recognition/20260923_round2'
PROTOCOL=ROOT/'configs/round2_protocol.json'
BASE={'model':'single','frames':1,'seed':42,'epochs':20,'samples_per_epoch':960,'batch_size':32,'lr':.001,
      'device':'mps','sampling':'intervals','training_mode':'core','weighted':True,
      'input_size':96,'letterbox':False,'augment':False,'crop':'per_frame','margin':0.0}


def event(name,**data):
    obj={'utc':datetime.now(timezone.utc).isoformat(),'event':name,**data}
    with (SUITE/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
    print(json.dumps(obj),flush=True)


def execute(args,log):
    with log.open('w') as f:
        p=subprocess.run([str(PY),str(ROOT/'scripts/round2_action.py'),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'Command failed; see {log}:\n'+log.read_text()[-3000:])


def run(name,cfg):
    out=SUITE/'runs'/name/'action_recognition'
    configfile=SUITE/'configs'/f'{name}.json'
    write_json(configfile,cfg)
    if (out/'run_summary.json').exists():
        recorded=json.loads((out/'config.yaml').read_text())
        assert all(recorded[k]==v for k,v in cfg.items()),'Attempt to reuse run with different config'
    else:
        event('training_start',run=name,model=cfg['model'],epochs=cfg['epochs'])
        execute(['train','--config',configfile,'--out',out],SUITE/'logs'/f'{name}.log')
    summary=json.loads((out/'run_summary.json').read_text())
    event('training_complete',run=name,val_macro_f1=summary['val_metrics']['macro_f1'],selected_epoch=summary['selected_epoch'])
    return summary


def raw_inference(name,video,subset):
    out=SUITE/'runs'/name/f'{subset}_continuous_raw_{video}'
    checkpoint=SUITE/'runs'/name/'action_recognition/best.pt'
    if not (out/'inference.json').exists():
        execute(['infer','--checkpoint',checkpoint,'--out',out,'--video',video],SUITE/'logs'/f'{name}_{subset}_{video}.log')
    return out


def main(stop_after_freeze=False):
    SUITE.mkdir(parents=True,exist_ok=True)
    for f in ['configs','logs','runs']:(SUITE/f).mkdir(exist_ok=True)
    protocol=json.loads(PROTOCOL.read_text())
    if (SUITE/'protocol.json').exists():assert json.loads((SUITE/'protocol.json').read_text())==protocol
    else:write_json(SUITE/'protocol.json',protocol)
    prepare();manifest=verify_inputs()
    # Preserve immutable-input checks before model work and the exact protocol hash.
    if not (SUITE/'inputs_at_start.json').exists():
        write_json(SUITE/'inputs_at_start.json',{'protocol_sha256':sha256(PROTOCOL),'derived_manifest_sha256':sha256(D2/'manifest.json'),
                   'split_sha256':sha256(V1/'split.json'),'source_manifest_sha256':sha256(V1/'source_manifest.jsonl')})
    screens=[]
    for name,cfg in [('S0_windows',{**BASE,'sampling':'windows'}),('S1_intervals',BASE),('S2_unweighted',{**BASE,'weighted':False})]:
        result=run(name,cfg);screens.append((name,dict(cfg),result['val_metrics']['macro_f1']))
    loss_winner=max(screens[1:],key=lambda r:r[2]);weighted=loss_winner[1]['weighted']
    for name,change in [('S3_letterbox96',{}),('S4_letterbox128',{'input_size':128}),
                        ('S5_augmentation',{'input_size':128,'augment':True}),
                        ('S6_union',{'input_size':128,'augment':True,'crop':'union','margin':.1})]:
        cfg={**BASE,'weighted':weighted,'letterbox':True,**change}
        result=run(name,cfg);screens.append((name,cfg,result['val_metrics']['macro_f1']))
    winner=max(screens[1:],key=lambda r:r[2])
    write_csv(SUITE/'screening.csv',[{'experiment':n,'val_macro_f1':s,**c} for n,c,s in screens])
    selected={**winner[1],'epochs':40,'sampling':'intervals'}
    event('preprocessing_selected',screen=winner[0],validation_macro_f1=winner[2])
    families={}
    for mode in ['core','stream']:
        for model in ['single','temporal']:
            name=f'{model}_{mode}_s42';cfg={**selected,'model':model,'frames':1 if model=='single' else 8,'training_mode':mode}
            families[name]=cfg;run(name,cfg)
    # Predict full validation videos without labels in separate guarded processes.
    validation=[]
    for name in families:
        paths=[raw_inference(name,v,'val') for v in manifest['split']['val']]
        for stride in [8,16]:
            for alpha in [1.0,.5]:
                score=pooled([process(p,stride,alpha) for p in paths])
                validation.append({'run':name,'stride':stride,'alpha':alpha,**score})
    write_csv(SUITE/'validation_continuous_selection.csv',validation)
    deployment=max(validation,key=lambda r:r['macro_f1'])
    selected_mode=families[deployment['run']]['training_mode']
    postprocess={model:max([r for r in validation if r['run']==f'{model}_{selected_mode}_s42'],key=lambda r:r['macro_f1']) for model in ['single','temporal']}
    finalconfigs={model:{**selected,'model':model,'frames':1 if model=='single' else 8,'training_mode':selected_mode} for model in ['single','temporal']}
    freeze={'protocol_sha256':sha256(PROTOCOL),'split_sha256':sha256(V1/'split.json'),
            'screen_winner':winner[0],'deployment_validation_winner':deployment,'selected_training_mode':selected_mode,
            'final_configs':finalconfigs,'postprocess':postprocess,'seeds':[42,3407,2026],
            'decision_basis':'train and validation only; no round2 test evaluation has run',
            'pilot_test_previously_seen':True,'freeze_created_utc':datetime.now(timezone.utc).isoformat()}
    if (SUITE/'freeze.json').exists():
        old=json.loads((SUITE/'freeze.json').read_text())
        for k in freeze:
            if k!='freeze_created_utc':assert old[k]==freeze[k],f'Frozen selection drift: {k}'
    else:write_json(SUITE/'freeze.json',freeze)
    event('protocol_frozen_before_test',mode=selected_mode,deployment=deployment['run'])
    if stop_after_freeze:return
    for seed in [3407,2026]:
        for model,cfg in finalconfigs.items():run(f'{model}_{selected_mode}_s{seed}',{**cfg,'seed':seed})
    # All six selected trainings now exist; evaluation cannot change selection.
    all_results=[]
    for seed in [42,3407,2026]:
        for model,cfg in finalconfigs.items():
            name=f'{model}_{selected_mode}_s{seed}';out=SUITE/'runs'/name/'action_recognition'
            test=out/'test'
            event('test_evaluation_start',run=name,freeze_sha256=sha256(SUITE/'freeze.json'))
            if not (test/'evaluation.json').exists():
                execute(['evaluate','--checkpoint',out/'best.pt','--out',test,'--subset','test'],SUITE/'logs'/f'{name}_test.log')
            clip=json.loads((test/'evaluation.json').read_text())['metrics']
            setting=postprocess[model];continuous=[]
            for v in manifest['split']['test']:
                path=raw_inference(name,v,'test');dest=path/'selected_output'
                if not (dest/'continuous_metrics.json').exists():
                    process(path,setting['stride'],setting['alpha'],dest)
                continuous.append(json.loads((dest/'continuous_metrics.json').read_text()))
            score=pooled(continuous)
            all_results.append({'run':name,'model':model,'training_mode':selected_mode,'seed':seed,
                **{'clip_'+k:v for k,v in clip.items()},**{'continuous_'+k:v for k,v in score.items()},
                'stride':setting['stride'],'alpha':setting['alpha']})
            # Core required test artifacts are at action_recognition/, in addition to the immutable test subdirectory.
            for f in ['metrics.csv','class_metrics.csv','predictions.csv','prediction_details.jsonl','confusion_matrix.png','confusion_matrix.json','per_video_metrics.csv','timing.json']:
                import shutil
                shutil.copy2(test/f,out/f)
            text=(out/'README.md').read_text().replace('测试尚未执行，验证结果在validation/，不能冒充测试指标。',
                '方案冻结后已执行test评价；根目录指标/CSV来自170个测试片段，validation/仍单独保留验证结果。')
            (out/'README.md').write_text(text+'\n\n连续预测另存在同run的test_continuous_raw_*/selected_output/，仅XML真值框原型；模型配置及推理参数在全局freeze.json测试前冻结。\n')
            event('test_evaluation_complete',run=name,clip_macro_f1=clip['macro_f1'],continuous_macro_f1=score['macro_f1'])
    write_csv(SUITE/'final_per_seed.csv',all_results)
    summary=[]
    for model in ['single','temporal']:
        rr=[r for r in all_results if r['model']==model]
        for metric in ['clip_accuracy','clip_macro_f1','continuous_accuracy','continuous_macro_f1','continuous_edit_video_mean','continuous_segment_f1_at_0.5']:
            x=np.array([r[metric] for r in rr])
            summary.append({'model':model,'training_mode':selected_mode,'metric':metric,'n':len(x),'mean':float(x.mean()),'sample_std':float(x.std(ddof=1))})
    write_csv(SUITE/'final_mean_std.csv',summary)
    write_json(SUITE/'complete.json',{'independent_final_trainings':6,'all_trainings':len(list((SUITE/'runs').glob('*/action_recognition/run_summary.json'))),
               'freeze_sha256':sha256(SUITE/'freeze.json'),'status':'all predeclared experiments and final tests completed'})
    event('suite_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stop-after-freeze',action='store_true')
    main(p.parse_args().stop_after_freeze)
