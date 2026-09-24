#!/usr/bin/env python3
"""Generate a report/figures from completed real artifacts, never from placeholder metrics."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from earthmoving.common import CLASSES,sha256,write_csv,write_json
from earthmoving.round2_data import ROOT

S=ROOT/'experiments/03_action_recognition/20260923_round2'


def readcsv(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    assert (S/'complete.json').exists() and (S/'verification.json').exists()
    rows=readcsv(S/'final_per_seed.csv');screens=readcsv(S/'screening.csv')
    freeze=json.loads((S/'freeze.json').read_text());verification=json.loads((S/'verification.json').read_text())
    family,classes,video_rows=[],[],[]
    for model in ['single','temporal']:
        group=[r for r in rows if r['model']==model]
        d={'model':model}
        for metric in ['clip_accuracy','clip_precision','clip_recall','clip_macro_f1','continuous_macro_f1','continuous_accuracy','continuous_edit_video_mean','continuous_segment_f1_at_0.5']:
            vals=[float(r[metric]) for r in group];d[metric+'_mean']=float(np.mean(vals));d[metric+'_std']=float(np.std(vals,ddof=1))
        family.append(d)
        each=[readcsv(S/'runs'/r['run']/'action_recognition/class_metrics.csv') for r in group]
        for i,c in enumerate(CLASSES):
            f=[float(rs[i]['f1']) for rs in each]
            classes.append({'model':model,'class':c,'support_per_test':int(each[0][i]['support']),
                            'f1_mean':float(np.mean(f)),'f1_std':float(np.std(f,ddof=1)),
                            'f1_seed42':f[0],'f1_seed3407':f[1],'f1_seed2026':f[2]})
        for v in ['104154','134516']:
            mm=[]
            for r in group:
                table=readcsv(S/'runs'/r['run']/'action_recognition/per_video_metrics.csv')
                mm.append(next(x for x in table if x['video_id']==v))
            video_rows.append({'model':model,'video_id':v,'clips':int(mm[0]['clips']),
                               'accuracy_mean':float(np.mean([float(x['accuracy']) for x in mm])),
                               'macro_f1_mean':float(np.mean([float(x['macro_f1']) for x in mm]))})
    write_csv(S/'class_f1_mean_std.csv',classes);write_csv(S/'per_video_mean.csv',video_rows)
    write_json(S/'summary.json',{'families':family,'frozen_selection':freeze,'verification':verification})
    timing_rows=[]
    for r in rows:
        root=S/'runs'/r['run'];t=json.loads((root/'action_recognition/timing.json').read_text())
        raw=[json.loads((root/f'test_continuous_raw_{v}/inference.json').read_text()) for v in ['104154','134516']]
        timing_rows.append({'run':r['run'],'model':r['model'],'seed':r['seed'],
                           'parameters':json.loads((root/'action_recognition/run_summary.json').read_text())['parameters'],
                           'checkpoint_bytes':(root/'action_recognition/best.pt').stat().st_size,
                           'classification_input_fps':t['input_frames_per_second'],
                           'classification_clips_per_second':t['clips_per_second'],
                           'classification_batch32_amortized_clip_ms':t['end_to_end_seconds']/t['clips']*1000,
                           'continuous_batch1_window_ms':sum(x['seconds'] for x in raw)/sum(x['windows'] for x in raw)*1000,
                           'continuous_windows_per_second':sum(x['windows'] for x in raw)/sum(x['seconds'] for x in raw)})
    write_csv(S/'timing_summary.csv',timing_rows)
    # Preserve a lightweight readable comparison as a standalone scientific figure.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,key,title in zip(axes,['clip_macro_f1','continuous_macro_f1'],['Clip classification','Continuous frame classification']):
        means=[r[key+'_mean']*100 for r in family];sd=[r[key+'_std']*100 for r in family]
        ax.bar(['Last-frame CNN','8-frame causal CNN'],means,yerr=sd,capsize=5,color=['#5177a6','#dc824d'])
        ax.set(ylim=(0,100),ylabel='Macro-F1 (%)',title=title)
        for i,r in enumerate(family):
            ax.text(i,means[i]+sd[i]+2,f'{means[i]:.1f} ± {sd[i]:.1f}',ha='center')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Earthmoving: three independent seeds; XML ground-truth crops')
    fig.savefig(S/'comparison.png',dpi=180);plt.close(fig)
    # Fixed-reference baseline is informational: different budget and input sampling.
    def pct(x):return f'{100*float(x):.2f}%'
    text=['# 3号第二轮改进执行结果','',
          '2026-09-23。已执行既定有限实验协议。全部结果为实际运行产物，仅使用本地Earthmoving和原始5类；无预训练、无补充数据、无人工动作补标。',
          '', '**结论：完成了计划，但未取得稳定的整体提升。** 新时序模型比本轮单帧对照更好，然而片段Macro-F1均值24.84%仍低于上轮pilot的37.33%；连续Macro-F1均值25.79%高于上轮参考21.75%，但标准差达8.72个百分点。不能据此宣布新模型已达标，原pilot产物保留，未用测试分数回改本轮冻结方案。',
          '', '上轮连续参考通过原有两段测试视频的混淆矩阵合并得到，评价帧数同为31,820，见`pilot_continuous_reference.json`。上轮只有一次训练，不能与本轮三种子的标准差做显著性结论。',
          '', '## 执行与选择', '',
          '共完成15次独立训练：7个20epoch采样/预处理对照、4个40epoch主组合和4次额外种子训练。其中6次训练进入冻结后的最终同协议测试。', '',
          f'验证集选择预处理为`{freeze["screen_winner"]}`，最终训练模式为`{freeze["selected_training_mode"]}`。部署候选由seed42验证连续Macro-F1选择为`{freeze["deployment_validation_winner"]["run"]}`，没有按测试分数反选。', '',
          '| 对照 | 验证片段Macro-F1 |','|---|---:|']
    text.extend(f'| {r["experiment"]} | {pct(r["val_macro_f1"])} |' for r in screens)
    text+=['','当前20epoch条件下，S2优于S1，支持先取消这组类别权重；等比补边/放大/增强/联合框未优于被选方案。这些是有限验证集与训练预算下的结果，不代表相关技术普遍无效。',
           '', '## 三种子测试结果', '',
           'seed=42、3407、2026；下表为均值±样本标准差（ddof=1），标准差单位为百分点。它反映固定划分下初始化/采样的随机性，不是跨工地泛化置信区间。同一视频和片段没有跨组，固定170个测试原区间内部窗口；单帧与时序模型读取的帧数不同，但输出对应同一窗口末帧。', '',
           '| 模型 | 片段Accuracy | 片段Macro-F1 | 连续逐帧Macro-F1 |','|---|---:|---:|---:|']
    for r in family:
        cells=[f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}' for k in ['clip_accuracy','clip_macro_f1','continuous_macro_f1']]
        text.append(f'| {r["model"]} | '+ ' | '.join(cells)+' |')
    text+=['','![三种子对比](../experiments/03_action_recognition/20260923_round2/comparison.png)', '',
           '上轮pilot为单次12epoch、中心单帧、不同网络投影和采样策略，片段Accuracy46.47%、Macro-F1 37.33%。本轮与pilot的差异包含训练预算、模型和输入目标变化，不能全部归因于时序。模型效果是否改善应看这里的真实数值，不能把完成实验当作指标一定提高。', '',
           '| 模型/seed | 片段Macro-F1 | 连续逐帧Macro-F1 | 遮蔽段F1@0.5 | 遮蔽Edit |','|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f'| {r["model"]}/{r["seed"]} | {pct(r["clip_macro_f1"])} | {pct(r["continuous_macro_f1"])} | {pct(r["continuous_segment_f1_at_0.5"])} | {pct(r["continuous_edit_video_mean"])} |')
    text+=['', '## 每类与逐视频', '', '| 模型 | 类别 | 每次测试support | F1均值±标准差 |','|---|---|---:|---:|']
    for r in classes:text.append(f'| {r["model"]} | {r["class"]} | {r["support_per_test"]} | {100*r["f1_mean"]:.2f} ± {100*r["f1_std"]:.2f} |')
    text+=['','| 模型 | 视频 | 片段Accuracy均值 | 片段Macro-F1均值 |','|---|---|---:|---:|']
    for r in video_rows:text.append(f'| {r["model"]} | {r["video_id"]} | {pct(r["accuracy_mean"])} | {pct(r["macro_f1_mean"])} |')
    text+=['','Move的测试support始终只有1；3个随机种子反复评价的是同一例，不能当成3条独立测试动作。仍未删除类别、补标签或扩大场景数量。',
           '', '## 连续输出与实验局限', '',
           '每次从固定64帧历史窗输出当前末帧动作，最新图像时滞为0；与pilot相比消除了固定32帧中帧时滞。前63帧暖机仍存在，缺框可能增加初始无输出。输出延迟仍包括决策步长、EMA和实际计算，不能写成零延迟。', '',
           '| 模型 | 冻结步长 | EMA alpha |','|---|---:|---:|']
    for m,r in freeze['postprocess'].items():text.append(f'| {m} | {r["stride"]} | {r["alpha"]} |')
    text+=['','`test_continuous_raw_<video>/probabilities.jsonl`由独立的禁止真值访问进程生成；`selected_output/action_segments.csv`在读取评价真值之前产生。帧/秒均左闭右开，按视频与设备ID组织。所有连续结果是XML真值框原型，未使用2号真实检测/跟踪，未执行4号工序判断。', '',
           '段级F1/Edit是在连续有预测范围内移除未知真值帧后计算的遮蔽版本，不能当作完整精确边界真值的标准成绩。边界结果使用类别对和真值位置区间、16帧容差，详见各视频continuous_metrics.json及[实验协议](3号_第二轮实验协议.md)。', '',
           'TXT端点仍未确认；目标使用四候选一致标签，缺失不补Idle。25FPS有发布方摘要依据，缺本地时间戳验证。原视频划分互斥，不保证工地互斥；验证只有两段视频，多次候选选择有过拟合验证集风险。pilot测试已在此前诊断中查看，本轮不能声称全新盲测。', '',
           '本轮未做8帧无序均值聚合或等参数量对照，因此只能比较整个8帧时序方案与末帧单帧方案，不能严格把全部差异归因于帧顺序。', '',
           '模块计时汇总在[timing_summary.csv](../experiments/03_action_recognition/20260923_round2/timing_summary.csv)：单帧90,277参数，时序127,333参数。分类计时batch32按窗口平均摊销，连续计时batch1；均包含读取/缓存、裁剪、预处理和模型，不包含2号检测跟踪。输入图像FPS与窗口吞吐分别报告，不能把8张输入图片当8次动作决策。256张原图LRU缓存、系统热缓存和单次运行计时也限制其作为真实部署基准的解释。', '',
           '## 核验和交付', '',
           f'原件核验数量：{verification.get("original_file_count","未运行")}。Data、原始ZIP和指导文件按SHA256及清单验证不变；v1与round2索引哈希不变，全部连续末帧目标和历史框检查通过。六份最终checkpoint不同，测试发生在freeze事件之后，全部测试support与原170片段一致。', '',
           '训练运行目录的action_recognition/保存规定的metrics.csv、class_metrics.csv、predictions.csv、confusion_matrix.png、config.yaml、README.md，以及best.pt、环境、采样曝光和训练时代码快照。validation/和test/分开保存。', '',
           '[全部种子结果](../experiments/03_action_recognition/20260923_round2/final_per_seed.csv) · [均值标准差](../experiments/03_action_recognition/20260923_round2/final_mean_std.csv) · [测试前冻结](../experiments/03_action_recognition/20260923_round2/freeze.json) · [核验证据](../experiments/03_action_recognition/20260923_round2/verification.json)', '',
           '## 命令与剩余工作', '',
           '```bash',"cd '/Users/mac/Project/AI+机械工程大赛'",'.venv/bin/python scripts/run_round2.py',
           '.venv/bin/python scripts/verify_round2.py --originals','.venv/bin/python scripts/report_round2.py','```', '',
           '总入口可复用完整且配置一致的运行，拒绝覆盖不完整或配置冲突的训练目录。新实验应使用新版本/运行目录，不改本轮freeze。详细单项命令见[实验协议](3号_第二轮实验协议.md)。', '',
           '下一轮应先检查训练集内部按视频留出的选择稳定性、取得工地分组依据，并检查少数类与裁剪输入的失败模式；不要继续围绕已经反复查看的测试集搜索。新的实验协议需另存版本，保留本轮失败/退步结果。', '',
           '剩余事项：由1/6号认可划分、保守对齐和遮蔽时序指标口径；由2号提供真实挖掘机轨迹做联调；4号评估工序正确性。当前原型不能替代上述团队验证，也没有真实复杂工况或异常工序结果。']
    (ROOT/'Docs/3号_第二轮改进结果_2026-09-23.md').write_text('\n'.join(text)+'\n')
    (S/'README.md').write_text('# Earthmoving round 2\n\n完整报告：[第二轮改进结果](../../../Docs/3号_第二轮改进结果_2026-09-23.md)。\n\nprotocol.json为预先声明的实验范围；screening.csv及validation_continuous_selection.csv只包含验证选择；freeze.json早于本轮测试；final_per_seed.csv及final_mean_std.csv保存全部六次最终测试，未根据测试分数筛选种子。\n\n每个run下的action_recognition/为完整动作分类提交目录，连续段位于test_continuous_raw_*/selected_output/。全部使用XML真值框，真实2号轨迹联调尚未执行。\n')
    print(json.dumps(family,indent=2))


if __name__=='__main__':main()
