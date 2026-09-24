# Earthmoving当前训练、评价与推理入口

更新：2026-09-23。本文维护第二轮代码的实际操作方式；初轮命令已移至[初轮运行与复现说明](archive/已完成计划/初轮运行与复现_2026-09-23.md)。当前完成情况见[会话交接](会话交接_3号任务推进.md)，完整指标见[第二轮结果](3号_第二轮改进结果_2026-09-23.md)。

## 环境与输入

只使用项目`Data/`中的Earthmoving原件，类别及输出规范以[3号任务说明](3号同学_动作识别任务说明.md)为准。数据统计、帧对齐、FPS证据以[核查记录](Earthmoving_本地数据核查记录.md)为准。

本机M4 Pro/MPS已实测。项目`.venv/bin/python`使用Python3.11.16/PyTorch2.14.0，通过`--system-site-packages`复用food11父环境，未修改父环境。版本见[依赖锁定清单](../configs/environment.lock.txt)，最小依赖见[requirements.txt](../requirements.txt)。父环境升级可能影响此虚拟环境，复现前核对版本；无需OpenCV/MMAction2/PyYAML。`config.yaml`使用JSON形式的合法YAML1.2。

以下命令均先在项目根目录执行：

```bash
cd '/Users/mac/Project/AI+机械工程大赛'
```

## 已完成实验的复核与重新汇总

```bash
.venv/bin/python scripts/run_round2.py
.venv/bin/python scripts/verify_round2.py --originals
.venv/bin/python scripts/report_round2.py
```

总入口会复用完整且配置一致的运行，拒绝覆盖不完整或配置冲突的训练目录；它可能追加调度事件、重新写汇总，不能将复用当作新训练。`verify_round2.py`检查索引、六份权重、测试前冻结和输出，`--originals`还校验原件。

`report_round2.py`从已有CSV/JSON生成汇总、图表及`Docs/3号_第二轮改进结果_2026-09-23.md`，会重写这些生成产物。该报告保留原路径，人工判断集中在[阶段复盘](3号_阶段复盘与数据扩充判断_2026-09-23.md)；不要直接改生成表格来修饰成绩。冻结JSON与训练时源码快照保持原样。

## 独立复现既定训练与评价

使用尚不存在的新目录，下面的`reproduce_round2_temporal_s42`只是新运行名称：

```bash
.venv/bin/python scripts/round2_action.py train \
  --config experiments/03_action_recognition/20260923_round2/configs/temporal_stream_s42.json \
  --out experiments/03_action_recognition/reproduce_round2_temporal_s42/action_recognition

.venv/bin/python scripts/round2_action.py evaluate \
  --checkpoint experiments/03_action_recognition/reproduce_round2_temporal_s42/action_recognition/best.pt \
  --subset val \
  --out experiments/03_action_recognition/reproduce_round2_temporal_s42/validation_reload
```

训练仅使用train，并按既定85个val片段选最早最佳checkpoint。独立评价前会检查索引哈希。需要评价固定测试集时使用`--subset test`和另一个新输出目录，但必须先冻结方案，不用测试选参数。完整有限对照和三种子安排以[第二轮协议](3号_第二轮实验协议.md)为准；不在本文复制超参数和指标表。

## 连续推理示例：已有XML框原型

以下使用测试前预选的第二轮候选权重，不代表它已稳定达标或已部署。输出目录必须尚不存在。

```bash
.venv/bin/python scripts/round2_action.py infer \
  --checkpoint experiments/03_action_recognition/20260923_round2/runs/temporal_stream_s42/action_recognition/best.pt \
  --video 104154 \
  --out experiments/03_action_recognition/inference_check_104154

.venv/bin/python scripts/round2_continuous.py \
  --inference experiments/03_action_recognition/inference_check_104154 \
  --stride 8 --alpha 0.5 \
  --out experiments/03_action_recognition/inference_check_104154/selected_output
```

`infer`默认读取XML派生框，安装禁止读取动作真值/片段索引的保护，仅按固定历史窗生成`probabilities.jsonl`。第二条命令先产生`action_segments.csv`，再读取本地真值评分；该评价入口仅支持本数据集的挖掘机ID=1，不能当作任意未标注视频的完整服务。`render_predictions`函数提供不读取真值的后处理，但真实输入联调尚未完成。

## 2号轨迹输入与输出位置

`round2_action.py infer`可以用`--tracks <CSV路径>`替换XML框，输入CSV字段如下：

```csv
video_id,equipment_id,frame,x1,y1,x2,y2
```

只传已筛选的挖掘机，帧号从0开始，每设备每帧唯一，框在图像内且面积为正，设备ID在视频内稳定。当前仍从`Data/frames_ce/<video>/<video>_I00000.jpg`读帧，没有相机流或MP4解码器。缺框不插值、不中断后补Idle。多设备或非ID=1的真实轨迹，需要另外明确真值关联后才能调用现有评分入口。

输出动作段字段、左右端点及职责边界以[3号任务说明](3号同学_动作识别任务说明.md)为准。当前历史窗、更新间隔、EMA和延迟解释以[第二轮协议](3号_第二轮实验协议.md)及各输出的`inference.json`、`postprocess.json`为准，不沿用初轮32帧步长/中帧采样说明。

第二轮索引在`derived/earthmoving_round2/`，父划分和原始索引仍在`derived/earthmoving_v1/`。各run的`action_recognition/`保存规定提交文件、权重、环境、训练/采样日志；验证和测试分别在`validation/`、`test/`，完整视频输出在同run的`test_continuous_raw_*/selected_output/`。
