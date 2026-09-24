# 面向无人施工的工序步骤正确性视觉感知方案

本目录包含比赛资料及3号的挖掘机动作识别工作。原5类为 **Idle、Swing Bucket、Load Bucket、Dump、Move**。

## 当前状态

两轮旧实验只用 Earthmoving。三来源 A/B/C 各三种子已完成 v1、v2 两轮本地训练；v2 调整类别与来源抽样后，B/C 在**已复用的 Earthmoving 测试集**上取得探索性片段改善，但 Idle/Move 仍未识别出来。见[联合训练 v2 方案与结果](Docs/3号_联合训练v2改进与结果_2026-09-23.md)。KIT/Rathan 的公开许可、动作语义和来源已进一步[核验](Docs/3号_KIT与Rathan语义许可来源核验_2026-09-23.md)：KIT 尚缺数据及原视频授权依据，Rathan 仅有静态 Idle 框。**原冻结划分未改动**；连续推理仍是 XML 框离线原型，尚待真实轨迹联调。

## 从这里继续

| 目的 | 文档 |
|---|---|
| 当前进度、待办和关键产物 | [3号会话交接](Docs/会话交接_3号任务推进.md) |
| 三来源数据、映射与训练前放行条件 | [联合训练前准备](Docs/3号_三数据集联合训练前准备_2026-09-23.md) |
| 环境、运行命令与输入接口 | [训练与推理入口](Docs/Earthmoving_训练入口方案.md) |
| 第二轮指标与限制 | [第二轮结果](Docs/3号_第二轮改进结果_2026-09-23.md) |
| 三来源训练改进与最新评价 | [联合训练 v2 方案与结果](Docs/3号_联合训练v2改进与结果_2026-09-23.md) |
| 双击打开的本地动作识别小演示 | [演示说明](demo/README.md) |
| 可解压运行的内部审阅演示 | [ZIP 下载](dist/excavator-action-demo_internal-review_2026-09-24.zip) |
| Windows/macOS 可直接用浏览器打开的静态网页 | [网页入口](web_demo/index.html) · [组内审阅 ZIP](dist/excavator-action-web-demo_internal-review_2026-09-24.zip) |
| KIT/Rathan 语义、来源、重复与使用门槛 | [补充数据核验](Docs/3号_KIT与Rathan语义许可来源核验_2026-09-23.md) |
| 职责、5类与提交格式 | [3号任务说明](Docs/3号同学_动作识别任务说明.md) |
| 原始标注与划分依据 | [Earthmoving核查记录](Docs/Earthmoving_本地数据核查记录.md) |

[项目概述](Docs/项目概述.md)和[指导原件](指导/工序步骤正确性视觉感知系统_实验设计与结果提交规范.md)保留项目背景；[历史档案](Docs/archive/README.md)收录已执行计划和旧检索。补充数据的来源哈希见[接收校验](Docs/补充数据接收与校验_2026-09-23.md)，外部候选与访问核验见[数据集检索](Docs/3号_挖掘机动作数据集补充核验_2026-09-23.md)。

`Data/`为 Earthmoving 原件，`Data_KIT/`和`Data_Rathan/`为隔离补充来源，`derived/`为派生索引与缓存，`experiments/`为不可覆盖的实验记录。机器事实以相应 JSON/CSV 为准；第二轮报告由`scripts/report_round2.py`生成。

## Git 仓库范围

仓库保存源码、配置、文档、冻结的 `derived/earthmoving_v1/split.json` 和轻量实验记录。原始数据、补充数据、模型权重、派生缓存、演示媒体及打包文件只保留在本机，不随 Git 推送。克隆仓库后需要按项目文档自行准备这些输入和产物。
