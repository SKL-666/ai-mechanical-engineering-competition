# P0/P1核查与验证产物

环境实测、全量图像/XML/动作/ZIP核查、派生索引、样本读取、8项合同测试及交付核验已完成。audit.log、prepare.log、smoke.log、tests.log、baseline_console.log、reevaluation.log和continuous_*.log为真实运行日志。

smoke_UNTRAINED_train_examples.csv只验证随机初始化5类输出及CSV表头，不是评价结果。xml_gt_CSV_INTERFACE_FIXTURE.csv只是从XML导出的外部CSV接口夹具，不是2号真实轨迹。

final_verification.json记录Data/原ZIP/指导原件SHA256及清单不变，checkpoint重载预测一致和固定连续网格检查。delivery_code/保存当前源代码，训练执行时的源代码另在对应run的action_recognition/code_snapshot/。

完整结论与可复现命令见 ../../Docs/3号_P0P1推进结果_2026-09-23.md 和 ../../Docs/Earthmoving_训练入口方案.md。
