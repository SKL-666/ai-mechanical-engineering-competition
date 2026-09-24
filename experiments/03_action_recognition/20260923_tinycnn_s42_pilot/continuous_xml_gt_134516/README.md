# Continuous XML ground-truth box prototype: 134516

仅使用XML挖掘机真值框和完整帧目录，固定64帧历史窗、32帧步长；推理阶段未读取动作标签。action_segments.csv为模型预测的合并动作段，window_predictions.csv记录窗口及输出可用时刻。前63帧暖机，单帧中心输入距决策时刻32帧；不使用未来帧，不倒填预测。output_gaps.csv明确记录无输出范围。

frame_evaluation.json和frame_class_metrics.csv由独立入口事后计算，只评价一致真值且有输出的帧；缺失和暖机覆盖另计。不是预切片分类指标，不含segmental F1或工序正确性。尚未使用2号真实检测跟踪，不能作为真实系统联调证据。

当前交付源代码及SHA256：../../../01_dataset/delivery_code/（实际项目位置见下）。训练时源代码另保存在../action_recognition/code_snapshot/。

交付代码绝对路径：/Users/mac/Project/AI+机械工程大赛/experiments/01_dataset/delivery_code/。运行命令见inference_config.json及项目Docs/Earthmoving_训练入口方案.md。
