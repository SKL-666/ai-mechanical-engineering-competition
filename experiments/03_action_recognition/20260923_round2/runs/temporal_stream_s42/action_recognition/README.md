# Round 2 training run

本目录从零训练，仅使用Earthmoving train；每轮以同一85个val片段Macro-F1选择最早最佳权重。方案冻结后已执行test评价；根目录指标/CSV来自170个测试片段，validation/仍单独保留验证结果。

采样曝光、配置、训练日志、代码/环境及权重哈希已保存。原始数据及v1划分未变。TXT端点仍不明确，只用保守一致目标；时间列按有来源的25FPS条件化。


连续预测另存在同run的test_continuous_raw_*/selected_output/，仅XML真值框原型；模型配置及推理参数在全局freeze.json测试前冻结。
