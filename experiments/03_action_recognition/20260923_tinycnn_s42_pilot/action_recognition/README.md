# Earthmoving P0/P1 pilot baseline

实际完成 12 epoch、seed=42 的单次独立训练，验证集选择 epoch 12。TinyActionCNN 从零训练，无预训练、无补充数据；不是最终模型，也没有完成三种子重复。

测试单位：170 条原始动作区间各取一个保守内部窗口，XML 真值框，单帧中心采样。训练使用 952 个窗口，不能将窗口数当独立视频或独立 Move 区间数。

测试 Accuracy=0.464706，Macro-F1=0.373291。Precision/Recall/F1 均固定 5 类宏平均、零分母计 0；Move 测试 support=1，无法稳健估计泛化。

TXT 编号/端点仍不确定：只使用四种候选解释一致的内部帧；缺失或冲突不补 Idle。25 FPS 来自论文发布方 Ground truth data 检索摘要，本地没有视频时间戳；时间列以此来源条件化。原始视频互斥，不保证不同工地互斥。

指标 fps=1120.69 是动作输入图像吞吐；实际范围见 timing.json。不含检测/跟踪，不代表整系统实时 FPS。预切片分类不能证明自动动作分段。

权重 best.pt SHA256: `547bf2add6f66d5bdc1e5fd198967b235b978cab8f94c81eb2cfbd2422849c32`。代码无 commit，见 code_snapshot/ 和 code_manifest.json。

运行命令（项目根目录）: `scripts/train_action.py --config configs/baseline_smoke.json --out experiments/03_action_recognition/20260923_tinycnn_s42_pilot/action_recognition`。环境、划分和输入索引哈希见 config.yaml；该文件使用 JSON 形式的合法 YAML 1.2。
