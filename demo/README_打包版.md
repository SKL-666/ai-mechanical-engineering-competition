# 挖掘机动作识别 Demo（内部审阅版）

本包仅供本地研究与组内审阅。**KIT/Rathan 补充来源的对外使用依据尚未核实，请勿公开传播此包、截图或把它当作已放行的比赛交付版。**

## 打开

macOS：解压后双击 `打开演示.command`，浏览器会打开本机页面。保持终端窗口运行，按 Ctrl+C 结束。需要系统可运行的 Python 3，无需安装第三方依赖或联网。

其他系统：在解压目录运行 `python3 server.py`，打开终端显示的 `http://127.0.0.1:端口/index.html`。服务只监听 `127.0.0.1`；视频支持时间轴跳转。

## 内容与边界

- 30秒固定验证视频片段，播放速度、A/B/C 模型切换、预测动作时间轴和置信度。
- 预测已在原项目中预计算；本包**不含模型权重、原始数据集、训练代码或真值动作标签**，也不能对新视频推理。
- 固定64帧历史窗，每8帧更新；前63帧暖机。未接入真实检测跟踪。回放按发布方资料中的25 FPS，原视频精确时间基准仍待确认。
- A仅用 Earthmoving；B/C 的训练分别加入 KIT / KIT+Rathan。三者均为 v2 验证集预选权重的输出。

## 视频来源与署名

视频片段截取自 Dominic Roberts、Mani Golparvar-Fard 的 [Earthmoving Equipment 数据集](https://data.mendeley.com/datasets/fyw6ps2d2j/1)，Mendeley Data v1，DOI 10.17632/fyw6ps2d2j.1，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本包截取验证视频 `142932` 的原始帧1500–2249，并压缩为H.264 MP4；模型预测与页面由本项目生成。署名不意味着原作者认可这些预测。

Rathan 的 [Productivity of Equipment](https://universe.roboflow.com/rathan/productivity-of-equipment-k83fo) 页面也标 CC BY 4.0；KIT 片段的独立数据许可及源视频权利链尚未查明。本包不包含两套补充来源的原视频或图片。

文件校验见 `package_manifest.json`，演示输入来源及模型选择见 `manifest.json`。
