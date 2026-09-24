# 挖掘机动作识别 · 本地小演示

已生成[内部审阅 ZIP](../dist/excavator-action-demo_internal-review_2026-09-24.zip)；含独立启动脚本、固定视频与预计算预测，不含权重和训练数据。解压后双击其中的 `打开演示.command`。随包附有使用范围与来源说明，并提供[SHA-256 校验文件](../dist/excavator-action-demo_internal-review_2026-09-24.zip.sha256)。重新打包可运行 `.venv/bin/python scripts/package_demo.py`，脚本不会覆盖现有同名包。

Windows 可使用[静态网页包](../dist/excavator-action-web-demo_internal-review_2026-09-24.zip)：完整解压后双击 `index.html`，用 Edge 或 Chrome 打开，不需要 Python 或本地服务。解压前不要直接在压缩包预览里打开网页，否则相对路径的 `assets/` 可能无法加载。网页源目录为[web_demo](../web_demo/index.html)，[校验文件](../dist/excavator-action-web-demo_internal-review_2026-09-24.zip.sha256)随包另存；打包脚本为 `scripts/package_web_demo.py`。此包同样只供组内审阅。

双击[打开演示.command](打开演示.command)。它在本机启动只监听 `127.0.0.1` 的演示服务并打开浏览器；保持终端窗口运行，按 Ctrl+C 结束。视频和预测数据都在本目录，**无需外网或启动训练**。

演示使用 Earthmoving 验证视频 `142932` 的原始第1500–2249帧，共30秒；片段按可见作业变化选定，没有读取真实动作标签或动作区间。回放暂按25 FPS。A/B/C 切换的是 v2 在测试前按验证片段 Macro-F1 选出的三个权重；页面显示预测时间轴和置信度，不显示真值或准确率。

固定窗口为64帧历史、每8帧更新，前63帧用于暖机。此演示只处理固定镜头里的单台挖掘机，**未接入2号真实检测跟踪**。B/C 用到的 KIT/Rathan 补充来源仍有[对外使用门槛](../Docs/3号_KIT与Rathan语义许可来源核验_2026-09-23.md)；本页限本机研究查看，不作为已放行的比赛展示包。

如需从保留的源帧和权重重新生成：

```bash
cd '/Users/mac/Project/AI+机械工程大赛'
.venv/bin/python demo/build_demo.py
```

构建脚本对动作真值文件启用读取保护，仅生成固定视频片段和模型输出。源视频、选中权重和导出文件的 SHA-256 见[manifest.json](manifest.json)。重建会覆盖本目录派生的 `assets/` 文件，不改 `Data/`、训练索引、权重或历史结果。
