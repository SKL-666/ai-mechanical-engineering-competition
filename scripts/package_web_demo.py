#!/usr/bin/env python3
"""Build a portable static webpage and ZIP; no local server is required."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demo"
WEB = ROOT / "web_demo"
DIST = ROOT / "dist"
PACKAGE = "excavator-action-web-demo_internal-review_2026-09-24"
FILES = (
    "index.html",
    "assets/data.js",
    "assets/poster.jpg",
    "assets/sample.mp4",
)

README = """挖掘机动作识别网页演示（组内审阅版）

Windows：先完整解压 ZIP，再双击 index.html，用 Edge 或 Chrome 打开。
macOS/Linux：完整解压后用浏览器打开 index.html。
保持 assets 文件夹与 index.html 在同一目录层级。无需 Python、联网或安装依赖。

这是固定 30 秒视频的预计算 A/B/C 模型预测，仅供研究查看。不能上传新视频推理；包内没有模型权重、训练数据或真值动作标注。
KIT/Rathan 补充来源的对外使用依据尚未核实，请勿公开传播本包、页面或截图，也不要作为已放行的比赛展示。

视频摘自 Dominic Roberts、Mani Golparvar-Fard 的 Earthmoving Equipment 数据集，Mendeley Data v1，DOI 10.17632/fyw6ps2d2j.1，CC BY 4.0：
https://data.mendeley.com/datasets/fyw6ps2d2j/1
此处截取验证视频 142932 的原始第 1500–2249 帧并压缩为 MP4；模型预测和页面由本项目制作。原作者不为模型预测背书。

文件哈希与使用范围见 package_manifest.json。
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, key in (("assets/data.js", "data_js_sha256"),
                      ("assets/sample.mp4", "sample_video_sha256")):
        if sha256(SOURCE / name) != source_manifest[key]:
            raise RuntimeError(f"Source file does not match demo manifest: {name}")
    if WEB.exists() or (DIST / f"{PACKAGE}.zip").exists():
        raise FileExistsError("Existing static webpage/package preserved; choose a new version")

    WEB.mkdir()
    (WEB / "assets").mkdir()
    for name in FILES:
        shutil.copyfile(SOURCE / name, WEB / name)
    page = WEB / "index.html"
    html = page.read_text()
    if "请先运行 build_demo.py" not in html:
        raise RuntimeError("Missing expected missing-data message")
    page.write_text(html.replace("请先运行 build_demo.py", "请完整解压网页包并保留 assets 文件夹"))
    (WEB / "README.txt").write_text(README)

    manifest = {
        "package": PACKAGE,
        "scope": "offline local internal review; not approved for public distribution",
        "entrypoint": "index.html; open directly with a browser, including file://",
        "server_required": False,
        "network_required": False,
        "model_weights_included": False,
        "raw_training_datasets_included": False,
        "action_ground_truth_included": False,
        "files": [
            {"path": name, "bytes": (WEB / name).stat().st_size, "sha256": sha256(WEB / name)}
            for name in (*FILES, "README.txt")
        ],
    }
    (WEB / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    DIST.mkdir(exist_ok=True)
    target = DIST / f"{PACKAGE}.zip"
    try:
        with zipfile.ZipFile(target, "x") as archive:
            for name in (*FILES, "README.txt", "package_manifest.json"):
                path = WEB / name
                method = zipfile.ZIP_STORED if path.suffix in (".mp4", ".jpg") else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=f"excavator-action-web-demo/{name}", compress_type=method)
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC verification failed")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    digest = sha256(target)
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n")
    print(json.dumps({"page": str(page), "archive": str(target), "bytes": target.stat().st_size,
                      "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
