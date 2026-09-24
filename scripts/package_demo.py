#!/usr/bin/env python3
"""Create a self-contained, internal-review ZIP of the precomputed local demo."""
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
DIST = ROOT / "dist"
NAME = "excavator-action-demo_internal-review_2026-09-24"
ARCHIVE_ROOT = "excavator-action-demo"
FILES = {
    "README.md": DEMO / "README_打包版.md",
    "打开演示.command": DEMO / "打开演示.command",
    "index.html": DEMO / "index.html",
    "server.py": DEMO / "server.py",
    "manifest.json": DEMO / "manifest.json",
    "assets/data.js": DEMO / "assets/data.js",
    "assets/poster.jpg": DEMO / "assets/poster.jpg",
    "assets/sample.mp4": DEMO / "assets/sample.mp4",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def entry(name, data, executable=False):
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", (2026, 9, 24, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
    info.compress_type = zipfile.ZIP_STORED if name.endswith((".mp4", ".jpg")) else zipfile.ZIP_DEFLATED
    return info, data


def main():
    demo_manifest = json.loads((DEMO / "manifest.json").read_text())
    if sha256(FILES["assets/sample.mp4"].read_bytes()) != demo_manifest["sample_video_sha256"]:
        raise RuntimeError("Demo video no longer matches its build manifest")
    if sha256(FILES["assets/data.js"].read_bytes()) != demo_manifest["data_js_sha256"]:
        raise RuntimeError("Demo prediction data no longer matches its build manifest")
    if not (FILES["打开演示.command"].stat().st_mode & stat.S_IXUSR):
        raise RuntimeError("Launcher must be executable before packaging")
    manifest = {
        "package": NAME,
        "scope": "offline local internal review; not approved for public distribution",
        "network": "localhost 127.0.0.1 only; no external assets",
        "model_weights_included": False,
        "raw_training_datasets_included": False,
        "action_ground_truth_included": False,
        "source_demo_manifest_sha256": sha256(FILES["manifest.json"].read_bytes()),
        "files": [],
    }
    payload = []
    for name, path in FILES.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Expected regular runtime file: {path}")
        data = path.read_bytes()
        manifest["files"].append({"path": name, "bytes": len(data), "sha256": sha256(data)})
        payload.append(entry(name, data, executable=name.endswith(".command")))
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    payload.append(entry("package_manifest.json", manifest_bytes))
    DIST.mkdir(exist_ok=True)
    target = DIST / f"{NAME}.zip"
    if target.exists():
        raise FileExistsError(f"Existing package preserved: {target}")
    temporary = DIST / f".{NAME}.tmp.zip"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for info, data in payload:
                archive.writestr(info, data)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC verification failed")
            assert len(archive.namelist()) == len(payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    hash_text = f"{sha256(target.read_bytes())}  {target.name}\n"
    target.with_suffix(".zip.sha256").write_text(hash_text)
    print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
                      "entries": len(payload), "sha256": hash_text.split()[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
