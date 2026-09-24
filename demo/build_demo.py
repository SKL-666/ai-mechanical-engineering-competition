#!/usr/bin/env python3
"""Build a local, label-blind 30-second action-recognition showcase."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio_ffmpeg
import numpy as np
from PIL import Image
import torch

from earthmoving.round2_model import ActionNet
from joint_preflight.data import letterbox
from scripts.round2_action import guard_truth

DEMO = Path(__file__).resolve().parent
ASSETS = DEMO / "assets"
RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v2"
VIDEO_ID = "142932"
START_FRAME = 1500
FRAME_COUNT = 750
FPS = 25
STRIDE = 8
HISTORY = 64
CLASSES = ["Idle", "Swing Bucket", "Load Bucket", "Dump", "Move"]
MODEL_INFO = {
    "A": {"label": "A · Earthmoving", "description": "仅主数据集"},
    "B": {"label": "B · + KIT", "description": "加入 KIT 视频"},
    "C": {"label": "C · + KIT + Rathan", "description": "再加入 Rathan 单帧辅助"},
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_video():
    ASSETS.mkdir(parents=True, exist_ok=True)
    source = ROOT / "Data/frames_ce" / VIDEO_ID / f"{VIDEO_ID}_I%05d.jpg"
    target = ASSETS / "sample.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(FPS), "-start_number", str(START_FRAME),
        "-i", str(source), "-frames:v", str(FRAME_COUNT),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(target),
    ]
    subprocess.run(command, check=True)
    with Image.open(ROOT / "Data/frames_ce" / VIDEO_ID / f"{VIDEO_ID}_I{START_FRAME:05d}.jpg") as image:
        image.save(ASSETS / "poster.jpg", quality=92)
    return target


def read_frames():
    frames = []
    for i in range(START_FRAME, START_FRAME + FRAME_COUNT):
        path = ROOT / "Data/frames_ce" / VIDEO_ID / f"{VIDEO_ID}_I{i:05d}.jpg"
        with Image.open(path) as image:
            frames.append(letterbox(image))
    array = np.stack(frames)
    assert array.shape == (FRAME_COUNT, 128, 128, 3)
    return array


def make_windows(frames):
    ends = list(range(HISTORY, FRAME_COUNT + 1, STRIDE))
    windows = []
    for end in ends:
        ids = np.linspace(end - HISTORY, end - 1, 8).round().astype(int)
        windows.append(frames[ids])
    return ends, np.stack(windows)


def predict(checkpoint, windows, device):
    model = ActionNet(temporal=True).to(device)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(windows), 16):
            chunk = windows[start:start + 16]
            x = torch.from_numpy(chunk.transpose(0, 4, 1, 2, 3).copy())
            x = x.to(device=device, dtype=torch.float32).div_(255)
            results.extend(model(x).softmax(-1).cpu().tolist())
    return results


def merge_segments(decisions):
    segments = []
    for i, row in enumerate(decisions):
        start = row["at_frame"]
        end = decisions[i + 1]["at_frame"] if i + 1 < len(decisions) else FRAME_COUNT
        label = row["label"]
        confidence = row["probabilities"][label]
        if segments and segments[-1]["label"] == label and segments[-1]["end_frame"] == start:
            current = segments[-1]
            current["end_frame"] = end
            current["confidence_sum"] += confidence
            current["windows"] += 1
        else:
            segments.append({"start_frame": start, "end_frame": end, "label": label,
                             "confidence_sum": confidence, "windows": 1})
    for row in segments:
        row["mean_confidence"] = round(row.pop("confidence_sum") / row["windows"], 6)
    return segments


def main():
    # The guard forbids reading action TXT, consensus arrays, or clip truth indices.
    sys.addaudithook(guard_truth)
    freeze_path = RUN_ROOT / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    selected = json.loads((RUN_ROOT / "continuous_test/selection_before_continuous_test.json").read_text())
    assert selected["freeze_sha256"] == digest(freeze_path)
    assert freeze["all_train_runs_complete"]
    for arm in MODEL_INFO:
        assert selected["selection"][arm]["run"] in {r["run"] for r in freeze["runs"]}

    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    video = export_video()
    ends, windows = make_windows(read_frames())
    models = {}
    for arm, info in MODEL_INFO.items():
        chosen = selected["selection"][arm]
        checkpoint = RUN_ROOT / "runs" / chosen["run"] / "best.pt"
        assert digest(checkpoint) == chosen["checkpoint_sha256"]
        probabilities = predict(checkpoint, windows, device)
        decisions = []
        for end, probs in zip(ends, probabilities):
            rounded = [round(float(p), 6) for p in probs]
            decisions.append({"at_frame": end - 1, "label": int(np.argmax(probs)),
                              "probabilities": rounded})
        models[arm] = {
            **info,
            "run": chosen["run"],
            "checkpoint_sha256": chosen["checkpoint_sha256"],
            "val_macro_f1": chosen["val_macro_f1"],
            "decisions": decisions,
            "segments": merge_segments(decisions),
        }
        print(f"{arm}: {chosen['run']}, {len(decisions)} decisions, {len(models[arm]['segments'])} segments")

    payload = {
        "video": {"id": VIDEO_ID, "split": "Earthmoving validation", "start_frame": START_FRAME,
                  "frame_count": FRAME_COUNT, "fps_assumed": FPS, "duration_seconds": FRAME_COUNT / FPS,
                  "file": "assets/sample.mp4"},
        "classes": CLASSES,
        "history_frames": HISTORY,
        "stride_frames": STRIDE,
        "default_arm": "C",
        "models": models,
        "notice": "本地研究演示；仅模型预测，不展示动作真值；B/C 补充来源未完成对外授权核验。",
    }
    data_file = ASSETS / "data.js"
    data_file.write_text("window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    manifest = {
        "video_source": f"Data/frames_ce/{VIDEO_ID}/{VIDEO_ID}_Ixxxxx.jpg",
        "fixed_frame_range": [START_FRAME, START_FRAME + FRAME_COUNT],
        "selection": "fixed visible interval chosen for motion, without reading action labels or boundaries",
        "model_selection": "v2 continuous_test/selection_before_continuous_test.json; validation-only seed choice",
        "freeze_sha256": digest(freeze_path),
        "sample_video_sha256": digest(video),
        "data_js_sha256": digest(data_file),
        "build_script_sha256": digest(Path(__file__)),
        "no_action_truth_access_guard": "active",
        "no_data_upload": True,
    }
    (DEMO / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"built": str(DEMO), "video_seconds": FRAME_COUNT / FPS,
                      "video_bytes": video.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
