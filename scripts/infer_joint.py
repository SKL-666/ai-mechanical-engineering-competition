#!/usr/bin/env python3
"""Label-blind full-frame continuous inference for a frozen joint checkpoint."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from earthmoving.common import read_jsonl, sha256, write_json, write_jsonl
from earthmoving.round2_model import ActionNet
from joint_preflight.data import letterbox
from scripts.round2_action import guard_truth

DEFAULT_RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v1"


@lru_cache(maxsize=256)
def frame(video, index):
    path = ROOT / "Data/frames_ce" / video / f"{video}_I{index:05d}.jpg"
    with Image.open(path) as image:
        return letterbox(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    freeze_path = run_root / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    selected = {r["run"]: r for r in freeze["runs"]}
    assert freeze["all_train_runs_complete"] and args.run in selected
    checkpoint = run_root / "runs" / args.run / "best.pt"
    assert sha256(checkpoint) == selected[args.run]["checkpoint_sha256"]
    # The guard runs before any video input is opened. Action labels/indices are forbidden.
    sys.addaudithook(guard_truth)
    boxes_path = ROOT / "derived/earthmoving_v1/boxes.jsonl"
    boxes = read_jsonl(boxes_path)
    available = sorted({int(r["frame"]) for r in boxes
                        if str(r["video_id"]) == args.video and str(r["equipment_id"]) == "1"})
    assert available
    files = sorted((ROOT / "Data/frames_ce" / args.video).glob("*.jpg"))
    assert [int(p.stem.split("_I")[1]) for p in files] == list(range(len(files)))
    n = len(files)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ActionNet(temporal=True).to(device)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    tracks = set(available)
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for end in range(64, n + 1, 8):
            if any(t not in tracks for t in range(end - 64, end)):
                continue
            ids = np.linspace(end - 64, end - 1, 8).round().astype(int)
            array = np.stack([frame(args.video, int(i)) for i in ids])
            x = torch.from_numpy(array.transpose(3, 0, 1, 2).copy()).unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32).div_(255)
            probabilities = model(x).softmax(-1)[0].cpu().tolist()
            outputs.append({"video_id": args.video, "equipment_id": "1",
                            "start_frame": end - 64, "end_frame": end,
                            "available_at_frame": end - 1, "probabilities": probabilities})
    elapsed = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "probabilities.jsonl", outputs)
    write_json(args.out / "inference.json", {
        "video_id": args.video, "equipment_ids": ["1"], "frame_count": n,
        "valid_tracks": {"1": available}, "source": "XML_GT_track_availability_full_frame_not_team2",
        "checkpoint_sha256": sha256(checkpoint), "freeze_sha256": sha256(freeze_path),
        "inference_script_sha256": sha256(Path(__file__)),
        "tracks_sha256": sha256(boxes_path), "window_frames": 64, "base_stride": 8,
        "warmup_frames": 63, "latest_sample_age_frames": 0,
        "no_action_truth_access_guard": "active", "seconds": elapsed,
        "windows": len(outputs), "sampled_input_frames_per_second": len(outputs) * 8 / elapsed,
        "timed_scope": "frame JPEG/letterbox, tensor transfer, forward and softmax; excludes detector/tracker",
        "test_labels_or_boundaries_read": False})
    print(json.dumps({"run": args.run, "video": args.video,
                      "windows": len(outputs), "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
