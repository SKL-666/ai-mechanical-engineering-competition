#!/usr/bin/env python3
"""Fixed stride-8 continuous test evaluation of each arm's val-selected seed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from earthmoving.common import sha256, write_json
from scripts.round2_continuous import pooled

DEFAULT_RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    freeze_path = run_root / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    assert freeze["all_train_runs_complete"] and len(freeze["runs"]) == 9
    split = json.loads((ROOT / "derived/earthmoving_v1/split.json").read_text())
    assert sha256(ROOT / "derived/earthmoving_v1/split.json") == freeze["earthmoving_split_sha256"]
    videos = split["groups"]["test"]
    assert len(videos) == 2
    selection = {}
    for arm in ("A", "B", "C"):
        candidates = [r for r in freeze["runs"] if r["run"].startswith(arm + "_")]
        assert len(candidates) == 3
        best = max(candidates, key=lambda r: r["val_metrics"]["macro_f1"])
        selection[arm] = {"run": best["run"], "val_macro_f1": best["val_metrics"]["macro_f1"],
                          "checkpoint_sha256": best["checkpoint_sha256"]}
    out = run_root / "continuous_test"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "selection_before_continuous_test.json", {
        "rule": "best frozen clip validation Macro-F1 per arm; never test-selected",
        "freeze_sha256": sha256(freeze_path), "selection": selection,
        "stride": 8, "ema_alpha": 1.0, "postprocess_tuning_on_test": False})
    summary = {}
    for arm, selected in selection.items():
        results = []
        for video in videos:
            root = out / arm / video
            infer = root / "inference"
            final = root / "output"
            root.mkdir(parents=True)
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/infer_joint.py"),
                       "--run", selected["run"], "--video", video, "--out", str(infer),
                       "--run-root", str(run_root)]
            with (root / "infer.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/round2_continuous.py"),
                       "--inference", str(infer), "--stride", "8", "--alpha", "1.0",
                       "--out", str(final)]
            with (root / "evaluation.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            results.append(json.loads((final / "continuous_metrics.json").read_text()))
            print(json.dumps({"arm": arm, "run": selected["run"], "video": video,
                              "frame_macro_f1": results[-1]["frame_metrics"]["macro_f1"]}), flush=True)
        summary[arm] = pooled(results)
    write_json(out / "summary.json", {
        "selection": selection, "pooled_test": summary,
        "evaluation": "XML ground-truth track availability, full-frame model input; not team-2 tracking",
        "metric_axis": "masked known-frame protocol inherited from round 2",
        "test_previously_viewed_in_prior_experiments": True,
        "test_based_selection": False})
    print(json.dumps({"continuous_test_complete": summary}), flush=True)


if __name__ == "__main__":
    main()
