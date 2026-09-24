#!/usr/bin/env python3
"""Run the fixed v2 A/B/C grid and freeze validation selections before test."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from earthmoving.common import sha256, write_json
from joint_preflight.data import verify_indexes

RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v2"
PROTOCOL = ROOT / "configs/joint_training_v2.json"


def event(name, **data):
    item = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": name, **data}
    with (RUN_ROOT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(item, ensure_ascii=False), flush=True)


def main():
    manifest, _ = verify_indexes()
    protocol = json.loads(PROTOCOL.read_text())
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    freeze_path = RUN_ROOT / "freeze.json"
    if freeze_path.exists():
        raise RuntimeError("v2 selection is already frozen")
    runs = []
    for arm in ("A", "B", "C"):
        for seed in protocol["seeds"]:
            key = f"{arm}_s{seed}"
            folder = RUN_ROOT / "runs" / key
            summary_path = folder / "run_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                assert summary["epochs_completed"] == protocol["epochs"]
                assert summary["checkpoint_sha256"] == sha256(folder / "best.pt")
                event("reuse_complete_run", run=key)
            else:
                if folder.exists():
                    raise RuntimeError(f"Incomplete run requires review: {folder}")
                command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/train_joint_v2.py"),
                           "train", "--arm", arm, "--seed", str(seed)]
                event("train_start", run=key, command=command)
                with (RUN_ROOT / f"{key}.log").open("w") as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                summary = json.loads(summary_path.read_text())
                assert summary["epochs_completed"] == protocol["epochs"]
                assert summary["checkpoint_sha256"] == sha256(folder / "best.pt")
                event("train_complete", run=key, selected_epoch=summary["selected_epoch"],
                      val_macro_f1=summary["val_metrics"]["macro_f1"])
            runs.append({"run": key, "selected_epoch": summary["selected_epoch"],
                         "val_metrics": summary["val_metrics"],
                         "checkpoint_sha256": summary["checkpoint_sha256"]})
    assert len(runs) == 9
    write_json(freeze_path, {
        "protocol_sha256": sha256(PROTOCOL),
        "candidate_index_sha256": manifest["train_candidates_sha256"],
        "validation_index_sha256": manifest["frozen_val_reference_sha256"],
        "earthmoving_split_sha256": manifest["original_split_sha256"],
        "test_index_sha256": manifest["original_test_index_sha256"],
        "checkpoint_selection": protocol["validation"],
        "all_train_runs_complete": True, "test_evaluated_at_freeze": False, "runs": runs})
    event("validation_selection_frozen", freeze_sha256=sha256(freeze_path), run_count=9)


if __name__ == "__main__":
    main()
