#!/usr/bin/env python3
"""Run the frozen A/B/C seed grid, then freeze all validation choices before test."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v1"
sys.path.insert(0, str(ROOT))
from earthmoving.common import sha256, write_json
from joint_preflight.data import verify_indexes


def event(name, **data):
    item = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": name, **data}
    with (RUN_ROOT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(item, ensure_ascii=False), flush=True)


def main():
    manifest, _ = verify_indexes()
    protocol_path = ROOT / "configs/joint_training_v1.json"
    protocol = json.loads(protocol_path.read_text())
    assert sha256(ROOT / "derived/joint_preflight/train_candidates.jsonl") == manifest["train_candidates_sha256"]
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = RUN_ROOT / "freeze.json"
    if frozen.exists():
        raise RuntimeError("Validation selection has already been frozen")
    runs = []
    for arm in ("A", "B", "C"):
        for seed in protocol["seeds"]:
            key = f"{arm}_s{seed}"
            folder = RUN_ROOT / "runs" / key
            summary = folder / "run_summary.json"
            if summary.exists():
                data = json.loads(summary.read_text())
                assert data["epochs_completed"] == protocol["epochs"]
                assert data["checkpoint_sha256"] == sha256(folder / "best.pt")
                event("reuse_complete_run", run=key, checkpoint_sha256=data["checkpoint_sha256"])
            else:
                if folder.exists():
                    raise RuntimeError(f"Incomplete run requires manual review: {folder}")
                command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/train_joint.py"),
                           "train", "--arm", arm, "--seed", str(seed)]
                event("train_start", run=key, command=command)
                log = RUN_ROOT / f"{key}.log"
                with log.open("w") as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                data = json.loads(summary.read_text())
                assert data["epochs_completed"] == protocol["epochs"]
                assert data["checkpoint_sha256"] == sha256(folder / "best.pt")
                event("train_complete", run=key, selected_epoch=data["selected_epoch"],
                      val_macro_f1=data["val_metrics"]["macro_f1"],
                      checkpoint_sha256=data["checkpoint_sha256"])
            runs.append({"run": key, "selected_epoch": data["selected_epoch"],
                         "val_metrics": data["val_metrics"],
                         "checkpoint_sha256": data["checkpoint_sha256"]})
    assert len(runs) == 9
    write_json(frozen, {
        "protocol_sha256": sha256(protocol_path),
        "candidate_index_sha256": manifest["train_candidates_sha256"],
        "validation_index_sha256": manifest["frozen_val_reference_sha256"],
        "earthmoving_split_sha256": manifest["original_split_sha256"],
        "test_index_sha256": manifest["original_test_index_sha256"],
        "checkpoint_selection": protocol["validation"],
        "all_train_runs_complete": True,
        "test_evaluated_at_freeze": False,
        "runs": runs,
    })
    event("validation_selection_frozen", freeze_sha256=sha256(frozen), run_count=len(runs))


if __name__ == "__main__":
    main()
