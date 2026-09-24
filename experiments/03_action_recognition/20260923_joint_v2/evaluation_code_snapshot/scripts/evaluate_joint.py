#!/usr/bin/env python3
"""Evaluate nine frozen joint runs on the unchanged Earthmoving clip test set."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from earthmoving.common import sha256, write_csv, write_json
from earthmoving.common import CLASSES
from earthmoving.round2_data import verify_inputs
from joint_preflight.data import earthmoving_input, verify_indexes
from scripts.train_joint import RUN_ROOT as DEFAULT_RUN_ROOT, PROTOCOL as DEFAULT_PROTOCOL, init, validate


def plot_confusion(path, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 6.3), layout="constrained")
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted action", ylabel="True action",
           title="Earthmoving clip classification: full-frame input")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    protocol = args.protocol.resolve()
    freeze_path = run_root / "freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("All training runs must finish and freeze before test access")
    freeze = json.loads(freeze_path.read_text())
    manifest, _ = verify_indexes()
    verify_inputs()
    assert freeze["all_train_runs_complete"] and not freeze["test_evaluated_at_freeze"]
    assert len(freeze["runs"]) == 9
    assert freeze["protocol_sha256"] == sha256(protocol)
    assert freeze["candidate_index_sha256"] == manifest["train_candidates_sha256"]
    assert freeze["validation_index_sha256"] == manifest["frozen_val_reference_sha256"]
    assert freeze["earthmoving_split_sha256"] == manifest["original_split_sha256"]
    assert freeze["test_index_sha256"] == sha256(ROOT / "derived/earthmoving_round2/test_clips.jsonl")
    output = run_root / "test_evaluation"
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in
            (ROOT / "derived/earthmoving_round2/test_clips.jsonl").read_text().splitlines() if line]
    assert len(rows) == 170 and {r["split"] for r in rows} == {"test"}
    cache_dir = output / "test_cache"
    cache_dir.mkdir()
    paths = {}
    for index, row in enumerate(rows):
        path = cache_dir / f"{index:03d}.npy"
        array = earthmoving_input(row)
        assert array.shape == (8, 128, 128, 3) and array.dtype == np.uint8
        np.save(path, array, allow_pickle=False)
        paths[row["sample_id"]] = path
    write_json(output / "test_cache_manifest.json", {
        "test_index_sha256": freeze["test_index_sha256"],
        "rows": [{"sample_id": r["sample_id"], "sha256": sha256(paths[r["sample_id"]])} for r in rows]})
    results = []
    for frozen in freeze["runs"]:
        key = frozen["run"]
        checkpoint = run_root / "runs" / key / "best.pt"
        assert sha256(checkpoint) == frozen["checkpoint_sha256"]
        seed = int(key.split("_s")[1])
        model = init(seed, "mps" if torch.backends.mps.is_available() else "cpu")
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        assert saved["epoch"] == frozen["selected_epoch"]
        model.load_state_dict(saved["state_dict"])
        device = next(model.parameters()).device
        started = time.perf_counter()
        metrics, per_class, cm, predictions = validate(model, rows, paths, device)
        elapsed = time.perf_counter() - started
        run_dir = output / key
        run_dir.mkdir()
        write_csv(run_dir / "metrics.csv", [{"model": key, **metrics, "fps": 170 * 8 / elapsed}])
        write_csv(run_dir / "class_metrics.csv", per_class)
        write_csv(run_dir / "predictions.csv", predictions)
        write_json(run_dir / "confusion_matrix.json", cm.tolist())
        plot_confusion(run_dir / "confusion_matrix.png", cm)
        write_json(run_dir / "evaluation.json", {
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "freeze_sha256": sha256(freeze_path),
            "evaluation_script_sha256": sha256(Path(__file__)),
            "test_index_sha256": freeze["test_index_sha256"],
            "metrics": metrics, "seconds": elapsed,
            "fps_scope": "cached clip tensors through model forward; excludes detector/tracker and source decoding",
            "original_test_previously_viewed": True,
            "source_mapping_and_rights_provisional": True})
        results.append({"run": key, **metrics, "fps": 170 * 8 / elapsed})
        print(json.dumps(results[-1]), flush=True)
    write_csv(output / "all_runs.csv", results)
    summary = {}
    for arm in ("A", "B", "C"):
        part = [r for r in results if r["run"].startswith(arm + "_")]
        summary[arm] = {metric: {"mean": float(np.mean([r[metric] for r in part])),
                                 "std_ddof1": float(np.std([r[metric] for r in part], ddof=1))}
                        for metric in ("accuracy", "precision", "recall", "macro_f1")}
    write_json(output / "summary.json", {
        "three_seed_test": summary,
        "freeze_sha256": sha256(freeze_path),
        "test_clips": 170,
        "test_source": "unchanged Earthmoving two original videos",
        "test_previously_viewed_in_prior_experiments": True,
        "no_test_based_model_or_arm_selection": True})
    print(json.dumps({"test_evaluation_complete": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
