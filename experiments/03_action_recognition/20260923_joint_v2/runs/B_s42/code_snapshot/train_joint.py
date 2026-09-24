#!/usr/bin/env python3
"""Frozen A/B/C local training on the joint preflight inputs; never reads test labels."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from earthmoving.common import CLASSES, sha256, write_csv, write_json
from earthmoving.metrics import classification_metrics
from earthmoving.round2_model import ActionNet
from joint_preflight.data import ROOT, PRE, earthmoving_input, read_rows, verify_indexes

PROTOCOL = ROOT / "configs/joint_training_v1.json"
RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v1"


def cache_path(row):
    import hashlib
    key = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
    return RUN_ROOT / "earthmoving_cache" / f"{key}.npy"


def prepare_cache():
    verify_indexes()
    rows = [r for name in ("train_candidates.jsonl", "frozen_val_reference.jsonl")
            for r in read_rows(name) if r["source"] == "earthmoving"]
    assert len(rows) == 309 and len({r["sample_id"] for r in rows}) == 309
    paths = [cache_path(r) for r in rows]
    assert len(set(paths)) == len(paths)
    folder = RUN_ROOT / "earthmoving_cache"
    folder.mkdir(parents=True, exist_ok=True)
    for i, (row, path) in enumerate(zip(rows, paths), 1):
        if not path.exists():
            array = earthmoving_input(row)
            assert array.shape == (8, 128, 128, 3) and array.dtype == np.uint8
            temp = path.with_suffix(".tmp.npy")
            np.save(temp, array, allow_pickle=False)
            temp.replace(path)
        if i % 50 == 0:
            print(json.dumps({"cached": i, "total": len(rows)}), flush=True)
    records = [{"sample_id": r["sample_id"], "path": str(p.relative_to(ROOT)),
                "sha256": sha256(p)} for r, p in zip(rows, paths)]
    write_json(RUN_ROOT / "earthmoving_cache_manifest.json", {
        "rows": len(records), "candidate_index_sha256": sha256(PRE / "train_candidates.jsonl"),
        "validation_index_sha256": sha256(PRE / "frozen_val_reference.jsonl"),
        "records": records})
    print(json.dumps({"cache_complete": len(records)}), flush=True)


def input_paths():
    manifest, cache = verify_indexes()
    em = json.loads((RUN_ROOT / "earthmoving_cache_manifest.json").read_text())
    assert em["rows"] == 309
    assert em["candidate_index_sha256"] == manifest["train_candidates_sha256"]
    assert em["validation_index_sha256"] == manifest["frozen_val_reference_sha256"]
    paths = {r["sample_id"]: ROOT / r["path"] for r in em["records"]}
    paths.update({r["sample_id"]: ROOT / r["cache_path"] for r in cache["records"]})
    return paths


@lru_cache(maxsize=800)
def load_array(path):
    return np.load(path, allow_pickle=False)


def batch(rows, paths, device):
    arrays = [load_array(str(paths[r["sample_id"]])) for r in rows]
    x = np.stack(arrays)
    assert x.dtype == np.uint8 and x.shape[2:] == (128, 128, 3)
    x = torch.from_numpy(x.transpose(0, 4, 1, 2, 3).copy()).to(device=device, dtype=torch.float32).div_(255)
    y = torch.tensor([r["label"] for r in rows], dtype=torch.long, device=device)
    return x, y


def validate(model, rows, paths, device):
    model.eval()
    truth, pred, confidences = [], [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            part = rows[start:start + 16]
            x, y = batch(part, paths, device)
            probs = model(x).softmax(-1).cpu().numpy()
            truth.extend(y.cpu().tolist())
            pred.extend(probs.argmax(axis=1).tolist())
            confidences.extend(probs.max(axis=1).tolist())
    metrics, class_metrics, cm = classification_metrics(truth, pred)
    predictions = [{"sample_id": r["sample_id"], "video_id": r["video_id"],
                    "start_frame": r["start_frame"], "end_frame": r["end_frame"],
                    "true_action": CLASSES[t], "pred_action": CLASSES[p], "confidence": c}
                   for r, t, p, c in zip(rows, truth, pred, confidences)]
    return metrics, class_metrics, cm, predictions


def init(seed, device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ActionNet(temporal=True).to(device)


def train(arm, seed, epochs_override=None, steps_override=None, out_override=None):
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["classes"] == ["Idle", "Swing Bucket", "Load Bucket", "Dump", "Move"]
    assert seed in protocol["seeds"] and arm in protocol["arms"]
    paths = input_paths()
    candidates = read_rows("train_candidates.jsonl")
    train_rows = {source: [r for r in candidates if r["source"] == source]
                  for source in ("earthmoving", "kit", "rathan")}
    val_rows = read_rows("frozen_val_reference.jsonl")
    assert len(train_rows["earthmoving"]) == 224 and len(train_rows["kit"]) == 304
    assert len(train_rows["rathan"]) == 100 and len(val_rows) == 85
    assert {r["source"] for r in val_rows} == {"earthmoving"}
    assert not set(r["sample_id"] for r in candidates) & set(r["sample_id"] for r in val_rows)
    counts = protocol["arms"][arm]
    assert counts["earthmoving_temporal_per_step"] + counts["kit_temporal_per_step"] == 16
    if arm != "C":
        assert counts["rathan_still_per_step"] == 0
    epochs = epochs_override or protocol["epochs"]
    steps = steps_override or protocol["steps_per_epoch"]
    out = out_override or RUN_ROOT / "runs" / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = init(seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=protocol["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.00001)
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__), ROOT / "earthmoving/round2_model.py",
                   ROOT / "earthmoving/metrics.py", ROOT / "joint_preflight/data.py", PROTOCOL):
        shutil.copy2(source, snapshot / source.name)
    config = {
        "arm": arm, "seed": seed, "epochs": epochs, "steps_per_epoch": steps,
        "source_draws_per_step": counts, "device": device, "protocol_sha256": sha256(PROTOCOL),
        "candidate_index_sha256": sha256(PRE / "train_candidates.jsonl"),
        "validation_index_sha256": sha256(PRE / "frozen_val_reference.jsonl"),
        "cache_manifest_sha256": sha256(RUN_ROOT / "earthmoving_cache_manifest.json"),
        "mapping_sha256": sha256(ROOT / "configs/joint_preflight_mapping.json"),
        "command": " ".join(sys.argv), "test_evaluated": False,
        "supplemental_mapping_and_rights": "provisional; local experiment only",
    }
    write_json(out / "config.json", config)
    logs = []
    best = -1.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        rng = random.Random(seed * 100003 + epoch)
        model.train()
        total_loss = 0.0
        exposures = Counter()
        for _ in range(steps):
            temporal = []
            for source, n in (("earthmoving", counts["earthmoving_temporal_per_step"]),
                              ("kit", counts["kit_temporal_per_step"])):
                drawn = rng.choices(train_rows[source], k=n)
                temporal.extend(drawn)
                exposures.update((r["source"], r["label"]) for r in drawn)
            rng.shuffle(temporal)
            x, y = batch(temporal, paths, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if counts["rathan_still_per_step"]:
                still = rng.choices(train_rows["rathan"], k=counts["rathan_still_per_step"])
                exposures.update((r["source"], r["label"]) for r in still)
                sx, sy = batch(still, paths, device)
                # A one-frame still supervises the spatial encoder and class head only.
                features = model.encoder(sx[:, :, 0]).unsqueeze(-1)
                still_logits = model.head(features)[:, :, 0]
                loss = loss + protocol["rathan_aux_loss_weight"] * torch.nn.functional.cross_entropy(still_logits, sy)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        metrics, _, _, _ = validate(model, val_rows, paths, device)
        item = {"epoch": epoch, "loss_per_step": total_loss / steps,
                "lr": optimizer.param_groups[0]["lr"], "val_macro_f1": metrics["macro_f1"],
                "val_accuracy": metrics["accuracy"], "elapsed_seconds": time.perf_counter() - started,
                "source_class_draws": {f"{src}:{cls}": n for (src, cls), n in sorted(exposures.items())}}
        logs.append(item)
        write_json(out / "training_log.json", logs)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch, "val_macro_f1": best, "config": config}, out / "best.pt")
        scheduler.step()
        print(json.dumps({"arm": arm, "seed": seed, **item}, ensure_ascii=False), flush=True)
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    metrics, class_metrics, cm, predictions = validate(model, val_rows, paths, device)
    assert abs(metrics["macro_f1"] - best) < 1e-10
    write_csv(out / "metrics.csv", [{"model": "temporal_joint_v1", **metrics, "fps": ""}])
    write_csv(out / "class_metrics.csv", class_metrics)
    write_csv(out / "predictions.csv", predictions)
    write_json(out / "confusion_matrix.json", cm.tolist())
    write_json(out / "run_summary.json", {
        "arm": arm, "seed": seed, "selected_epoch": saved["epoch"],
        "val_metrics": metrics, "checkpoint_sha256": sha256(out / "best.pt"),
        "epochs_completed": epochs, "test_evaluated": False,
        "training_seconds": time.perf_counter() - started})
    print(json.dumps({"completed": str(out), "selected_epoch": saved["epoch"],
                      "val_macro_f1": best}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-cache")
    tr = sub.add_parser("train")
    tr.add_argument("--arm", choices=["A", "B", "C"], required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-cache":
        prepare_cache()
    else:
        if args.smoke:
            train(args.arm, args.seed, epochs_override=1, steps_override=1,
                  out_override=RUN_ROOT / "smoke" / f"{args.arm}_s{args.seed}")
        else:
            train(args.arm, args.seed)
