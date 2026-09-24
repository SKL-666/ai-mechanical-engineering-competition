#!/usr/bin/env python3
"""Joint v2: balanced classes and reduced supplemental exposure, with frozen inputs."""
import argparse
from collections import Counter, defaultdict
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
from joint_preflight.data import ROOT, PRE, read_rows, verify_indexes
from scripts.train_joint import batch, init, input_paths, validate

PROTOCOL = ROOT / "configs/joint_training_v2.json"
RUN_ROOT = ROOT / "experiments/03_action_recognition/20260923_joint_v2"
SOURCE_CACHE_MANIFEST = ROOT / "experiments/03_action_recognition/20260923_joint_v1/earthmoving_cache_manifest.json"


class BalancedSource:
    """Class probability is proportional to sqrt(class count); rows uniform within class."""

    def __init__(self, rows, count_power):
        groups = defaultdict(list)
        for row in rows:
            groups[int(row["label"])].append(row)
        self.groups = dict(groups)
        self.classes = sorted(groups)
        self.weights = [len(groups[cls]) ** count_power for cls in self.classes]
        self.counts = {str(cls): len(groups[cls]) for cls in self.classes}

    def draw(self, rng, amount):
        labels = rng.choices(self.classes, weights=self.weights, k=amount)
        return [rng.choice(self.groups[label]) for label in labels]


def train(arm, seed, epochs_override=None, steps_override=None, out_override=None):
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["classes"] == ["Idle", "Swing Bucket", "Load Bucket", "Dump", "Move"]
    assert seed in protocol["seeds"] and arm in protocol["arms"]
    manifest, _ = verify_indexes()
    assert SOURCE_CACHE_MANIFEST.exists()
    paths = input_paths()
    candidates = read_rows("train_candidates.jsonl")
    train_rows = {source: [r for r in candidates if r["source"] == source]
                  for source in ("earthmoving", "kit", "rathan")}
    val_rows = read_rows("frozen_val_reference.jsonl")
    assert [len(train_rows[s]) for s in ("earthmoving", "kit", "rathan")] == [224, 304, 100]
    assert len(val_rows) == 85 and {r["source"] for r in val_rows} == {"earthmoving"}
    assert not {r["sample_id"] for r in candidates} & {r["sample_id"] for r in val_rows}
    counts = protocol["arms"][arm]
    assert counts["earthmoving_temporal_per_step"] + counts["kit_temporal_per_step"] == 16
    assert counts["rathan_still_per_step"] == (2 if arm == "C" else 0)
    samplers = {source: BalancedSource(rows, protocol["class_count_power"])
                for source, rows in train_rows.items()}
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
    for source in (Path(__file__), ROOT / "scripts/train_joint.py",
                   ROOT / "earthmoving/round2_model.py", ROOT / "earthmoving/metrics.py",
                   ROOT / "joint_preflight/data.py", PROTOCOL):
        shutil.copy2(source, snapshot / source.name)
    config = {
        "arm": arm, "seed": seed, "epochs": epochs, "steps_per_epoch": steps,
        "source_draws_per_step": counts, "class_count_power": protocol["class_count_power"],
        "source_class_counts": {source: sampler.counts for source, sampler in samplers.items()},
        "device": device, "protocol_sha256": sha256(PROTOCOL),
        "candidate_index_sha256": manifest["train_candidates_sha256"],
        "validation_index_sha256": manifest["frozen_val_reference_sha256"],
        "source_cache_manifest_sha256": sha256(SOURCE_CACHE_MANIFEST),
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
            for source, amount in (("earthmoving", counts["earthmoving_temporal_per_step"]),
                                   ("kit", counts["kit_temporal_per_step"])):
                drawn = samplers[source].draw(rng, amount)
                temporal.extend(drawn)
                exposures.update((r["source"], r["label"]) for r in drawn)
            rng.shuffle(temporal)
            x, y = batch(temporal, paths, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if counts["rathan_still_per_step"]:
                still = samplers["rathan"].draw(rng, counts["rathan_still_per_step"])
                exposures.update((r["source"], r["label"]) for r in still)
                sx, sy = batch(still, paths, device)
                spatial = model.encoder(sx[:, :, 0]).unsqueeze(-1)
                still_logits = model.head(spatial)[:, :, 0]
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
    write_csv(out / "metrics.csv", [{"model": "temporal_joint_v2", **metrics, "fps": ""}])
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
    tr = parser.add_subparsers(dest="command", required=True).add_parser("train")
    tr.add_argument("--arm", choices=["A", "B", "C"], required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        train(args.arm, args.seed, epochs_override=1, steps_override=1,
              out_override=RUN_ROOT / "smoke" / f"{args.arm}_s{args.seed}")
    else:
        train(args.arm, args.seed)
