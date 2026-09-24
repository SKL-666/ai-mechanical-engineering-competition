#!/usr/bin/env python3
"""Screen cached KIT clips for near-identical frames across action and publisher split."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from earthmoving.common import sha256, write_json
from scripts.audit_joint_overlap import phash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived/supplemental_source_audit/kit_internal_similarity.json"


def main():
    rows = [json.loads(line) for line in
            (ROOT / "derived/joint_preflight/train_candidates.jsonl").read_text().splitlines() if line]
    clips = [r for r in rows if r["source"] == "kit"]
    cache = json.loads((ROOT / "derived/joint_preflight/cache_manifest.json").read_text())
    paths = {r["sample_id"]: ROOT / r["cache_path"] for r in cache["records"]}
    assert len(clips) == 304
    hashes = []
    for clip in clips:
        array = np.load(paths[clip["sample_id"]], allow_pickle=False)
        assert array.shape == (8, 128, 128, 3)
        for frame in array:
            gray = Image.fromarray(frame).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
            hashes.append(phash(np.asarray(gray)))
    bits = np.asarray(hashes, dtype=np.uint64)
    distances = np.bitwise_count(np.bitwise_xor(bits[:, None], bits[None, :]))
    clip_min = distances.reshape(304, 8, 304, 8).min(axis=(1, 3))
    np.fill_diagonal(clip_min, 65)
    pairs = []
    for i in range(304):
        for j in range(i + 1, 304):
            d = int(clip_min[i, j])
            if d <= 12:
                pairs.append({"a": clips[i]["sample_id"], "b": clips[j]["sample_id"],
                              "distance": d, "a_label": clips[i]["raw_label"],
                              "b_label": clips[j]["raw_label"],
                              "a_split": clips[i]["source_original_split"],
                              "b_split": clips[j]["source_original_split"]})
    close8 = [p for p in pairs if p["distance"] <= 8]
    close12 = pairs
    parent = list(range(304))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(304):
        for j in range(i + 1, 304):
            if clip_min[i, j] <= 8:
                parent[find(j)] = find(i)
    components = {}
    for i in range(304):
        components.setdefault(find(i), []).append(i)
    linked = [group for group in components.values() if len(group) > 1]
    result = {
        "method": "all 8 fixed 128px cached frames per KIT clip; 64-bit DCT pHash; minimum Hamming distance across clip-frame pairs",
        "cache_manifest_sha256": sha256(ROOT / "derived/joint_preflight/cache_manifest.json"),
        "clips": len(clips), "frames": len(hashes),
        "pairs_le8": len(close8), "pairs_le12": len(close12),
        "pairs_distance0": sum(p["distance"] == 0 for p in close8),
        "cross_publisher_split_pairs_le8": sum(p["a_split"] != p["b_split"] for p in close8),
        "cross_publisher_split_pairs_le12": sum(p["a_split"] != p["b_split"] for p in close12),
        "cross_publisher_split_pairs_distance0": sum(
            p["a_split"] != p["b_split"] and p["distance"] == 0 for p in close8),
        "cross_action_pairs_le8": sum(p["a_label"] != p["b_label"] for p in close8),
        "cross_action_pairs_le12": sum(p["a_label"] != p["b_label"] for p in close12),
        "linked_components_le8": len(linked),
        "largest_component_clips_le8": max((len(group) for group in linked), default=1),
        "cross_split_components_le8": sum(
            len({clips[i]["source_original_split"] for i in group}) > 1 for group in linked),
        "cross_action_components_le8": sum(
            len({clips[i]["raw_label"] for i in group}) > 1 for group in linked),
        "cross_split_pairs_le8_first_50": sorted(
            (p for p in close8 if p["a_split"] != p["b_split"]), key=lambda p: p["distance"])[:50],
        "cross_action_pairs_le8_first_50": sorted(
            (p for p in close8 if p["a_label"] != p["b_label"]), key=lambda p: p["distance"])[:50],
        "pairs_le12_first_100": sorted(pairs, key=lambda p: p["distance"])[:100],
        "caveat": "Near-frame similarity may indicate overlapping or visually similar source footage; absence does not establish independent original video IDs"}
    write_json(OUT, result)
    print(json.dumps({k: result[k] for k in
                      ("clips", "frames", "pairs_le8", "pairs_le12",
                       "cross_publisher_split_pairs_le8", "cross_action_pairs_le8")}), flush=True)


if __name__ == "__main__":
    main()
