#!/usr/bin/env python3
"""Read-only provenance/semantic screen for retained KIT clips and Rathan Idle stills."""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from earthmoving.common import sha256, write_json
from scripts.audit_joint_overlap import image_hash, kit_rows, rathan_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "derived/supplemental_source_audit"


def quantile_samples(records):
    selected = {}
    for action in ("Digging", "Dumping", "Hauling", "Swinging"):
        chosen = []
        for split in ("train", "val"):
            rows = sorted((r for r in records if r["action"] == action and r["split"] == split),
                          key=lambda r: r["path"])
            chosen.extend(rows[i] for i in (0, len(rows) // 2, len(rows) - 1))
        selected[action] = chosen
    return selected


def semantic_sheets(kit_manifest):
    cache = json.loads((ROOT / "derived/joint_preflight/cache_manifest.json").read_text())
    paths = {r["sample_id"]: ROOT / r["cache_path"] for r in cache["records"]}
    selected = quantile_samples(kit_manifest["records"])
    samples = []
    for action, rows in selected.items():
        sheet = Image.new("RGB", (8 * 128, 6 * 154), "white")
        draw = ImageDraw.Draw(sheet)
        for y, item in enumerate(rows):
            sample_id = f'kit:{action}:{Path(item["path"]).stem}'
            frames = np.load(paths[sample_id], allow_pickle=False)
            assert frames.shape == (8, 128, 128, 3)
            for x, frame in enumerate(frames):
                sheet.paste(Image.fromarray(frame), (x * 128, y * 154 + 22))
            draw.text((3, y * 154 + 3), f'{item["split"]} {Path(item["path"]).name}', fill="black")
            samples.append({"action": action, "split": item["split"], "path": item["path"],
                            "source_sha256": item["sha256"]})
        sheet.save(OUT / f"KIT_{action}_contact_sheet.jpg", quality=92)
    return samples


def earthmoving_rows_stride5():
    rows = []
    for folder in sorted((ROOT / "Data/frames_ce").iterdir()):
        if not folder.is_dir():
            continue
        for i, path in enumerate(sorted(folder.glob("*.jpg"))):
            if i % 5 == 0:
                rows.append({"group": folder.name, "key": str(path.relative_to(ROOT)),
                             "hash": image_hash(path.read_bytes())})
    return rows


def nearest(left, right, threshold=8):
    a = np.asarray([r["hash"] for r in left], dtype=np.uint64)
    b = np.asarray([r["hash"] for r in right], dtype=np.uint64)
    closest = []
    for start in range(0, len(a), 128):
        d = np.bitwise_count(np.bitwise_xor(a[start:start + 128, None], b[None, :]))
        indices = d.argmin(axis=1)
        closest.extend((int(d[i, j]), int(j)) for i, j in enumerate(indices))
    flagged = [{"left": left[i]["key"], "right": right[j]["key"], "distance": distance}
               for i, (distance, j) in enumerate(closest) if distance <= threshold]
    return {"left_frames": len(left), "right_frames": len(right),
            "phash_hamming_threshold": threshold, "flagged_frames": len(flagged),
            "minimum_distance": min((x[0] for x in closest), default=None),
            "flagged_first_30": flagged[:30]}


def rathan_filename_groups(manifest):
    numbers = sorted(int(re.search(r"BV6-FRAMES-(\d+)", r["sample_id"]).group(1))
                     for r in manifest["records"])
    contiguous = []
    start = last = numbers[0]
    for number in numbers[1:]:
        if number == last + 1:
            last = number
        else:
            contiguous.append([start, last])
            start = last = number
    contiguous.append([start, last])
    return {"selected_images": len(numbers), "prefixes": manifest["all_prefixes"],
            "number_min": min(numbers), "number_max": max(numbers),
            "median_adjacent_number_gap": float(np.median(np.diff(numbers))),
            "contiguous_runs": len(contiguous),
            "longest_contiguous_run": max(b - a + 1 for a, b in contiguous),
            "number_bands": {"84-300": sum(84 <= n <= 300 for n in numbers),
                             "2190-2253": sum(2190 <= n <= 2253 for n in numbers),
                             "2556-2586": sum(2556 <= n <= 2586 for n in numbers)},
            "caveat": "BV6 prefix and frame-like numbers suggest shared source footage, not a verified original video ID"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kit_manifest = json.loads((ROOT / "Data_KIT/excavator_actions/manifest.json").read_text())
    rathan_manifest = json.loads((ROOT / "Data_Rathan/idle_sources/manifest.json").read_text())
    selected = semantic_sheets(kit_manifest)
    kit = kit_rows()
    earth = earthmoving_rows_stride5()
    rathan = rathan_rows()
    split = json.loads((ROOT / "derived/earthmoving_v1/split.json").read_text())["groups"]
    held_out = set(split["val"] + split["test"])
    result = {
        "method": "KIT 1 fps decoded pHash; Earthmoving every 5th original JPG; Rathan 100 retained Idle images; 64-bit DCT Hamming <=8 only screens near-identical frames",
        "input_hashes": {
            "kit_manifest_sha256": sha256(ROOT / "Data_KIT/excavator_actions/manifest.json"),
            "rathan_manifest_sha256": sha256(ROOT / "Data_Rathan/idle_sources/manifest.json"),
            "earthmoving_split_sha256": sha256(ROOT / "derived/earthmoving_v1/split.json")},
        "kit_semantic_contact_sheet_clips": selected,
        "rathan_filename_groups": rathan_filename_groups(rathan_manifest),
        "kit_vs_earthmoving_all": nearest(kit, earth),
        "kit_vs_earthmoving_val_test": nearest(kit, [r for r in earth if r["group"] in held_out]),
        "rathan_vs_earthmoving_all": nearest(rathan, earth),
        "rathan_vs_kit": nearest(rathan, kit),
        "limitation": "No pHash match does not prove distinct original videos, sites, creators, or rights; sampled visual review is not relabeling or exhaustive semantic validation",
    }
    write_json(OUT / "audit.json", result)
    print(json.dumps({"kit_frames": len(kit), "earthmoving_frames": len(earth),
                      "rathan_images": len(rathan),
                      "flags": {k: v["flagged_frames"] for k, v in result.items() if "_vs_" in k}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
