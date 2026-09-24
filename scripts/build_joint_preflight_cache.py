#!/usr/bin/env python3
"""Cache fixed KIT temporal inputs and Rathan still inputs, without training."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'derived/joint_preflight'
SIZE = 128
TEMPORAL_FRAMES = 8
SPAN_SECONDS = 2.56


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def letterbox(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    scale = min(SIZE / image.width, SIZE / image.height)
    small = image.resize((max(1, round(image.width * scale)),
                          max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (SIZE, SIZE), (128, 128, 128))
    canvas.paste(small, ((SIZE - small.width) // 2, (SIZE - small.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def save_array(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npy.part')
    with temp.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temp.replace(path)
    return sha256(path)


def cache_kit(row: dict) -> dict:
    source = ROOT / row['media_path']
    target = OUT / 'cache/kit' / f'{row["sample_id"].replace(":", "_")}.npy'
    filt = (
        f'scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease:flags=bilinear,'
        f'pad={SIZE}:{SIZE}:(ow-iw)/2:(oh-ih)/2:color=gray,format=rgb24'
    )
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '1',
               '-i', str(source), '-vf', filt, '-f', 'rawvideo', '-']
    finished = subprocess.run(command, capture_output=True, timeout=90)
    if finished.returncode or len(finished.stdout) % (SIZE * SIZE * 3):
        raise RuntimeError(f'KIT decode failed: {source}: {finished.stderr[-300:]}')
    frames = np.frombuffer(finished.stdout, dtype=np.uint8).reshape(-1, SIZE, SIZE, 3)
    if len(frames) != row['decoded_frames']:
        raise RuntimeError(f'KIT frame count changed: {source}')
    span = min(len(frames), max(1, round(SPAN_SECONDS * row['fps_reported'])))
    start = (len(frames) - span) // 2
    indices = np.linspace(start, start + span - 1, TEMPORAL_FRAMES).round().astype(int)
    selected = frames[indices].copy()
    digest = save_array(target, selected)
    return {'sample_id': row['sample_id'], 'cache_path': str(target.relative_to(ROOT)),
            'sha256': digest, 'shape': list(selected.shape), 'dtype': 'uint8',
            'source_frame_indices': indices.tolist(),
            'effective_span_seconds': round(span / row['fps_reported'], 4)}


def cache_rathan(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        with Image.open(ROOT / row['image_path']) as image:
            x1, y1, x2, y2 = row['bbox_xyxy']
            crop = image.crop((int(np.floor(x1)), int(np.floor(y1)),
                               int(np.ceil(x2)), int(np.ceil(y2))))
            selected = letterbox(crop)[None, ...]
        target = OUT / 'cache/rathan' / f'{row["sample_id"].replace(":", "_")}.npy'
        digest = save_array(target, selected)
        records.append({'sample_id': row['sample_id'],
                        'cache_path': str(target.relative_to(ROOT)),
                        'sha256': digest, 'shape': list(selected.shape), 'dtype': 'uint8'})
    return records


def main() -> None:
    rows = [json.loads(line) for line in (OUT / 'train_candidates.jsonl').read_text().splitlines()]
    kit = [row for row in rows if row['source'] == 'kit']
    rathan = [row for row in rows if row['source'] == 'rathan']
    if len(kit) != 304 or len(rathan) != 100:
        raise RuntimeError('Unexpected candidate source counts')
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(cache_kit, row) for row in kit]
        for n, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if n % 50 == 0:
                print(f'KIT cached {n}/{len(kit)}', flush=True)
    records.extend(cache_rathan(rathan))
    records.sort(key=lambda row: row['sample_id'])
    if len({row['sample_id'] for row in records}) != len(records):
        raise RuntimeError('Duplicate cache sample ID')
    manifest = {
        'status': 'data_preflight_only',
        'kit_clips': len(kit),
        'rathan_stills': len(rathan),
        'temporal_frames': TEMPORAL_FRAMES,
        'image_size': SIZE,
        'kit_centered_span_s': SPAN_SECONDS,
        'kit_preprocessing': 'full frame RGB, aspect-preserving letterbox gray 128, bilinear',
        'rathan_preprocessing': 'provided excavator bbox clipped <=0.5px, RGB, aspect-preserving letterbox gray 128, bilinear',
        'train_candidate_index_sha256': sha256(OUT / 'train_candidates.jsonl'),
        'records': records,
    }
    path = OUT / 'cache_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'kit_clips': len(kit), 'rathan_stills': len(rathan),
                      'cache_bytes': sum((ROOT / row['cache_path']).stat().st_size for row in records)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
