#!/usr/bin/env python3
"""Screen sampled frames for cross-source near duplicates; not proof of independence."""
import io
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'derived/joint_preflight'


def phash(gray: np.ndarray) -> int:
    transform = dctn(gray.astype(np.float32), norm='ortho')[:8, :8]
    values = transform.flatten()[1:]
    median = np.median(values)
    bits = transform.flatten() > median
    return sum(int(bit) << i for i, bit in enumerate(bits))


def image_hash(raw: bytes) -> int:
    with Image.open(io.BytesIO(raw)) as im:
        gray = im.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        return phash(np.asarray(gray))


def kit_rows() -> list[dict]:
    manifest = json.loads((ROOT / 'Data_KIT/excavator_actions/manifest.json').read_text())
    rows = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for item in manifest['records']:
        path = ROOT / item['path']
        run = subprocess.run(
            [ffmpeg, '-v', 'error', '-threads', '1', '-i', str(path),
             '-vf', 'fps=1,scale=32:32,format=gray', '-f', 'rawvideo', '-'],
            capture_output=True, timeout=60,
        )
        if run.returncode or len(run.stdout) % 1024:
            raise RuntimeError(f'KIT fingerprint decode failed: {path}')
        frames = np.frombuffer(run.stdout, np.uint8).reshape(-1, 32, 32)
        for second, gray in enumerate(frames):
            rows.append({'source': 'kit', 'group': path.stem,
                         'key': f'{item["path"]}@{second}s', 'hash': phash(gray)})
    return rows


def earthmoving_rows() -> list[dict]:
    rows = []
    for folder in sorted((ROOT / 'Data/frames_ce').iterdir()):
        if not folder.is_dir():
            continue
        frames = sorted(folder.glob('*.jpg'))
        for i in range(0, len(frames), 25):
            path = frames[i]
            rows.append({'source': 'earthmoving', 'group': folder.name,
                         'key': str(path.relative_to(ROOT)), 'hash': image_hash(path.read_bytes())})
    return rows


def rathan_rows() -> list[dict]:
    manifest = json.loads((ROOT / 'Data_Rathan/idle_sources/manifest.json').read_text())
    return [
        {'source': 'rathan_idle', 'group': record['sample_id'].split(':', 1)[1].split('-')[0],
         'key': record['path'], 'hash': image_hash((ROOT / record['path']).read_bytes())}
        for record in manifest['records']
    ]


def compare(left: list[dict], right: list[dict], limit=8) -> dict:
    matches, nearest = [], []
    for a in left:
        best = min((((a['hash'] ^ b['hash']).bit_count(), b) for b in right
                    if a['key'] != b['key']), key=lambda pair: pair[0])
        nearest.append(best[0])
        if best[0] <= limit:
            matches.append({'left': a['key'], 'right': best[1]['key'], 'distance': best[0]})
    return {'left_count': len(left), 'right_count': len(right),
            'threshold': limit, 'near_match_count': len(matches),
            'minimum_distance': min(nearest) if nearest else None,
            'near_matches_first_30': sorted(matches, key=lambda x: x['distance'])[:30]}


def main() -> None:
    kit, earth, rathan = kit_rows(), earthmoving_rows(), rathan_rows()
    if len(rathan) != 100:
        raise RuntimeError(f'Unexpected Rathan source count: {len(rathan)}')
    splits = json.loads((ROOT / 'derived/earthmoving_v1/split.json').read_text())['groups']
    held_out = set(splits['val'] + splits['test'])
    result = {
        'method': '64-bit DCT perceptual hash; KIT 1 fps decoded frames, Earthmoving every 25th JPG, one Rathan export per source filename; Hamming <= 8 flags only',
        'sampled_counts': {'kit_frames': len(kit), 'earthmoving_frames': len(earth), 'rathan_idle_images': len(rathan)},
        'kit_vs_earthmoving_all': compare(kit, earth),
        'kit_vs_earthmoving_val_test': compare(kit, [x for x in earth if x['group'] in held_out]),
        'rathan_vs_earthmoving_val_test': compare(rathan, [x for x in earth if x['group'] in held_out]),
        'rathan_vs_kit': compare(rathan, kit),
        'caveat': 'No near matches in sampled frames does not prove different original videos. Similar backgrounds can also cause false matches.',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'overlap_screen.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sampled_counts': result['sampled_counts'],
                      'near_counts': {k: v['near_match_count'] for k, v in result.items() if k.endswith('_all') or '_vs_' in k}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
