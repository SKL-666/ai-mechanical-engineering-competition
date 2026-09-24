#!/usr/bin/env python3
"""Decode every isolated KIT excavator clip; never edit source media or train."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import sys

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def audit(path: Path) -> dict:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'info',
        '-nostats', '-threads', '1', '-i', str(path), '-map', '0:v:0',
        '-f', 'null', '-', '-progress', 'pipe:1',
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return {'path': str(path.relative_to(ROOT)), 'ok': False, 'error': 'decode_timeout_90s'}
    progress = re.findall(r'^frame=(\d+)$', done.stdout, re.MULTILINE)
    duration = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', done.stderr)
    stream = next((line for line in done.stderr.splitlines() if 'Video:' in line), '')
    shape = re.search(r'(\d{2,5})x(\d{2,5})', stream)
    fps = re.search(r'([\d.]+) fps', stream)
    result = {
        'path': str(path.relative_to(ROOT)),
        'ok': done.returncode == 0 and bool(progress) and int(progress[-1]) > 0,
        'decoded_frames': int(progress[-1]) if progress else None,
        'duration_s': (int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])) if duration else None,
        'width': int(shape[1]) if shape else None,
        'height': int(shape[2]) if shape else None,
        'fps_reported': float(fps[1]) if fps else None,
    }
    if not result['ok']:
        result['error'] = done.stderr[-1500:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'Data_KIT/excavator_actions/manifest.json').read_text())
    paths = [ROOT / item['path'] for item in manifest['records']]
    if len(paths) != 304 or len(set(paths)) != 304 or not all(p.is_file() for p in paths):
        raise SystemExit('KIT manifest is incomplete')
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit, p): p for p in paths}
        for n, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if n % 50 == 0:
                print(f'decoded {n}/{len(paths)}', flush=True)
    results.sort(key=lambda row: row['path'])
    output = ROOT / 'derived/joint_preflight/kit_media_audit.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in results))
    summary = {
        'clips': len(results),
        'decode_ok': sum(row['ok'] for row in results),
        'total_decoded_frames': sum(row['decoded_frames'] or 0 for row in results),
        'total_duration_s': sum(row['duration_s'] or 0 for row in results),
        'fps_counts': {str(f): sum(row['fps_reported'] == f for row in results)
                       for f in sorted({row['fps_reported'] for row in results})},
        'audit_path': str(output.relative_to(ROOT)),
        'ffmpeg': imageio_ffmpeg.get_ffmpeg_exe(),
        'imageio_ffmpeg_version': imageio_ffmpeg.__version__,
    }
    (output.parent / 'kit_media_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary['decode_ok'] != len(results):
        raise SystemExit('Some KIT clips did not fully decode')


if __name__ == '__main__':
    main()
