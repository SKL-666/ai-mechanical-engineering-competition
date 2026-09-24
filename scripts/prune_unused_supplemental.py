#!/usr/bin/env python3
"""Delete only two redundant full ZIPs after verifying retained training sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIT_ZIP = ROOT / 'Data_KIT/source/Datasets.zip'
RATHAN_ZIP = ROOT / 'Productivity of Equipment.v7i.coco.zip'
EXPECTED = {
    KIT_ZIP: ('af7b54188b3eb80bd22fea55210254bf0ba0dd7dda889540a3958b72ec0f02e3', 5815544519),
    RATHAN_ZIP: ('229e7700cae14f3be9adcdbf13d784151b1e47b5a82aafea95a743aa6af7782a', 241713848),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def verify_retained() -> dict:
    kit = json.loads((ROOT / 'Data_KIT/excavator_actions/manifest.json').read_text())
    rathan = json.loads((ROOT / 'Data_Rathan/idle_sources/manifest.json').read_text())
    cache = json.loads((ROOT / 'derived/joint_preflight/cache_manifest.json').read_text())
    candidate = json.loads((ROOT / 'derived/joint_preflight/candidate_manifest.json').read_text())
    if len(kit['records']) != 304 or len(rathan['records']) != 100 or len(cache['records']) != 404:
        raise RuntimeError('Unexpected retained file counts')
    if kit['archive_sha256'] != EXPECTED[KIT_ZIP][0]:
        raise RuntimeError('KIT provenance hash mismatch')
    if rathan['source_archive_sha256'] != EXPECTED[RATHAN_ZIP][0]:
        raise RuntimeError('Rathan provenance hash mismatch')
    if candidate['rathan_original_archive_sha256'] != EXPECTED[RATHAN_ZIP][0]:
        raise RuntimeError('Candidate manifest provenance mismatch')
    for item in kit['records'] + rathan['records']:
        path = ROOT / item['path']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise RuntimeError(f'Retained source missing or changed: {path}')
    for item in cache['records']:
        path = ROOT / item['cache_path']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise RuntimeError(f'Candidate cache missing or changed: {path}')
    verify = subprocess.run(
        [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/verify_joint_preflight.py')],
        cwd=ROOT, capture_output=True, text=True,
    )
    if verify.returncode:
        raise RuntimeError(f'Data adapter verification failed: {verify.stderr[-1000:]}')
    return {
        'retained_kit_mp4': len(kit['records']),
        'retained_rathan_original_images': len(rathan['records']),
        'retained_cache_files': len(cache['records']),
        'candidate_rows': candidate['train_candidate_rows'],
        'verification_stdout': verify.stdout.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='actually remove the exact two full archives')
    args = parser.parse_args()
    retained = verify_retained()
    old = {}
    for path, (expected_hash, expected_bytes) in EXPECTED.items():
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise RuntimeError(f'Full archive missing or unexpected size: {path}')
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f'Full archive hash changed: {path}')
        old[str(path.relative_to(ROOT))] = {'bytes': expected_bytes, 'sha256': actual}
    result = {
        'scope': 'remove redundant complete KIT and Rathan ZIPs only',
        'removed_archives': old,
        'retained': retained,
        'earthmoving_data_and_frozen_split_untouched': True,
        'formal_training_started': False,
        'applied': args.apply,
    }
    if args.apply:
        for path in EXPECTED:
            path.unlink()
        KIT_ZIP.parent.rmdir()
        if any(path.exists() for path in EXPECTED):
            raise RuntimeError('An archive still exists after cleanup')
        out = ROOT / 'derived/joint_preflight/cleanup_record.json'
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
