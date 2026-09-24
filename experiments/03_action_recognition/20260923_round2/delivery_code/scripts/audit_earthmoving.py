#!/usr/bin/env python3
"""Full image decode/hash, XML validation, ambiguous interval analysis, ZIP CRC."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from earthmoving.common import sha256, write_json, write_jsonl, label_hypotheses, spans


def inspect_image(path):
    data = path.read_bytes()
    import hashlib
    with Image.open(io.BytesIO(data)) as im:
        im.load()  # Full decode, not just JPEG header verification.
        size = im.size
    return path, size, hashlib.sha256(data).hexdigest()


def run(root, out, verify_zip):
    out.mkdir(parents=True, exist_ok=True)
    data = root / 'Data'
    report = {'source': 'Earthmoving Equipment fyw6ps2d2j v1', 'videos': {},
              'fps': {'value': 25, 'status': 'publisher_search_excerpt; no local timestamp metadata',
                      'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0926580518308525',
                      'section': 'Ground truth data', 'access_date': '2026-09-23',
                      'direct_page': '403', 'local_frame_cadence_independently_verified': False},
              'mapping_status': 'unresolved; retain only unanimous valid labels under four hypotheses',
              'hypotheses': ['zero_closed', 'one_closed', 'zero_half_open', 'one_half_open']}
    manifest, duplicates = [], defaultdict(list)
    raw_intervals, boxes_all, ignored_all = [], [], []
    for folder in sorted((data / 'frames_ce').iterdir()):
        if not folder.is_dir():
            continue
        video = folder.name
        files = sorted(folder.glob('*.jpg'))
        ids = [int(re.fullmatch(rf'{video}_I(\d+)\.jpg', p.name)[1]) for p in files]
        assert ids == list(range(len(files))), f'Frame numbering gap/duplicate: {video}'
        sizes = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for p, size, digest in pool.map(inspect_image, files):
                t = int(p.stem.split('_I')[1])
                sizes[t] = size
                rel = p.relative_to(root).as_posix()
                manifest.append({'path': rel, 'sha256': digest, 'bytes': p.stat().st_size})
                duplicates[digest].append((video, t))
        label_file = data / 'Labels' / video / 'excavator.txt'
        rows = [list(map(int, l.split())) for l in label_file.read_text().splitlines() if l.strip()]
        assert all(len(r) == 3 and 0 <= r[2] <= 4 and r[0] <= r[1] for r in rows)
        assert all(a[0] <= b[0] for a, b in zip(rows, rows[1:]))
        consensus, hypotheses = label_hypotheses(rows, len(files))
        np.save(out / f'{video}_consensus.npy', consensus)
        for s, e in spans(consensus < 0):
            ignored_all.append({'video_id': video, 'start_frame': s, 'end_frame': e,
                                'reason': 'missing_conflicting_or_hypothesis_disagreement'})
        for line, (s, e, c) in enumerate(rows, 1):
            raw_intervals.append({'video_id': video, 'raw_line': line,
                                  'raw_start': s, 'raw_end': e, 'label': c,
                                  'raw_file': str(label_file.relative_to(root))})
        tree = ET.parse(data / 'DetectionTrackingAnnotations' / f'{video}.xml')
        objects, excavators = [], []
        for obj in tree.findall('object'):
            name, eid = obj.findtext('name'), obj.findtext('id')
            if name == 'excavator':
                excavators.append(eid)
            seen, invalid, clipped, missing_frames = set(), [], [], []
            object_boxes = []
            for poly in obj.findall('polygon'):
                t = int(poly.findtext('t'))
                assert t not in seen, f'Duplicate XML frame {video}/{eid}/{t}'
                seen.add(t)
                xy = [(float(pt.findtext('x')), float(pt.findtext('y'))) for pt in poly.findall('pt')]
                if t not in sizes or len(xy) < 3:
                    invalid.append(t)
                    continue
                x1, y1 = min(x for x, y in xy), min(y for x, y in xy)
                x2, y2 = max(x for x, y in xy), max(y for x, y in xy)
                w, h = sizes[t]
                if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                    invalid.append(t)
                    continue
                bounded = [max(0, min(w, x1)), max(0, min(h, y1)),
                           max(0, min(w, x2)), max(0, min(h, y2))]
                if bounded != [x1, y1, x2, y2]:
                    clipped.append(t)
                if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                    invalid.append(t)
                    continue
                object_boxes.append({'video_id': video, 'equipment_id': str(eid), 'frame': t,
                                     'x1': bounded[0], 'y1': bounded[1],
                                     'x2': bounded[2], 'y2': bounded[3], 'bbox_source': 'xml_gt'})
            missing_frames = sorted(set(range(len(files))) - seen)
            objects.append({'name': name, 'equipment_id': eid, 'polygon_count': len(seen),
                            'frame_min': min(seen), 'frame_max': max(seen),
                            'invalid_frames': invalid, 'clipped_frames': clipped,
                            'missing_frame_count': len(missing_frames)})
            if name == 'excavator':
                boxes_all.extend(object_boxes)
        assert len(excavators) == 1, f'Excavator action association ambiguous: {video}'
        report['videos'][video] = {'frames': len(files), 'frame_range': [ids[0], ids[-1]],
                                  'image_dimensions': [list(x) for x in sorted(set(sizes.values()))],
                                  'raw_intervals': len(rows), 'class_counts': dict(Counter(r[2] for r in rows)),
                                  'boundary_delta_counts': dict(Counter(b[0] - a[1] for a, b in zip(rows, rows[1:]))),
                                  'hypotheses': hypotheses, 'consensus_frames': int((consensus >= 0).sum()),
                                  'ignored_frames': int((consensus < 0).sum()), 'xml_objects': objects}
        print(f'{video}: decoded {len(files)} frames; {len(rows)} intervals; ignored {(consensus < 0).sum()}', flush=True)
    # Also fingerprint all non-JPEG source files and guidance originals.
    for folder in [data, root / '指导']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix.lower() != '.jpg':
                manifest.append({'path': p.relative_to(root).as_posix(), 'sha256': sha256(p), 'bytes': p.stat().st_size})
    cross_video = [{'sha256': h, 'frames': rows} for h, rows in duplicates.items()
                   if len({v for v, _ in rows}) > 1]
    report['cross_video_exact_duplicates'] = cross_video
    report['duplicate_hash_groups_within_or_across_videos'] = sum(len(v) > 1 for v in duplicates.values())
    report['totals'] = {'frames': sum(v['frames'] for v in report['videos'].values()),
                        'raw_intervals': len(raw_intervals), 'excavator_boxes': len(boxes_all),
                        'ignored_frames': sum(v['ignored_frames'] for v in report['videos'].values())}
    archive = root / 'earthmoving-equipment-fyw6ps2d2j-v1.zip'
    manifest.append({'path': archive.name, 'sha256': sha256(archive), 'bytes': archive.stat().st_size})
    if verify_zip:
        # Original archive contains one nested Data.zip; verify both levels without writing Data.
        with zipfile.ZipFile(archive) as outer:
            assert outer.testzip() is None, 'Outer ZIP CRC failed'
            names = [n for n in outer.namelist() if n.endswith('.zip')]
            assert len(names) == 1
            with tempfile.TemporaryFile() as tmp:
                import shutil
                with outer.open(names[0]) as src:
                    shutil.copyfileobj(src, tmp)
                tmp.seek(0)
                with zipfile.ZipFile(tmp) as inner:
                    assert inner.testzip() is None, 'Inner ZIP CRC failed'
                    local = {r['path']: r for r in manifest}
                    matched, unmatched, mismatch = 0, [], []
                    # CRC ties every local raw file to the nested archive bytes, in addition to SHA manifest.
                    import zlib
                    for info in inner.infolist():
                        if info.is_dir() or '__MACOSX' in info.filename:
                            continue
                        candidate = info.filename if info.filename.startswith('Data/') else 'Data/' + info.filename
                        p = root / candidate
                        if candidate not in local:
                            unmatched.append(info.filename)
                            continue
                        if (zlib.crc32(p.read_bytes()) & 0xffffffff) != info.CRC or p.stat().st_size != info.file_size:
                            mismatch.append(candidate)
                        matched += 1
                    report['archive'] = {'outer_crc': 'pass', 'inner_crc': 'pass', 'nested_members': len(inner.infolist()),
                                         'local_matches_checked': matched, 'unmatched_entries': unmatched,
                                         'local_crc_mismatches': mismatch}
                    assert not mismatch
                    leaf_archives = []
                    for info in inner.infolist():
                        if not info.filename.endswith('.zip'):
                            continue
                        with tempfile.TemporaryFile() as leaf_tmp:
                            with inner.open(info.filename) as src:
                                shutil.copyfileobj(src, leaf_tmp)
                            leaf_tmp.seek(0)
                            with zipfile.ZipFile(leaf_tmp) as leaf:
                                assert leaf.testzip() is None, f'Leaf CRC failed: {info.filename}'
                                checked, missing, bad = 0, [], []
                                for item in leaf.infolist():
                                    if item.is_dir() or '__MACOSX' in item.filename:
                                        continue
                                    candidate = 'Data/' + item.filename
                                    p = root / candidate
                                    if candidate not in local:
                                        missing.append(candidate)
                                        continue
                                    if (zlib.crc32(p.read_bytes()) & 0xffffffff) != item.CRC or p.stat().st_size != item.file_size:
                                        bad.append(candidate)
                                    checked += 1
                                leaf_archives.append({'archive': info.filename, 'crc': 'pass',
                                                      'local_files_checked': checked, 'missing': missing, 'mismatch': bad})
                                assert not bad and not missing
                    report['archive']['leaf_archives'] = leaf_archives
    write_jsonl(out / 'source_manifest.jsonl', sorted(manifest, key=lambda x: x['path']))
    write_jsonl(out / 'raw_intervals.jsonl', raw_intervals)
    write_jsonl(out / 'boxes.jsonl', boxes_all)
    write_jsonl(out / 'ignored_ranges.jsonl', ignored_all)
    write_json(out / 'audit.json', report)
    print(json.dumps(report['totals']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, default=Path('derived/earthmoving_v1'))
    p.add_argument('--verify-zip', action='store_true')
    a = p.parse_args()
    run(a.root.resolve(), a.out, a.verify_zip)
