#!/usr/bin/env python3
"""Retain only the 100 Rathan Idle source images and their original COCO boxes."""
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Productivity of Equipment.v7i.coco.zip'
DEST = ROOT / 'Data_Rathan/idle_sources'
INDEX = ROOT / 'derived/joint_preflight/train_candidates.jsonl'
EXPECTED_SHA = '229e7700cae14f3be9adcdbf13d784151b1e47b5a82aafea95a743aa6af7782a'


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if hash_file(SOURCE) != EXPECTED_SHA:
        raise RuntimeError('Rathan source archive changed')
    selected = [json.loads(line) for line in INDEX.read_text().splitlines()
                if json.loads(line)['source'] == 'rathan']
    if len(selected) != 100 or len({row['sample_id'] for row in selected}) != 100:
        raise RuntimeError('Expected exactly 100 unique Idle source images')
    records, coco_images, coco_annotations = [], [], []
    source_bbox_overflow_count = 0
    (DEST / 'images').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as archive:
        coco_by_split = {}
        for split in ('train', 'valid', 'test'):
            data = json.loads(archive.read(f'{split}/_annotations.coco.json'))
            categories = {item['id']: item['name'] for item in data['categories']}
            images = {item['file_name']: item for item in data['images']}
            images_by_id = {item['id']: item for item in data['images']}
            by_image = defaultdict(list)
            for item in data['annotations']:
                by_image[item['image_id']].append(item)
                original = images_by_id[item['image_id']]
                x, y, w, h = item['bbox']
                if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
                    raise RuntimeError('Nonfinite or empty source COCO bounding box')
                overflow = max(0, -x, -y, x + w - original['width'], y + h - original['height'])
                if overflow > 0.501:
                    raise RuntimeError('Source COCO bounding box overflow exceeds 0.501px')
                if overflow > 0:
                    source_bbox_overflow_count += 1
            coco_by_split[split] = (categories, images, by_image)
        for readme in ('README.dataset.txt', 'README.roboflow.txt'):
            (DEST / readme).write_bytes(archive.read(readme))
        for new_id, row in enumerate(sorted(selected, key=lambda item: item['sample_id']), 1):
            member = row.get('archive_member') or row['source_archive_member']
            split, filename = member.split('/', 1)
            categories, images, by_image = coco_by_split[split]
            original_image = images[filename]
            matching = [item for item in by_image[original_image['id']]
                        if categories[item['category_id']] == 'Excavator - Idle']
            if len(matching) != 1:
                raise RuntimeError(f'Expected exactly one original Idle box: {member}')
            raw = archive.read(member)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.size != (original_image['width'], original_image['height']):
                    raise RuntimeError(f'Image dimensions changed: {member}')
            name = row['sample_id'].split(':', 1)[1] + '.jpg'
            dest = DEST / 'images' / name
            dest.write_bytes(raw)
            image_sha = hash_file(dest)
            if image_sha != hashlib.sha256(raw).hexdigest():
                raise RuntimeError(f'Image write mismatch: {member}')
            box = matching[0]['bbox']
            coco_images.append({
                'id': new_id, 'file_name': f'images/{name}',
                'width': original_image['width'], 'height': original_image['height'],
            })
            coco_annotations.append({
                'id': new_id, 'image_id': new_id, 'category_id': 6,
                'bbox': box, 'area': matching[0].get('area', box[2] * box[3]),
                'iscrowd': matching[0].get('iscrowd', 0),
            })
            records.append({
                'sample_id': row['sample_id'],
                'path': str(dest.relative_to(ROOT)),
                'sha256': image_sha,
                'bytes': len(raw),
                'source_archive_member': member,
                'source_original_split': split,
                'source_image_id': original_image['id'],
                'source_annotation_id': matching[0]['id'],
                'source_bbox_xywh': box,
                'bbox_xyxy_clipped': row['bbox_xyxy'],
                'source_variants': row['variants_in_source_group'],
                'width': original_image['width'],
                'height': original_image['height'],
            })
    coco = {
        'info': {'description': 'Rathan v7 selected original bytes: one Idle image per source-like filename'},
        'licenses': [{'name': 'CC BY 4.0', 'url': 'https://creativecommons.org/licenses/by/4.0/'}],
        'categories': [{'id': 6, 'name': 'Excavator - Idle'}],
        'images': coco_images, 'annotations': coco_annotations,
    }
    (DEST / 'annotations.coco.json').write_text(json.dumps(coco, ensure_ascii=False, indent=2) + '\n')
    manifest = {
        'source_archive_name': SOURCE.name,
        'source_archive_sha256': EXPECTED_SHA,
        'source_archive_bytes': SOURCE.stat().st_size,
        'selection_rule': 'one lexicographically first v7 export per source-like filename; only single Excavator - Idle box images',
        'selected_images': len(records),
        'all_source_boxes_slightly_outside_image': source_bbox_overflow_count,
        'all_prefixes': sorted({r['sample_id'].split(':', 1)[1].split('-')[0] for r in records}),
        'records': records,
    }
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'images': len(records), 'retained_bytes': sum(r['bytes'] for r in records),
                      'source_archive_sha256': EXPECTED_SHA}, ensure_ascii=False))


if __name__ == '__main__':
    main()
