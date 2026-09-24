#!/usr/bin/env python3
"""Build immutable-source, train-candidate indexes for three datasets. No training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'derived/joint_preflight'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))


def earthmoving_rows(mapping: dict) -> tuple[list[dict], list[dict]]:
    v1 = ROOT / 'derived/earthmoving_v1'
    split = json.loads((v1 / 'split.json').read_text())['groups']
    rows, validation = [], []
    for subset in ('train', 'val'):
        for row in read_jsonl(ROOT / 'derived/earthmoving_round2' / f'{subset}_clips.jsonl'):
            assert row['video_id'] in split[subset]
            assert row['label'] in range(5)
            item = {
                'sample_id': f'earthmoving:{row["sample_id"]}',
                'source': 'earthmoving',
                'source_group': f'earthmoving_video:{row["video_id"]}',
                'source_original_split': subset,
                'candidate_partition': 'train_candidate' if subset == 'train' else 'frozen_validation_reference',
                'media_type': 'frame_dir',
                'frame_dir': row['frame_dir'],
                'video_id': row['video_id'],
                'equipment_id': row['equipment_id'],
                'start_frame': row['start_frame'],
                'end_frame': row['end_frame'],
                'bbox_source': 'frozen_xml_gt',
                'label': row['label'],
                'raw_label': mapping['classes'][row['label']],
                'mapping_status': 'frozen_existing',
            }
            (rows if subset == 'train' else validation).append(item)
    return rows, validation


def kit_rows(mapping: dict) -> list[dict]:
    source = json.loads((ROOT / 'Data_KIT/excavator_actions/manifest.json').read_text())
    audit = {row['path']: row for row in read_jsonl(OUT / 'kit_media_audit.jsonl')}
    rows = []
    for item in source['records']:
        media = audit[item['path']]
        if not media['ok'] or not media['decoded_frames'] or not media['fps_reported']:
            raise RuntimeError(f'KIT media audit incomplete: {item["path"]}')
        action = item['action']
        rows.append({
            'sample_id': f'kit:{action}:{Path(item["path"]).stem}',
            'source': 'kit',
            'source_group': 'kit_origin_unresolved_ALL',
            'source_original_split': item['split'],
            'candidate_partition': 'train_candidate',
            'media_type': 'video',
            'media_path': item['path'],
            'decoded_frames': media['decoded_frames'],
            'fps_reported': media['fps_reported'],
            'bbox_source': 'none_full_frame',
            'label': mapping['kit'][action]['class_id'],
            'raw_label': action,
            'mapping_status': 'provisional_publisher_category',
            'media_sha256': item['sha256'],
        })
    assert len(rows) == 304
    return rows


def rathan_rows(mapping: dict) -> tuple[list[dict], dict]:
    curated_path = ROOT / 'Data_Rathan/idle_sources/manifest.json'
    curated = json.loads(curated_path.read_text())
    if curated['selected_images'] != 100:
        raise RuntimeError('Expected exactly 100 curated Rathan Idle images')
    rows = []
    for record in sorted(curated['records'], key=lambda item: item['sample_id']):
        path = ROOT / record['path']
        if not path.is_file() or digest(path) != record['sha256']:
            raise RuntimeError(f'Rathan curated source changed: {path}')
        stem = record['sample_id'].split(':', 1)[1]
        rows.append({
            'sample_id': record['sample_id'],
            'source': 'rathan',
            'source_group': f'rathan_source_prefix:{stem.split("-")[0]}',
            'source_original_split': record['source_original_split'],
            'candidate_partition': 'train_candidate',
            'media_type': 'file_image',
            'image_path': record['path'],
            'source_archive_member': record['source_archive_member'],
            'bbox_source': 'provided_coco_clipped_max_half_pixel',
            'bbox_xyxy': record['bbox_xyxy_clipped'],
            'image_width': record['width'],
            'image_height': record['height'],
            'label': mapping['rathan']['Excavator - Idle']['class_id'],
            'raw_label': 'Excavator - Idle',
            'mapping_status': 'provisional_still_image_only',
            'temporal_supervision': False,
            'variants_in_source_group': record['source_variants'],
        })
    if len(rows) != 100 or {row['source_group'] for row in rows} != {'rathan_source_prefix:BV6'}:
        raise RuntimeError('Unexpected Rathan idle source inventory')
    return rows, {'boxes_slightly_outside_image_in_original_full_export':
                  curated['all_source_boxes_slightly_outside_image']}


def main() -> None:
    mapping_path = ROOT / 'configs/joint_preflight_mapping.json'
    mapping = json.loads(mapping_path.read_text())
    frozen = ROOT / 'derived/earthmoving_v1/split.json'
    original_split_hash = digest(frozen)
    assert original_split_hash == '5741a11b8098b29c7604e9454006544f63d67c23388785426286fa10ef585327'
    train, validation = earthmoving_rows(mapping)
    kit = kit_rows(mapping)
    rathan, rathan_audit = rathan_rows(mapping)
    train.extend(kit)
    train.extend(rathan)
    ids = [row['sample_id'] for row in train + validation]
    assert len(ids) == len(set(ids))
    assert not ({r['source_group'] for r in train} & {r['source_group'] for r in validation})
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / 'train_candidates.jsonl'
    val_path = OUT / 'frozen_val_reference.jsonl'
    write_jsonl(train_path, train)
    write_jsonl(val_path, validation)
    counts = {src: {mapping['classes'][c]: sum(r['source'] == src and r['label'] == c for r in train)
                    for c in range(5)} for src in ('earthmoving', 'kit', 'rathan')}
    result = {
        'status': 'pretraining_candidate_index_not_training_release',
        'original_split_sha256': original_split_hash,
        'original_test_index_sha256': digest(ROOT / 'derived/earthmoving_round2/test_clips.jsonl'),
        'mapping_sha256': digest(mapping_path),
        'rathan_original_archive_sha256': json.loads(
            (ROOT / 'Data_Rathan/idle_sources/manifest.json').read_text())['source_archive_sha256'],
        'rathan_curated_manifest_sha256': digest(ROOT / 'Data_Rathan/idle_sources/manifest.json'),
        'kit_archive_sha256': json.loads((ROOT / 'Data_KIT/excavator_actions/manifest.json').read_text())['archive_sha256'],
        'train_candidates_sha256': digest(train_path),
        'frozen_val_reference_sha256': digest(val_path),
        'train_candidate_rows': len(train),
        'validation_reference_rows': len(validation),
        'source_class_counts': counts,
        'rathan_audit': rathan_audit,
        'supplemental_source_group_policy': 'all KIT clips train-only single unresolved source group; Rathan Idle one BV6 group, train-only',
        'rathan_temporal_policy': 'still-image auxiliary input only, never repeated as a true motion clip',
        'test_policy': 'frozen Earthmoving test index remains in original directory; no new test split or claim of blind evaluation',
    }
    (OUT / 'candidate_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'train_candidate_rows': len(train), 'validation_reference_rows': len(validation),
                      'source_class_counts': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
