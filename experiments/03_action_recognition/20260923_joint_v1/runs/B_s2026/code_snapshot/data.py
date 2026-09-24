"""Shared input shapes for the proposed joint experiment; no trainer is defined here."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / 'derived/joint_preflight'
SIZE = 128
TEMPORAL_FRAMES = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (PRE / name).read_text().splitlines() if line]


def verify_indexes() -> tuple[dict, dict]:
    manifest = json.loads((PRE / 'candidate_manifest.json').read_text())
    caches = json.loads((PRE / 'cache_manifest.json').read_text())
    if sha256(PRE / 'train_candidates.jsonl') != manifest['train_candidates_sha256']:
        raise RuntimeError('Joint candidate index changed')
    if sha256(PRE / 'frozen_val_reference.jsonl') != manifest['frozen_val_reference_sha256']:
        raise RuntimeError('Frozen validation reference changed')
    if sha256(ROOT / 'derived/earthmoving_v1/split.json') != manifest['original_split_sha256']:
        raise RuntimeError('Frozen Earthmoving split changed')
    if sha256(ROOT / 'derived/earthmoving_round2/test_clips.jsonl') != manifest['original_test_index_sha256']:
        raise RuntimeError('Frozen Earthmoving test index changed')
    if sha256(ROOT / 'configs/joint_preflight_mapping.json') != manifest['mapping_sha256']:
        raise RuntimeError('Provisional class mapping changed')
    if sha256(ROOT / 'Data_Rathan/idle_sources/manifest.json') != manifest['rathan_curated_manifest_sha256']:
        raise RuntimeError('Curated Rathan source manifest changed')
    if caches['train_candidate_index_sha256'] != manifest['train_candidates_sha256']:
        raise RuntimeError('Cache is for a different candidate index')
    return manifest, caches


def letterbox(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    scale = min(SIZE / image.width, SIZE / image.height)
    small = image.resize((max(1, round(image.width * scale)),
                          max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (SIZE, SIZE), (128, 128, 128))
    canvas.paste(small, ((SIZE - small.width) // 2, (SIZE - small.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def earthmoving_input(row: dict) -> np.ndarray:
    ids = np.linspace(row['start_frame'], row['end_frame'] - 1, TEMPORAL_FRAMES).round().astype(int)
    frames = []
    for index in ids:
        path = ROOT / row['frame_dir'] / f'{row["video_id"]}_I{index:05d}.jpg'
        with Image.open(path) as image:
            frames.append(letterbox(image))
    return np.stack(frames)


def tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim != 4 or array.shape[1:] != (SIZE, SIZE, 3):
        raise ValueError(f'Unexpected cached shape: {array.shape}')
    return torch.from_numpy(array.transpose(3, 0, 1, 2).copy()).float().div_(255)


class TemporalCandidates(Dataset):
    """Earthmoving and KIT temporal clips; frozen Earthmoving validation is separate."""

    def __init__(self, partition: str):
        if partition not in {'train_candidate', 'frozen_val_reference'}:
            raise ValueError('Only train candidates and frozen validation references are exposed')
        _, caches = verify_indexes()
        filename = 'train_candidates.jsonl' if partition == 'train_candidate' else 'frozen_val_reference.jsonl'
        self.rows = [row for row in read_rows(filename) if row['source'] in {'earthmoving', 'kit'}]
        self.cache = {row['sample_id']: ROOT / row['cache_path'] for row in caches['records']}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        if row['source'] == 'earthmoving':
            array = earthmoving_input(row)
        else:
            array = np.load(self.cache[row['sample_id']], allow_pickle=False)
        if array.shape[0] != TEMPORAL_FRAMES:
            raise ValueError('Temporal input must contain eight frames')
        return tensor(array), row['label'], row['sample_id']


class RathanStillCandidates(Dataset):
    """Single-frame, one-representative-per-source Idle auxiliary inputs only."""

    def __init__(self):
        _, caches = verify_indexes()
        self.rows = [row for row in read_rows('train_candidates.jsonl') if row['source'] == 'rathan']
        self.cache = {row['sample_id']: ROOT / row['cache_path'] for row in caches['records']}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        array = np.load(self.cache[row['sample_id']], allow_pickle=False)
        if array.shape[0] != 1 or row['label'] != 0:
            raise ValueError('Rathan auxiliary input must be one Idle still frame')
        return tensor(array), row['label'], row['sample_id']
