"""Round-2 inputs: immutable v1 evaluation clips, interval sampling and end supervision."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from earthmoving.common import read_jsonl, write_jsonl, write_json, sha256
from earthmoving.data import box_lookup, sample_frames

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'derived/earthmoving_v1'
D2 = ROOT / 'derived/earthmoving_round2'


def prepare():
    """No model predictions. Partition labels by already-frozen original video membership."""
    if (D2 / 'manifest.json').exists():
        verify_inputs()
        return
    D2.mkdir(parents=True, exist_ok=False)
    split = json.loads((V1 / 'split.json').read_text())
    clips = read_jsonl(V1 / 'clips.jsonl')
    windows = read_jsonl(V1 / 'windows.jsonl')
    boxes = box_lookup(read_jsonl(V1 / 'boxes.jsonl'))
    stats = {}
    for subset, videos in split['groups'].items():
        cc = [r for r in clips if r['split'] == subset]
        ww = [r for r in windows if r['split'] == subset]
        continuous = []
        for v in videos:
            labels = np.load(V1 / f'{v}_consensus.npy')
            owner = np.full(len(labels), -1, dtype=int)
            for i, r in enumerate(cc):
                if r['video_id'] == v:
                    owner[r['core_start']:r['core_end']] = i
            for end in range(64, len(labels) + 1, 8):
                target = end - 1
                if labels[target] < 0 or owner[target] < 0:
                    continue
                r = cc[int(owner[target])]
                if any((v, r['equipment_id'], t) not in boxes for t in range(end - 64, end)):
                    continue
                assert labels[target] == r['label']
                continuous.append({**r, 'sample_id': f'{v}_end{target:05d}',
                                   'start_frame': end - 64, 'end_frame': end,
                                   'target_frame': target, 'supervision': 'last_frame_only',
                                   'crosses_label_boundary': bool(np.any(labels[end-64:end] != labels[target]))})
        for name, rows in [('clips', cc), ('windows', ww), ('continuous', continuous)]:
            write_jsonl(D2 / f'{subset}_{name}.jsonl', rows)
        stats[subset] = {'clips': len(cc), 'windows': len(ww), 'continuous_targets': len(continuous),
                         'continuous_class_support': [sum(r['label'] == c for r in continuous) for c in range(5)],
                         'continuous_raw_interval_groups': len({(r['video_id'], r['raw_line']) for r in continuous}),
                         'continuous_crossing_windows': sum(r['crosses_label_boundary'] for r in continuous)}
        assert all(stats[subset]['continuous_class_support'])
    hashes = {p.name: sha256(p) for p in sorted(D2.glob('*.jsonl'))}
    write_json(D2 / 'manifest.json', {'parent_split_sha256': sha256(V1 / 'split.json'),
               'parent_hashes': {n: sha256(V1 / n) for n in ['clips.jsonl', 'windows.jsonl', 'boxes.jsonl', 'source_manifest.jsonl']},
               'files': hashes, 'stats': stats, 'split': split['groups'],
               'continuous_index': 'global 64-frame history / stride 8, valid consensus last-frame label only; no missing label filled'})


def verify_inputs():
    m = json.loads((D2 / 'manifest.json').read_text())
    assert sha256(V1 / 'split.json') == m['parent_split_sha256']
    for n, h in m['parent_hashes'].items():
        assert sha256(V1 / n) == h, f'Frozen v1 drift: {n}'
    for n, h in m['files'].items():
        assert sha256(D2 / n) == h, f'Round2 index drift: {n}'
    return m


@lru_cache(maxsize=256)
def source_image(video, t):
    with Image.open(ROOT / f'Data/frames_ce/{video}/{video}_I{t:05d}.jpg') as im:
        return im.convert('RGB')


def frame_ids(row, frames):
    # Critical difference from pilot: a one-frame model sees the SAME endpoint as the temporal model.
    if frames == 1:
        return np.array([row['end_frame'] - 1])
    return sample_frames(row['start_frame'], row['end_frame'], frames)


def resize_crop(im, size, letterbox):
    if not letterbox:
        return im.resize((size, size), Image.Resampling.BILINEAR)
    ratio = min(size / im.width, size / im.height)
    small = im.resize((max(1, round(im.width * ratio)), max(1, round(im.height * ratio))), Image.Resampling.BILINEAR)
    result = Image.new('RGB', (size, size), (128, 128, 128))
    result.paste(small, ((size-small.width)//2, (size-small.height)//2))
    return result


def load_input(row, boxes, cfg, augment_seed=None):
    v, eid = row['video_id'], row['equipment_id']
    ids = frame_ids(row, cfg['frames'])
    union = None
    if cfg['crop'] == 'union':
        history = np.array([boxes[v, eid, t] for t in range(row['start_frame'], row['end_frame'])])
        union = [history[:, 0].min(), history[:, 1].min(), history[:, 2].max(), history[:, 3].max()]
    rng = random.Random(augment_seed)
    brightness = rng.uniform(.85, 1.15) if augment_seed is not None and cfg['augment'] else 1.0
    contrast = rng.uniform(.85, 1.15) if augment_seed is not None and cfg['augment'] else 1.0
    images = []
    for t in ids:
        im = source_image(v, int(t))
        x1, y1, x2, y2 = union if union is not None else boxes[v, eid, int(t)]
        assert np.isfinite([x1, y1, x2, y2]).all() and 0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height
        margin = cfg['margin']
        dx, dy = (x2-x1)*margin, (y2-y1)*margin
        bounds = (max(0, int(np.floor(x1-dx))), max(0, int(np.floor(y1-dy))),
                  min(im.width, int(np.ceil(x2+dx))), min(im.height, int(np.ceil(y2+dy))))
        crop = im.crop(bounds)
        if brightness != 1.0:
            crop = ImageEnhance.Brightness(crop).enhance(brightness)
            crop = ImageEnhance.Contrast(crop).enhance(contrast)
        crop = resize_crop(crop, cfg['input_size'], cfg['letterbox'])
        images.append(np.asarray(crop, dtype=np.float32)/255)
    return torch.from_numpy(np.stack(images).transpose(3, 0, 1, 2).copy())


class Round2Dataset(Dataset):
    def __init__(self, subset, cfg, training=False):
        self.cfg, self.training = cfg, training
        name = ('continuous' if cfg['training_mode'] == 'stream' else
                ('windows' if cfg['sampling'] == 'windows' else 'clips')) if training else 'clips'
        self.rows = read_jsonl(D2 / f'{subset}_{name}.jsonl')
        self.boxes = box_lookup(read_jsonl(V1 / 'boxes.jsonl'))
        groups = defaultdict(list)
        for i, r in enumerate(self.rows):
            groups[r['video_id'], r['raw_line']].append(i)
        self.groups = list(groups.values())
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if not self.training:
            self.schedule = [(r, None) for r in self.rows]
            return
        rng = random.Random(self.cfg['seed'] * 100003 + epoch)
        schedule = []
        for _ in range(self.cfg['samples_per_epoch']):
            i = rng.randrange(len(self.rows)) if self.cfg['sampling'] == 'windows' else rng.choice(rng.choice(self.groups))
            row = dict(self.rows[i])
            if self.cfg['sampling'] == 'intervals' and self.cfg['training_mode'] == 'core':
                n = min(64, row['core_end'] - row['core_start'])
                start = rng.randint(row['core_start'], row['core_end'] - n)
                row.update(start_frame=start, end_frame=start+n)
            schedule.append((row, rng.randrange(2**32)))
        self.schedule = schedule

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, i):
        r, seed = self.schedule[i]
        return load_input(r, self.boxes, self.cfg, seed), r['label'], i


def count_groups(dataset):
    counts = Counter(dataset.rows[g[0]]['label'] for g in dataset.groups)
    return [counts[c] for c in range(5)]
