from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from earthmoving.common import read_jsonl


def sample_frames(start, end, frames):
    if end <= start or frames < 1:
        raise ValueError('Nonempty half-open frame interval and frames >= 1 required')
    if frames == 1:
        return np.array([(start + end - 1) // 2], dtype=int)
    return np.linspace(start, end - 1, frames).round().astype(int)


def box_lookup(rows):
    out = {}
    for r in rows:
        key = (str(r['video_id']), str(r['equipment_id']), int(r['frame']))
        if key in out:
            raise ValueError(f'Duplicate track row: {key}')
        out[key] = tuple(float(r[k]) for k in ['x1', 'y1', 'x2', 'y2'])
    return out


def load_clip(root, row, boxes, frames, size):
    images = []
    for t in sample_frames(row['start_frame'], row['end_frame'], frames):
        v, eid = str(row['video_id']), str(row['equipment_id'])
        bbox = boxes[(v, eid, int(t))]  # No missing-box substitution or interpolation.
        with Image.open(Path(root) / 'Data' / 'frames_ce' / v / f'{v}_I{t:05d}.jpg') as im:
            w, h = im.size
            x1, y1, x2, y2 = bbox
            if not np.isfinite(bbox).all() or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f'Invalid box {v}/{eid}/{t}: {bbox}')
            # Pixel coordinates interpreted as geometric edges; floor/ceil and clip, no +1 guess.
            bounds = (int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2)))
            crop = im.convert('RGB').crop(bounds).resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.array(crop, dtype=np.float32) / 255.0)
    array = np.stack(images).transpose(3, 0, 1, 2).copy()  # [C,T,H,W]
    return torch.from_numpy(array)


class EarthmovingClips(Dataset):
    def __init__(self, root, derived, split, frames=1, size=96, training=False):
        self.root, self.frames, self.size = root, frames, size
        index = 'windows.jsonl' if training else 'clips.jsonl'
        self.rows = [r for r in read_jsonl(Path(derived) / index) if r['split'] == split]
        self.boxes = box_lookup(read_jsonl(Path(derived) / 'boxes.jsonl'))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return load_clip(self.root, row, self.boxes, self.frames, self.size), row['label'], i
