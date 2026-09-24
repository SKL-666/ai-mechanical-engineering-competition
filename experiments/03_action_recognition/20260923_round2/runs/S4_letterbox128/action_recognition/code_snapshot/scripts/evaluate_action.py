#!/usr/bin/env python3
"""Re-evaluate an existing checkpoint without any training or model selection."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from earthmoving.common import sha256, write_json
from earthmoving.data import EarthmovingClips
from earthmoving.model import TinyActionCNN
from scripts.train_action import export_test

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = saved['config']
    d = Path(cfg['derived'])
    assert sha256(d / 'split.json') == cfg['split_sha256']
    for name, digest in cfg['input_hashes'].items():
        assert sha256(d / name) == digest, f'Input drift: {name}'
    torch.set_num_threads(cfg['torch_num_threads'])
    torch.use_deterministic_algorithms(cfg['deterministic_algorithms'])
    model = TinyActionCNN().to(cfg['device'])
    model.load_state_dict(saved['state_dict'])
    dataset = EarthmovingClips(Path(cfg['data_root']).parent, d, 'test', cfg['frames'], cfg['input_size'])
    metrics, timing = export_test(model, dataset, cfg, a.out, cfg['device'])
    write_json(a.out / 'evaluation.json', {'checkpoint_sha256': sha256(a.checkpoint), 'metrics': metrics,
                                        'config': cfg, 'command': ' '.join(sys.argv)})
    print(metrics)
