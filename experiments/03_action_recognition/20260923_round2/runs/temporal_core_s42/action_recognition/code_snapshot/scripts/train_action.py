#!/usr/bin/env python3
"""Train from scratch on local Earthmoving; test only after validation selection."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from earthmoving.common import CLASSES, sha256, write_json, write_csv, write_jsonl
from earthmoving.data import EarthmovingClips, sample_frames
from earthmoving.model import TinyActionCNN
from earthmoving.metrics import classification_metrics, save_confusion


def sync(device):
    if str(device) == 'mps':
        torch.mps.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    x, _, _ = next(iter(loader))
    for _ in range(3):
        model(x.to(device))
    sync(device)
    started = time.perf_counter()
    y, pred, confidence, indices = [], [], [], []
    model_seconds, nframes = 0.0, 0
    for x, target, idx in loader:
        nframes += x.shape[0] * x.shape[2]
        x = x.to(device)
        sync(device)
        step = time.perf_counter()
        probs = model(x).softmax(-1)
        sync(device)
        model_seconds += time.perf_counter() - step
        conf, p = probs.cpu().max(-1)
        y.extend(target.tolist()); pred.extend(p.tolist())
        confidence.extend(conf.tolist()); indices.extend(idx.tolist())
    sync(device)
    elapsed = time.perf_counter() - started
    metrics, classes, cm = classification_metrics(y, pred)
    timing = {'input_frames': nframes, 'clips': len(y), 'end_to_end_seconds': elapsed,
              'model_seconds': model_seconds, 'input_frames_per_second': nframes / elapsed,
              'clips_per_second': len(y) / elapsed, 'model_input_frames_per_second': nframes / model_seconds,
              'timed_scope': 'JPEG read/decode + crop/resize + tensor transfer + model/softmax + collect; excludes CSV/plot writing',
              'model_scope': 'forward + softmax with device synchronization; excludes transfer and preprocessing',
              'warmup_batches': 3, 'includes_detector_tracker': False}
    return metrics, classes, cm, list(zip(indices, y, pred, confidence)), timing


def export_test(model, dataset, cfg, out, device):
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    metrics, classes, cm, results, timing = evaluate(model, loader, device)
    write_csv(out / 'metrics.csv', [{'model': cfg['model'], **metrics, 'fps': timing['input_frames_per_second']}])
    write_csv(out / 'class_metrics.csv', classes)
    write_json(out / 'timing.json', timing)
    write_json(out / 'confusion_matrix.json', cm.tolist())
    save_confusion(out / 'confusion_matrix.png', cm)
    predictions, details = [], []
    for idx, true, pred, conf in results:
        r = dataset.rows[idx]
        predictions.append({'video_id': r['video_id'], 'start_time': r['start_frame'] / cfg['fps'],
                            'end_time': r['end_frame'] / cfg['fps'], 'true_action': CLASSES[true],
                            'pred_action': CLASSES[pred], 'confidence': conf})
        details.append({**r, 'true_id': true, 'pred_id': pred, 'confidence': conf,
                        'sampled_frames': sample_frames(r['start_frame'], r['end_frame'], cfg['frames']).tolist()})
    write_csv(out / 'predictions.csv', predictions)
    write_jsonl(out / 'prediction_details.jsonl', details)
    return metrics, timing


def main(args):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.config.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a historical run.
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['device'])
    if str(device) == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS unavailable; select CPU explicitly in a new config')
    d = args.derived.resolve()
    split = json.loads((d / 'split.json').read_text())
    cfg.update({'data_root': str(root / 'Data'), 'derived': str(d), 'classes': CLASSES,
                'split_sha256': sha256(d / 'split.json'),
                'input_hashes': {n: sha256(d / n) for n in ['clips.jsonl', 'windows.jsonl', 'boxes.jsonl', 'source_manifest.jsonl']},
                'command': ' '.join(sys.argv), 'code_commit': 'none; code snapshot and SHA256 provided',
                'deterministic_algorithms': True, 'torch_num_threads': 4,
                'determinism_limit': 'same software/hardware intended; cross-release bitwise equality not claimed'})
    # JSON is valid YAML 1.2; avoids installing PyYAML into the shared parent environment.
    write_json(out / 'config.yaml', cfg)
    shutil.copy2(d / 'split.json', out / 'split.json')
    shutil.copy2(root / 'experiments/01_dataset/environment.json', out / 'environment.json')
    shutil.copy2(root / 'configs/environment.lock.txt', out / 'environment.lock.txt')
    code = []
    for folder in ['earthmoving', 'scripts', 'tests', 'configs']:
        for p in sorted((root / folder).glob('*')):
            if p.is_file() and p.suffix in ['.py', '.json', '.txt']:
                dest = out / 'code_snapshot' / p.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                code.append({'file': str(p.relative_to(root)), 'sha256': sha256(p)})
    write_json(out / 'code_manifest.json', code)
    train = EarthmovingClips(root, d, 'train', cfg['frames'], cfg['input_size'], training=True)
    val = EarthmovingClips(root, d, 'val', cfg['frames'], cfg['input_size'])
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                        generator=torch.Generator().manual_seed(cfg['seed']))
    val_loader = DataLoader(val, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    model = TinyActionCNN().to(device)
    counts = np.array(split['raw_interval_support']['train'])
    assert np.all(counts > 0)
    weights = 1 / np.sqrt(counts)
    weights /= weights.mean()
    write_json(out / 'training_support.json', {'raw_interval_support': counts.tolist(), 'weights': weights.tolist(),
                                             'training_windows': len(train), 'validation_clips': len(val)})
    lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    logs, best, best_epoch = [], -1.0, None
    started = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        loss_total, seen = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            assert logits.shape == (len(y), 5)
            loss = lossfn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward(); opt.step()
            loss_total += loss.item() * len(y); seen += len(y)
        metrics, _, _, _, _ = evaluate(model, val_loader, device)
        log = {'epoch': epoch, 'train_loss_batch_weighted_mean': loss_total / seen,
               **{'val_' + k: v for k, v in metrics.items()}, 'elapsed_seconds': time.perf_counter() - started}
        logs.append(log)
        write_csv(out / 'training_log.csv', logs)
        print(json.dumps(log), flush=True)
        if metrics['macro_f1'] > best:
            best, best_epoch = metrics['macro_f1'], epoch
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': cfg, 'epoch': epoch, 'val_macro_f1': best}, out / 'best.pt')
    duration = time.perf_counter() - started
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    test = EarthmovingClips(root, d, 'test', cfg['frames'], cfg['input_size'])
    metrics, timing = export_test(model, test, cfg, out, device)
    write_json(out / 'run_summary.json', {'selected_epoch': best_epoch, 'val_macro_f1': best,
                                         'training_seconds': duration, 'test_metrics': metrics,
                                         'parameters': sum(p.numel() for p in model.parameters()),
                                         'checkpoint_sha256': sha256(out / 'best.pt'),
                                         'checkpoint_bytes': (out / 'best.pt').stat().st_size})
    (out / 'README.md').write_text(
        '# Earthmoving P0/P1 pilot baseline\n\n'
        f'实际完成 {cfg["epochs"]} epoch、seed={cfg["seed"]} 的单次独立训练，验证集选择 epoch {best_epoch}。'
        'TinyActionCNN 从零训练，无预训练、无补充数据；不是最终模型，也没有完成三种子重复。\n\n'
        f'测试单位：{len(test)} 条原始动作区间各取一个保守内部窗口，XML 真值框，单帧中心采样。'
        f'训练使用 {len(train)} 个窗口，不能将窗口数当独立视频或独立 Move 区间数。\n\n'
        f'测试 Accuracy={metrics["accuracy"]:.6f}，Macro-F1={metrics["macro_f1"]:.6f}。'
        'Precision/Recall/F1 均固定 5 类宏平均、零分母计 0；Move 测试 support=1，无法稳健估计泛化。\n\n'
        'TXT 编号/端点仍不确定：只使用四种候选解释一致的内部帧；缺失或冲突不补 Idle。'
        '25 FPS 来自论文发布方 Ground truth data 检索摘要，本地没有视频时间戳；时间列以此来源条件化。'
        '原始视频互斥，不保证不同工地互斥。\n\n'
        f'指标 fps={timing["input_frames_per_second"]:.2f} 是动作输入图像吞吐；实际范围见 timing.json。'
        '不含检测/跟踪，不代表整系统实时 FPS。预切片分类不能证明自动动作分段。\n\n'
        f'权重 best.pt SHA256: `{sha256(out / "best.pt")}`。代码无 commit，见 code_snapshot/ 和 code_manifest.json。\n\n'
        f'运行命令（项目根目录）: `{cfg["command"]}`。环境、划分和输入索引哈希见 config.yaml；'
        '该文件使用 JSON 形式的合法 YAML 1.2。\n')
    print(json.dumps({'test': metrics, 'out': str(out)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/baseline_smoke.json'))
    p.add_argument('--derived', type=Path, default=Path('derived/earthmoving_v1'))
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
