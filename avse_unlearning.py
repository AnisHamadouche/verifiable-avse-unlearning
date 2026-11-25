#!/usr/bin/env python3
"""LRS3 audio-visual enhancement + sample deletion/unlearning research baseline.

Requires ffmpeg/ffprobe on PATH; no torchaudio or torchvision video I/O.
This is a test of training-set removal, not cryptographic proof of forgetting.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from secure_cache import EncryptedCache
from speech_metrics import calculate as speech_metrics

SR = 16000
DURATION = 2.0
NSAMPLES = int(SR * DURATION)
FPS = 10
NFRAMES = int(FPS * DURATION)
VWIDTH = 48
FFT = 512
HOP = 256
FREQ = FFT // 2 + 1


def digest(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()[:20]


def ffmpeg_decode(src):
    """Decode first two seconds as audio and as low-res visual face-track frames."""
    common = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(src), '-t', str(DURATION)]
    acmd = common + ['-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
    vcmd = common + ['-an', '-vf', f'fps={FPS},scale={VWIDTH}:{VWIDTH},format=gray',
                     '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1']
    a = np.frombuffer(subprocess.run(acmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout,
                      dtype='<f4').copy()
    vb = subprocess.run(vcmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if len(a) < SR // 2 or len(vb) < VWIDTH * VWIDTH * 5:
        raise ValueError('Clip too short or missing audio/video')
    a = np.pad(a[:NSAMPLES], (0, max(0, NSAMPLES - len(a))))
    v = np.frombuffer(vb, dtype=np.uint8).copy()
    v = v[:(len(v) // (VWIDTH * VWIDTH)) * VWIDTH * VWIDTH].reshape(-1, VWIDTH, VWIDTH)
    if len(v) < NFRAMES:
        v = np.concatenate([v, np.repeat(v[-1:], NFRAMES - len(v), axis=0)])
    v = v[:NFRAMES]
    return a.astype('float32'), v


def scan(root, subset, max_files, seed):
    paths = sorted((root / subset).rglob('*.mp4'))
    if not paths:
        raise FileNotFoundError(f'No MP4 files found in {root / subset}')
    rng = random.Random(seed + (0 if subset == 'trainval' else 999))
    rng.shuffle(paths)
    return paths[:max_files]


def cache_items(root, work, subset, n, seed, vault):
    rows = []
    paths = scan(root, subset, n * 2, seed)
    for p in paths:
        if len(rows) >= n:
            break
        rel = p.relative_to(root).as_posix()
        key = digest(rel)
        cachefile, keyfile = vault._paths(key)
        try:
            if not cachefile.exists() and not keyfile.exists():
                audio, video = ffmpeg_decode(p)
                vault.store(key, audio, video)
            # Never silently regenerate revoked records: a missing key is an error.
            audio, video = vault.load(key)
            assert audio.shape == (NSAMPLES,)
            assert video.shape == (NFRAMES, VWIDTH, VWIDTH)
            rows.append({'id': key, 'source': rel, 'subset': subset, 'cache': str(cachefile)})
        except (OSError, subprocess.CalledProcessError, ValueError, AssertionError) as exc:
            print(f'SKIP {rel}: {exc}', flush=True)
            # Preserve files for diagnosis; do not automatically recreate revoked keys.
    if len(rows) < 10:
        raise RuntimeError(f'Only {len(rows)} valid clips for {subset}. Need >= 10.')
    return rows


def group_split(rows, seed, val_fraction=0.15):
    """Prevent same TED source directory appearing in both train and validation."""
    groups = {}
    for row in rows:
        parts = Path(row['source']).parts
        group = parts[1] if len(parts) > 2 else row['id']
        groups.setdefault(group, []).append(row)
    names = list(groups)
    random.Random(seed).shuffle(names)
    nval = max(1, int(len(names) * val_fraction))
    vgroups = set(names[:nval])
    train = [r for g in names if g not in vgroups for r in groups[g]]
    val = [r for g in names if g in vgroups for r in groups[g]]
    if len(train) < 8 or len(val) < 2:
        raise RuntimeError('Not enough distinct sources for a group-disjoint split; increase --train-clips.')
    return train, val


def load_clean(row, vault):
    return vault.load(row['id'])[0]


def mix_babble(clean, pool, target_id, seed, snr_db, vault):
    """Babble from other LRS3 speech clips; clean target is never used as its own masker."""
    candidates = [r for r in pool if r['id'] != target_id]
    if len(candidates) < 2:
        raise ValueError('Need at least two distinct babble clips')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=2, replace=False)
    babble = np.zeros_like(clean)
    for j in idx:
        other = load_clean(candidates[int(j)], vault)
        shift = int(rng.integers(0, len(other)))
        babble += np.roll(other, shift) / 2
    babble -= babble.mean()
    p_clean = np.mean(clean ** 2) + 1e-9
    p_noise = np.mean(babble ** 2) + 1e-9
    babble *= math.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    return (clean + babble).astype('float32')


class AVSEDataset(Dataset):
    def __init__(self, rows, masker_pool, vault, seed=42, snr_db=0):
        self.rows = rows
        self.pool = masker_pool
        self.seed = seed
        self.snr_db = snr_db
        self.vault = vault

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean, frames = self.vault.load(row['id'])
        video = frames.astype('float32') / 255.0
        local_seed = int(digest(row['id'] + str(self.seed)), 16) % (2 ** 32)
        noisy = mix_babble(clean, self.pool, row['id'], local_seed, self.snr_db, self.vault)
        return (torch.from_numpy(noisy), torch.from_numpy(video[:, None]),
                torch.from_numpy(clean), row['id'])


class TinyAVSE(nn.Module):
    """STFT magnitude + visual CNN, GRU mask. Compact educational AVSE baseline."""
    def __init__(self, visual=True):
        super().__init__()
        self.visual = visual
        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 32), nn.ReLU())
        self.audio_net = nn.Sequential(nn.Linear(FREQ, 128), nn.ReLU(), nn.LayerNorm(128))
        self.gru = nn.GRU(128 + 32, 128, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(256, FREQ), nn.Sigmoid())
        self.register_buffer('window', torch.hann_window(FFT))

    def forward(self, noisy, frames):
        stft = torch.stft(noisy, FFT, HOP, window=self.window, return_complex=True)
        mag = stft.abs().transpose(1, 2)
        audio = self.audio_net(torch.log1p(mag))
        B, V, C, H, W = frames.shape
        if self.visual:
            visual = self.visual_net(frames.reshape(B * V, C, H, W)).reshape(B, V, -1)
            visual = F.interpolate(visual.transpose(1, 2), size=audio.shape[1],
                                   mode='linear', align_corners=False).transpose(1, 2)
        else:
            visual = audio.new_zeros(B, audio.shape[1], 32)
        z, _ = self.gru(torch.cat([audio, visual], -1))
        mask = self.head(z).transpose(1, 2)
        estimate = torch.istft(stft * mask, FFT, HOP, window=self.window, length=noisy.shape[-1])
        return estimate


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sisnr(pred, ref):
    pred = pred - pred.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    target = torch.sum(pred * ref, -1, keepdim=True) * ref / (torch.sum(ref * ref, -1, keepdim=True) + 1e-8)
    residual = pred - target
    return 10 * torch.log10((target.square().sum(-1) + 1e-8) / (residual.square().sum(-1) + 1e-8))


def train_model(train_rows, val_rows, pool, args, destination, vault, mode='from_scratch', source=None):
    seed_all(args.seed)
    model = TinyAVSE(visual=not args.audio_only).to(args.device)
    if source:
        model.load_state_dict(torch.load(source, map_location='cpu', weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainloader = DataLoader(AVSEDataset(train_rows, pool, vault, args.seed, args.snr),
                             batch_size=args.batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(AVSEDataset(val_rows, pool, vault, args.seed + 3000, args.snr),
                           batch_size=args.batch_size, shuffle=False, num_workers=0)
    best = float('inf')
    epochs = args.finetune_epochs if mode == 'finetune' else args.epochs
    records = []
    start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        tloss = 0.0
        for noisy, frames, clean, _ in trainloader:
            noisy, frames, clean = [x.to(args.device) for x in (noisy, frames, clean)]
            estimate = model(noisy, frames)
            loss = F.l1_loss(estimate, clean)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            tloss += loss.item() * len(noisy)
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for noisy, frames, clean, _ in valloader:
                noisy, frames, clean = [x.to(args.device) for x in (noisy, frames, clean)]
                vloss += F.l1_loss(model(noisy, frames), clean).item() * len(noisy)
        tloss /= len(train_rows)
        vloss /= len(val_rows)
        records.append({'model': destination.stem, 'epoch': epoch + 1,
                        'train_l1': tloss, 'val_l1': vloss})
        print(f'{destination.stem} epoch {epoch+1}/{epochs}: train L1 {tloss:.5f} val L1 {vloss:.5f}', flush=True)
        if vloss < best:
            best = vloss
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, destination)
    print(f'{destination.stem} training_seconds={time.monotonic()-start:.1f}')
    return records


def evaluate(model_path, rows, pool, args, label, vault):
    model = TinyAVSE(visual=not args.audio_only).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    dataset = AVSEDataset(rows, pool, vault, args.seed + 5000, args.snr)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    result = []
    with torch.no_grad():
        for noisy, frames, clean, ids in dl:
            noisy, frames, clean = [x.to(args.device) for x in (noisy, frames, clean)]
            enhanced = model(noisy, frames)
            clean_sisnr = sisnr(enhanced, clean)
            noisy_sisnr = sisnr(noisy, clean)
            for i, clip_id in enumerate(ids):
                row = {'model': label, 'sample_id': clip_id,
                       'input_si_snr': float(noisy_sisnr[i]),
                       'output_si_snr': float(clean_sisnr[i]),
                       'si_snri': float(clean_sisnr[i] - noisy_sisnr[i])}
                if not args.skip_intelligibility:
                    row.update(speech_metrics(clean[i].cpu().numpy(), noisy[i].cpu().numpy(),
                                              enhanced[i].cpu().numpy(), sr=SR,
                                              include_pesq=args.pesq))
                result.append(row)
    return result


def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/home/ah2161/LRS3-TED')
    parser.add_argument('--out-dir', default='runs/lrs3_avse_delete')
    parser.add_argument('--train-clips', type=int, default=160)
    parser.add_argument('--test-clips', type=int, default=24)
    parser.add_argument('--delete-fraction', type=float, default=0.10)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--finetune-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snr', type=float, default=0.0)
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('--skip-intelligibility', action='store_true', help='Legacy SI-SNR-only evaluation')
    parser.add_argument('--pesq', action='store_true', help='Compute PESQ-WB (optional; review PESQ licence)')
    parser.add_argument('--key-dir', default=None, help='User-controlled location for per-sample DEKs; ideally separate from the ciphertext disk.')
    parser.add_argument('--retain-ciphertext-after-delete', action='store_true', help='Retain withdrawn encrypted cache files after removing their keys, for explicit access-revocation tests.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--purge-derived-on-delete', action='store_true', help='Purge deleted clips derived cache files at end (never original MP4s)')
    args = parser.parse_args()
    if not 0 < args.delete_fraction < .5:
        parser.error('--delete-fraction must be between 0 and 0.5')
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg binary is required; install with sudo apt install ffmpeg')
    root = Path(args.data_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    vault = EncryptedCache(out, args.key_dir)
    print('Preparing encrypted samples (first run decodes MP4 clips)...', flush=True)
    trainval = cache_items(root, out, 'trainval', args.train_clips, args.seed, vault)
    test = cache_items(root, out, 'test', args.test_clips, args.seed, vault)
    train, val = group_split(trainval, args.seed)
    random.Random(args.seed).shuffle(train)
    ndelete = max(1, round(args.delete_fraction * len(train)))
    deleted = train[:ndelete]
    retained = train[ndelete:]
    deleted_ids = {r['id'] for r in deleted}
    assert len(retained) >= 4
    # Training pool excludes official test set; post-deletion pool excludes withdrawn clips.
    initial_pool = train
    remaining_pool = retained
    records = {'settings': vars(args), 'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
               'deleted_ids': sorted(deleted_ids),
               'scope': 'Deletion affects selected encrypted derived cache keys and training membership; original LRS3 MP4 unchanged. Checkpoints retain learned influence.'}
    (out / 'experiment.json').write_text(json.dumps(records, indent=2))
    write_csv(out / 'manifest.csv', trainval + test)
    write_csv(out / 'deletion_requests.csv', [{**r, 'request': 'delete_from_training'} for r in deleted])
    all_history = []
    print(f'Train={len(train)} val={len(val)} test={len(test)} delete={len(deleted)} retained={len(retained)}')
    original = out / 'original.pt'
    retrained = out / 'retrained_without_deleted.pt'
    finetuned = out / 'finetuned_without_deleted.pt'
    all_history += train_model(train, val, initial_pool, args, original, vault)
    # IMPORTANT: no deleted clip can be used as either a training target or babble masker in retraining.
    all_history += train_model(retained, val, remaining_pool, args, retrained, vault)
    all_history += train_model(retained, val, remaining_pool, args, finetuned, vault, mode='finetune', source=original)
    write_csv(out / 'training_history.csv', all_history)
    results = []
    for label, model_path in [('original', original), ('retrained', retrained), ('finetuned', finetuned)]:
        for split, rows in [('test', test), ('withdrawn', deleted), ('retained', retained[:len(deleted)])]:
            # Controlled evaluation: all three models see the same fixed mixtures from retained maskers.
            for r in evaluate(model_path, rows, remaining_pool, args, label, vault):
                r['split'] = split
                results.append(r)
    write_csv(out / 'per_clip_metrics.csv', results)
    for label in ('original', 'retrained', 'finetuned'):
        subset = [r for r in results if r['model'] == label and r['split'] == 'test']
        print(f'{label} held-out test SI-SNRi = {np.mean([r["si_snri"] for r in subset]):+.3f} dB')
    if args.purge_derived_on_delete:
        receipts = [vault.revoke(row['id'], remove_ciphertext=not args.retain_ciphertext_after_delete) for row in deleted]
        assert all(x['decrypt_blocked_by_missing_key'] for x in receipts)
        (out / 'key_revocation_receipts.json').write_text(json.dumps(receipts, indent=2))
        # remove withdrawn records from active manifest
        write_csv(out / 'manifest.csv', [r for r in trainval + test if r['id'] not in deleted_ids])
        print('Revoked derived-cache keys; original MP4s are untouched.')
    print('Outputs:', out)
    print('NOTE: retraining is a reference removal experiment, not proof of forgetting or protection from memorisation.')
    print('NOTE: original and finetuned checkpoints may still contain influence from deleted data.')
    print('NOTE: local key deletion does not prove sanitisation of key backups or prevent earlier exfiltration.')

if __name__ == '__main__':
    main()
