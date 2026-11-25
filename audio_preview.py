"""In-memory per-sample preview of clean, two-speaker babble, and AVSE output.
Uses active (not withdrawn) training records as noise candidates.
"""
import numpy as np
import torch
from avse_unlearning import TinyAVSE, SR, mix_babble, digest


def make_preview_audio(lab, sample_id, checkpoint=None, seed=42, snr=0.0):
    row = next((r for r in lab.records() if r['id'] == sample_id), None)
    if row is None or row['status'] != 'active':
        raise FileNotFoundError('Sample does not exist or its key is revoked')
    clean, frames = lab.vault.load(sample_id)
    pool = [r for r in lab.active('train') if r['id'] != sample_id]
    if len(pool) < 2:
        raise ValueError('At least two other active training records are needed for babble noise')
    # Same deterministic seed derivation used by the training dataset.
    local_seed = int(digest(sample_id + str(seed + 5000)), 16) % (2**32)
    noisy = mix_babble(clean, pool, sample_id, local_seed, snr, lab.vault)
    if checkpoint is None:
        return clean, noisy, None
    model = TinyAVSE(visual=True).cpu()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    with torch.inference_mode():
        estimate = model(torch.tensor(noisy[None], dtype=torch.float32),
                         torch.tensor(frames[None, :, None],dtype=torch.float32)/255.).squeeze(0)
    return clean, noisy, estimate.numpy().astype('float32')
