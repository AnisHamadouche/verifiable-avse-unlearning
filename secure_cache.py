"""Prototype per-sample encryption and key deletion for AVSE-derived caches.

Security boundary: ciphertext files and a separately protected local key directory.
This demo does NOT attest storage deletion or prevent key/plaintext exfiltration.
Production: put KEKs/DEKs in user-controlled KMS/HSM with audited non-exportable keys.
"""
import hashlib
import io
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _write_private(path: Path, data: bytes):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Atomic create: no overwriting an existing key/file.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


class EncryptedCache:
    """Encrypted cache with user-held per-record DEKs in a separate directory."""

    def __init__(self, out_dir, key_dir=None):
        self.out = Path(out_dir).resolve()
        self.cache_dir = self.out / 'encrypted_cache'
        self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.key_dir = Path(key_dir).expanduser().resolve() if key_dir else self.out / 'private_keys'
        self.key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _paths(self, sample_id):
        if not isinstance(sample_id, str) or not sample_id or any(c not in '0123456789abcdef' for c in sample_id):
            raise ValueError('sample_id must be a hex digest')
        return self.cache_dir / f'{sample_id}.enc', self.key_dir / f'{sample_id}.key'

    def store(self, sample_id, clean, frames):
        ciphertext_path, key_path = self._paths(sample_id)
        if ciphertext_path.exists() or key_path.exists():
            raise FileExistsError('Encrypted cache or key already exists')
        with io.BytesIO() as f:
            np.savez_compressed(f, clean=clean, frames=frames)
            plaintext = f.getvalue()
        key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)  # unique per newly generated per-sample key
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, sample_id.encode('ascii'))
        # ciphertext contains no plaintext key; key is kept in a separate path.
        _write_private(key_path, key)
        try:
            _write_private(ciphertext_path, b'AVSE1' + nonce + ciphertext)
        except Exception:
            key_path.unlink(missing_ok=True)
            raise
        return str(ciphertext_path)

    def load(self, sample_id):
        ciphertext_path, key_path = self._paths(sample_id)
        key = key_path.read_bytes()  # raises FileNotFoundError if revoked
        payload = ciphertext_path.read_bytes()
        if not payload.startswith(b'AVSE1') or len(payload) < 5 + 12 + 16:
            raise ValueError('Invalid encrypted cache')
        nonce = payload[5:17]
        clear = AESGCM(key).decrypt(nonce, payload[17:], sample_id.encode('ascii'))
        with np.load(io.BytesIO(clear), allow_pickle=False) as z:
            return z['clean'].copy(), z['frames'].copy()

    def revoke(self, sample_id, remove_ciphertext=False):
        ciphertext_path, key_path = self._paths(sample_id)
        before = ciphertext_path.exists() and key_path.exists()
        if not before:
            raise FileNotFoundError(f'Cannot revoke absent sample/key: {sample_id}')
        sha = hashlib.sha256(ciphertext_path.read_bytes()).hexdigest()
        key_path.unlink()  # a deletion *simulation*, not assured sanitisation of key media/backups
        if remove_ciphertext:
            ciphertext_path.unlink(missing_ok=True)
        try:
            self.load(sample_id)
        except FileNotFoundError:
            blocked = True
        else:
            blocked = False
        receipt = {
            'sample_id': sample_id,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'action': 'remove_local_dek',
            'ciphertext_sha256': sha,
            'ciphertext_retained': ciphertext_path.exists(),
            'key_absent_in_active_vault': not key_path.exists(),
            'decrypt_blocked_by_missing_key': blocked,
            'limitations': 'No independent proof of complete key sanitisation, no attestation of backups or external copies, no ML forgetting guarantee.'
        }
        return receipt


def self_test(tmp_dir):
    vault = EncryptedCache(tmp_dir)
    sample_id = hashlib.sha256(b'demonstration clip').hexdigest()[:20]
    audio = np.arange(160, dtype='float32')
    frames = np.arange(48 * 48, dtype='uint8').reshape(1, 48, 48)
    vault.store(sample_id, audio, frames)
    decoded_audio, decoded_frames = vault.load(sample_id)
    assert np.array_equal(decoded_audio, audio) and np.array_equal(decoded_frames, frames)
    encrypted_path, key_path = vault._paths(sample_id)
    assert not audio.tobytes() in encrypted_path.read_bytes()
    receipt = vault.revoke(sample_id, remove_ciphertext=False)
    assert receipt['decrypt_blocked_by_missing_key'] and encrypted_path.exists() and not key_path.exists()
    return receipt


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as path:
        print(json.dumps(self_test(path), indent=2))
