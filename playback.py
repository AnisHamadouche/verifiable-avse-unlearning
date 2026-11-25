"""Decrypted AVSE sample preview in memory; no persistent plaintext preview files.

Only the stored mouth-region frames are available; source/full-face video is
not reconstructible. MP4 is generated with FFmpeg and in-memory Linux memfd.
"""
from __future__ import annotations

import io
import os
import subprocess
import wave
import numpy as np


def audio_wav_bytes(audio, sample_rate=16000):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype('<i2').tobytes()
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def lip_preview_mp4(audio, lips, sample_rate=16000, frame_size=384):
    """Create playable, synchronised MP4 in memory on Linux; requires FFmpeg.

    Audio uses an anonymous memfd (RAM-backed) instead of a plaintext disk file.
    """
    import cv2
    lips = np.asarray(lips, dtype=np.uint8)
    if lips.ndim != 3 or lips.shape[1:] != (48, 48) or len(lips) == 0:
        raise ValueError('Expected lip frames of shape (N,48,48)')
    if not hasattr(os, 'memfd_create'):
        raise RuntimeError('Synchronous video preview requires Linux memfd_create')
    wav_bytes = audio_wav_bytes(audio, sample_rate)
    duration = len(np.asarray(audio).reshape(-1)) / sample_rate
    if duration <= 0:
        raise ValueError('Audio is empty')
    fps = len(lips) / duration
    frame_bytes = b''.join(
        cv2.cvtColor(cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST),
                     cv2.COLOR_GRAY2BGR).tobytes() for frame in lips
    )
    audio_fd = os.memfd_create('avse-preview-wav', flags=0)
    try:
        os.write(audio_fd, wav_bytes)
        os.lseek(audio_fd, 0, os.SEEK_SET)
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{frame_size}x{frame_size}',
            '-r', f'{fps:.8f}', '-i', 'pipe:0',
            '-i', f'/proc/self/fd/{audio_fd}',
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '96k', '-shortest',
            '-movflags', 'frag_keyframe+empty_moov',
            '-f', 'mp4', 'pipe:1',
        ]
        result = subprocess.run(cmd, input=frame_bytes, capture_output=True,
                                pass_fds=(audio_fd,), timeout=45, check=False)
        if result.returncode != 0:
            raise RuntimeError('FFmpeg preview failed: ' + result.stderr.decode(errors='replace')[-500:])
        if not result.stdout:
            raise RuntimeError('FFmpeg produced an empty video')
        return result.stdout
    finally:
        os.close(audio_fd)
