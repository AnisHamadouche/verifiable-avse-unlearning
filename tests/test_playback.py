import io
import wave
import shutil
import numpy as np
import pytest

from playback import audio_wav_bytes, lip_preview_mp4
from privacy_backend import Lab


def test_audio_wav_format():
    payload = audio_wav_bytes(np.zeros(32000, dtype='float32'))
    with wave.open(io.BytesIO(payload), 'rb') as f:
        assert f.getnframes() == 32000
        assert f.getframerate() == 16000
        assert f.getnchannels() == 1


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')
def test_preview_mp4_and_revocation(tmp_path):
    lab=Lab(tmp_path/'experiment',tmp_path/'keys')
    samples=np.sin(2*np.pi*220*np.arange(32000)/16000).astype('float32')*.2
    lips=np.zeros((20,48,48),dtype='uint8')
    lips[:,20:26,12:36]=180
    first=lab.add(samples,lips,'volunteer-a')
    second=lab.add(samples,lips,'volunteer-b')
    audio, frames=lab.vault.load(first['id'])
    video=lip_preview_mp4(audio,frames)
    assert len(video)>1000 and b'ftyp' in video[:100]
    lab.delete([first['id']])
    with pytest.raises(FileNotFoundError):
        lab.vault.load(first['id'])
    assert lab.vault.load(second['id'])[0].size==32000
