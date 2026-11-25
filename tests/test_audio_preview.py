import numpy as np
import pytest
from privacy_backend import Lab
from audio_preview import make_preview_audio


def test_preview_triplet_and_revocation(tmp_path):
    lab = Lab(tmp_path/'data', tmp_path/'keys')
    ids=[]
    for i in range(4):
        rng=np.random.default_rng(i)
        wav=rng.normal(0,0.1,32000).astype('float32')
        lip=np.full((20,48,48),i*20,dtype='uint8')
        ids.append(lab.add(wav,lip,str(i))['id'])
    clean,noisy,enhanced=make_preview_audio(lab,ids[0])
    assert clean.shape==noisy.shape==(32000,)
    assert enhanced is None
    assert np.mean((clean-noisy)**2)>0
    lab.delete([ids[0]])
    with pytest.raises(FileNotFoundError):
        make_preview_audio(lab,ids[0])
    # Remaining record still works, with 2 other available masker clips
    clean,noisy,_=make_preview_audio(lab,ids[1])
    assert len(noisy)==32000
