import json
import numpy as np
import pytest
from privacy_backend import Lab,verify_certificate


def test_key_revocation_and_certificate(tmp_path):
    lab=Lab(tmp_path/'data',tmp_path/'secret')
    a=np.zeros(32000,dtype='float32')
    v=np.zeros((20,48,48),dtype='uint8')
    r1=lab.add(a,v,'one')
    r2=lab.add(a,v,'two')
    assert np.array_equal(lab.vault.load(r1['id'])[0],a)
    result=lab.delete([r1['id']])
    assert result['receipts'][0]['decrypt_blocked_by_missing_key']
    assert np.array_equal(lab.vault.load(r2['id'])[0],a)
    with pytest.raises(FileNotFoundError):lab.vault.load(r1['id'])
    cert=lab.issue_certificate()
    assert verify_certificate(cert)
    cert['report']['withdrawn_sample_ids']=[]
    with pytest.raises(Exception):verify_certificate(cert)


def test_cannot_delete_every_sample(tmp_path):
    lab=Lab(tmp_path/'data',tmp_path/'secret')
    r=lab.add(np.zeros(32000,'float32'),np.zeros((20,48,48),'uint8'),'single')
    with pytest.raises(ValueError,match='every'):
        lab.delete([r['id']])
