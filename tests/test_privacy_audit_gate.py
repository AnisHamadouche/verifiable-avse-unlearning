from pathlib import Path
import numpy as np
from privacy_backend import Lab
from scripts.privacy_audit import audit

def test_gate_detects_missing_revoked_key_and_retains_active(tmp_path):
    lab=Lab(tmp_path/'run',tmp_path/'keys')
    for i in range(3):
        lab.add(np.zeros(32000,dtype='float32'),np.zeros((20,48,48),dtype='uint8'),str(i))
    selected=lab.active('train')[0]['id']
    lab.delete([selected]);lab.issue_certificate()
    assert audit(lab.root,lab.key_root)['passed']
    # Regression test: an accidentally restored key must fail audit.
    (lab.key_root/'samples'/f'{selected}.key').write_bytes(b'x'*32)
    assert not audit(lab.root,lab.key_root)['passed']

def test_wrong_trust_anchor_rejected(tmp_path):
    lab=Lab(tmp_path/'run',tmp_path/'keys')
    for i in range(2):
        lab.add(np.zeros(32000,dtype='float32'),np.zeros((20,48,48),dtype='uint8'),str(i))
    lab.delete([lab.active('train')[0]['id']]);lab.issue_certificate()
    wrong=tmp_path/'wrong.hex';wrong.write_text('00'*32)
    result=audit(lab.root,lab.key_root,require_certificate=True,pinned_key=wrong,witnessed_root='0'*64)
    assert not result['passed']

def test_gate_rejects_corrupted_active_ciphertext(tmp_path):
    lab = Lab(tmp_path/'run', tmp_path/'keys')
    for i in range(3):
        lab.add(np.zeros(32000,dtype='float32'), np.zeros((20,48,48),dtype='uint8'),str(i))
    sid = lab.active('train')[0]['id']
    path = lab.root/'encrypted_cache'/f'{sid}.enc'
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    assert not audit(lab.root, lab.key_root)['passed']


def test_gate_rejects_corrupted_revocation_receipt(tmp_path):
    import json
    lab = Lab(tmp_path/'run', tmp_path/'keys')
    for i in range(3):
        lab.add(np.zeros(32000,dtype='float32'), np.zeros((20,48,48),dtype='uint8'),str(i))
    sid = lab.active('train')[0]['id']
    lab.delete([sid]);lab.issue_certificate()
    receipts_path = lab.root/'revocation_receipts.json'
    receipts = json.loads(receipts_path.read_text())
    receipts[0]['ciphertext_sha256'] = '0'*64
    receipts_path.write_text(json.dumps(receipts))
    assert not audit(lab.root, lab.key_root)['passed']
