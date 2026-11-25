import json
import numpy as np
import pytest
from lineage import register, impact
from evidence import chain,verify_chain,verify_certificate_pinned
from privacy_backend import Lab

def test_lineage_transitive_impact(tmp_path):
    register(tmp_path,'a','sample',[])
    register(tmp_path,'model.pt','model',['a'])
    register(tmp_path,'report.csv','report',['model.pt'])
    assert set(impact(tmp_path,['a']))=={'a','model.pt','report.csv'}

def test_chain_detects_modification_and_requires_external_root():
    events=[{'action':'added','id':'a'},{'action':'deleted','id':'a'}]
    entries=chain(events)
    root=verify_chain(entries)
    assert len(root)==64
    entries[0]['event']['id']='b'
    with pytest.raises(ValueError):verify_chain(entries)

def test_pinned_public_key_and_certificate(tmp_path):
    lab=Lab(tmp_path/'data',tmp_path/'private')
    a=np.zeros(32000,dtype='float32');v=np.zeros((20,48,48),dtype='uint8')
    sid=lab.add(a,v,'one')['id'];lab.add(a,v,'two')
    lab.delete([sid]);envelope=lab.issue_certificate()
    key=(lab.root/'verifier_public_key.hex').read_text().strip()
    assert verify_certificate_pinned(envelope,key,envelope['report']['event_chain_root'])
    with pytest.raises(ValueError):verify_certificate_pinned(envelope,'00'*32)
    with pytest.raises(ValueError):verify_certificate_pinned(envelope,key,'ff'*32)
    assert (lab.root/'withdrawal_impact.json').exists()
    assert json.loads((lab.root/'model_serving_policy.json').read_text())['original.pt'].startswith('retired')
