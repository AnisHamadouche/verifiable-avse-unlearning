"""Synthetic-only regression: data lineage, delete, retrain, cross-attention and receipts."""
import numpy as np
import torch
from privacy_backend import Lab
from evidence import verify_chain
import json

def test_complete_privacy_workflow(tmp_path):
    torch.set_num_threads(2)
    lab=Lab(tmp_path/'experiment',tmp_path/'keys')
    for i in range(12):
        rng=np.random.default_rng(i)
        audio=rng.normal(0,.02,32000).astype('float32')
        video=np.full((20,48,48),i*5,dtype='uint8')
        lab.add(audio,video,f'sample_{i}',split='train' if i<10 else 'test')
    lab.train_original(epochs=1,batch_size=4,architecture='cross_attention')
    deleted_id=lab.active('train')[0]['id']
    impact=lab.delete([deleted_id])['impact']
    assert 'original.pt' in impact['affected_registered_artifacts']
    assert not any(r['id']==deleted_id for r in lab.active('train'))
    result=lab.unlearn(epochs=1,batch_size=4,approximate=True)
    assert len(result['privacy_audit'])==3
    assert (lab.root/'unlearning_privacy_metrics.csv').exists()
    lab.issue_certificate()
    chain=json.loads((lab.root/'event_chain.json').read_text())
    assert verify_chain(chain)
