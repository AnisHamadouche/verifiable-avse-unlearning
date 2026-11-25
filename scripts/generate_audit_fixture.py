"""Synthetic-only fixture: independent of LRS3 and real recordings."""
import json,sys
from pathlib import Path
# Allow direct execution from repository root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from privacy_backend import Lab

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('audit_fixture')
keys=Path(sys.argv[2]) if len(sys.argv)>2 else Path('audit_fixture_keys')
lab=Lab(root,keys)
for i in range(9):
    a=np.random.default_rng(i).normal(size=32000).astype('float32')*.05
    f=np.full((20,48,48),i,dtype='uint8')
    lab.add(a,f,f'synthetic_{i}',split='train' if i<8 else 'test',source='synthetic')
# Mark original as simulated without creating or training a checkpoint.
revoked=lab.active('train')[0]['id']
lab.delete([revoked]);lab.issue_certificate()
print(json.dumps({'root':str(root),'keys':str(keys),'revoked':revoked}))
