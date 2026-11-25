import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from privacy_backend import Lab
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp); lab=Lab(tmp/'experiment',tmp/'keys')
    sample=lab.add(np.zeros(32000,'float32'),np.zeros((20,48,48),'uint8'),'test_1')['id']
    lab.add(np.zeros(32000,'float32'),np.zeros((20,48,48),'uint8'),'test_2')
    lab.delete([sample]);lab.issue_certificate()
    command=[sys.executable,'verify_independent.py','--certificate',str(lab.root/'audit_certificate.json'),
             '--pinned-public-key',str(lab.root/'verifier_public_key.hex'),
             '--chain',str(lab.root/'event_chain.json')]
    subprocess.run(command,check=True)
