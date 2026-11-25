"""Local prototype: encrypted clips, subset deletion, retraining, signed audit statements.
The operator controls signing and storage; signatures prove authorship/integrity only.
"""
import csv
import hashlib
import io
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from secure_cache import EncryptedCache
from avse_unlearning import (SR, NSAMPLES, NFRAMES, VWIDTH, TinyAVSE, train_model,
                             evaluate, seed_all)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + secrets.token_hex(4) + '.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf8')
    os.replace(temp, path)


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf8')


class Lab:
    def __init__(self, root, key_root=None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_root = Path(key_root or Path.home() / '.local/share/avse-streamlit-keys').expanduser().resolve()
        self.vault = EncryptedCache(self.root, self.key_root / 'samples')
        self.metadata = self.root / 'records.json'
        self.events_path = self.root / 'events.json'
        self.signing_key_file = self.key_root / 'signing_ed25519.pem'
        self.key_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.metadata.exists(): atomic_json(self.metadata, [])
        if not self.events_path.exists(): atomic_json(self.events_path, [])

    def records(self):
        return json.loads(self.metadata.read_text())

    def events(self):
        return json.loads(self.events_path.read_text())

    def event(self, action, **fields):
        event = {'at_utc': timestamp(), 'action': action, **fields}
        log = self.events()
        log.append(event)
        atomic_json(self.events_path, log)
        return event

    def add(self, clean, frames, label, split='train', source='recording'):
        if split not in ('train', 'test'): raise ValueError('Invalid split')
        clean = np.asarray(clean, dtype='float32').flatten()
        frames = np.asarray(frames, dtype='uint8')
        if clean.size < SR // 2 or frames.ndim != 3 or frames.shape[1:] != (VWIDTH,VWIDTH):
            raise ValueError('Need >=0.5s audio and mouth frames of shape [N,48,48]')
        clean = np.pad(clean[:NSAMPLES], (0,max(0,NSAMPLES-len(clean))))
        picks = np.linspace(0, len(frames)-1, NFRAMES).astype(int)
        frames = frames[picks]
        sid = hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:20]
        self.vault.store(sid, clean, frames)
        row = {'id': sid, 'label': str(label)[:120], 'split': split,
               'source': source, 'status': 'active', 'created': timestamp()}
        all_rows = self.records(); all_rows.append(row); atomic_json(self.metadata, all_rows)
        self.event('sample_added', sample_id=sid, split=split, ciphertext_sha256=sha256(self.vault._paths(sid)[0]))
        return row

    def active(self, split=None):
        return [r for r in self.records() if r['status']=='active' and (split is None or r['split']==split)]

    def delete(self, sample_ids):
        selected = set(sample_ids)
        records = self.records()
        eligible = {r['id'] for r in records if r['status']=='active' and r['split']=='train'}
        if not selected or not selected.issubset(eligible):
            raise ValueError('Select one or more active TRAIN samples only')
        if not eligible - selected:
            raise ValueError('Cannot delete every training sample in this prototype')
        receipts=[]
        for row in records:
            if row['id'] in selected:
                receipts.append(self.vault.revoke(row['id'], remove_ciphertext=False))
                row['status']='revoked'
                row['deleted_at']=timestamp()
        atomic_json(self.metadata, records)
        event=self.event('subset_key_revocation', sample_ids=sorted(selected),
                         receipt_count=len(receipts),
                         limitations='Local key absence only; original model checkpoint remains trained on withdrawn examples.')
        out={'event':event,'receipts':receipts}
        atomic_json(self.root / 'last_deletion_receipts.json', out)
        ledger=self.root / 'revocation_receipts.json'
        prior=json.loads(ledger.read_text()) if ledger.exists() else []
        atomic_json(ledger,prior+receipts)
        return out

    def _split(self):
        rows=self.active('train')
        tests=self.active('test')
        if len(rows) < 8 or len(tests) < 2:
            raise ValueError('Need >=8 active training clips and >=2 active test clips')
        rng=np.random.default_rng(42)
        ids=rng.permutation(len(rows))
        n_val=max(2,round(len(rows)*0.2))
        train=[rows[i] for i in ids[n_val:]]
        val=[rows[i] for i in ids[:n_val]]
        if len(train)<3: raise ValueError('Need >=3 train clips to create babble')
        return train,val,tests

    def train_original(self, epochs=3, batch_size=4, lr=0.001):
        # A training dataset snapshot prevents selection of stale deleted samples.
        if any(r['status']=='revoked' for r in self.records()):
            raise ValueError('Original baseline must be trained before deleting samples. Retrain on retained samples instead.')
        train,val,tests=self._split()
        if (self.root/'original.pt').exists():
            raise ValueError('Original model already exists. Use a new experiment directory for a different baseline.')
        args=self._args(epochs,batch_size,lr)
        history=train_model(train,val,train+val,args,self.root/'original.pt',self.vault)
        atomic_json(self.root/'original_training_ids.json',
                    {'train_ids':[r['id'] for r in train], 'val_ids':[r['id'] for r in val], 'test_ids':[r['id'] for r in tests]})
        self._save_metrics('original', tests, train+val, args)
        self._append_history(history)
        self.event('model_training', model='original', train_count=len(train), model_sha256=sha256(self.root/'original.pt'))
        return history

    def _args(self,epochs,batch_size,lr):
        return SimpleNamespace(seed=42,snr=0.,batch_size=int(batch_size),lr=float(lr),
                               epochs=int(epochs),finetune_epochs=int(epochs),
                               device='cuda' if torch.cuda.is_available() else 'cpu',
                               audio_only=False,skip_intelligibility=False,pesq=False)

    def unlearn(self, epochs=3, batch_size=4, approximate=True):
        if not (self.root/'original.pt').exists(): raise ValueError('Train original model first')
        deleted=[r for r in self.records() if r['status']=='revoked']
        if not deleted: raise ValueError('Revoke at least one training sample first')
        train,val,tests=self._split()
        args=self._args(epochs,batch_size,0.001)
        hist=train_model(train,val,train+val,args,self.root/'retrained_without_deleted.pt',self.vault)
        self._save_metrics('retrained',tests,train+val,args)
        self._append_history(hist)
        result={'retrained':sha256(self.root/'retrained_without_deleted.pt')}
        if approximate:
            hist2=train_model(train,val,train+val,args,self.root/'finetuned_without_deleted.pt',self.vault,
                              mode='finetune', source=self.root/'original.pt')
            self._save_metrics('finetuned',tests,train+val,args)
            self._append_history(hist2)
            result['finetuned']=sha256(self.root/'finetuned_without_deleted.pt')
        self.event('unlearning_experiment', withdrawn_ids=[r['id'] for r in deleted],
                   retained_training_ids=[r['id'] for r in train+val],
                   models_sha256=result,
                   limitations='Retraining excludes withdrawn IDs in this run; no independent proof of forgetting or removal of old checkpoints.')
        return result

    def _save_metrics(self, name, tests, pool, args):
        model_file={'original':'original.pt','retrained':'retrained_without_deleted.pt',
                    'finetuned':'finetuned_without_deleted.pt'}[name]
        metrics=evaluate(self.root/model_file,tests,pool,args,name,self.vault)
        # Serialize only scalar metrics and pseudonymous IDs.
        path=self.root/f'metrics_{name}.csv'
        if metrics:
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)

    def _append_history(self, rows):
        path=self.root/'training_history.csv'
        if rows:
            old=[]
            if path.exists():
                with path.open() as f: old=list(csv.DictReader(f))
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(old+rows)

    def signer(self):
        if not self.signing_key_file.exists():
            k=Ed25519PrivateKey.generate()
            pem=k.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
            fd=os.open(self.signing_key_file,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as f:f.write(pem)
        return serialization.load_pem_private_key(self.signing_key_file.read_bytes(),password=None)

    def issue_certificate(self):
        records=self.records()
        revoked=[r['id'] for r in records if r['status']=='revoked']
        if not revoked: raise ValueError('Nothing revoked yet')
        receipt_file=self.root/'revocation_receipts.json'
        if not receipt_file.exists(): raise ValueError('Missing deletion evidence')
        proofs=json.loads(receipt_file.read_text())
        checked={r['sample_id']:r for r in proofs}
        report={'schema':'AVSE-LOCAL-AUDIT-v1', 'issued_at':timestamp(),
                'scope':'local per-sample encrypted derived cache; model experiment',
                'withdrawn_sample_ids':revoked,
                'last_receipt_status':{i: bool(checked.get(i,{}).get('decrypt_blocked_by_missing_key',False)) for i in revoked},
                'artifacts_sha256':{p.name:sha256(p) for p in sorted(self.root.iterdir())
                                    if p.is_file() and (p.name.endswith('.csv') or p.name.endswith('.pt'))},
                'limitations':('Operator-generated signature: proves integrity/issuer identity only. '
                               'No independent confirmation of key sanitisation, copy completeness, '
                               'past plaintext leakage, or certified machine unlearning. Original checkpoint may retain influence.')}
        key=self.signer()
        pub=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
        envelope={'report':report,'public_key_ed25519_hex':pub.hex(),
                  'signature_ed25519_hex':key.sign(canonical(report)).hex()}
        atomic_json(self.root/'audit_certificate.json',envelope)
        self.event('audit_certificate_issued',report_sha256=hashlib.sha256(canonical(report)).hexdigest())
        return envelope


def verify_certificate(envelope):
    pub=Ed25519PublicKey.from_public_bytes(bytes.fromhex(envelope['public_key_ed25519_hex']))
    pub.verify(bytes.fromhex(envelope['signature_ed25519_hex']),canonical(envelope['report']))
    return True


def image_to_lips(bgr):
    """OpenCV Haar face detector, lower-face mouth-region proxy (not landmark-quality lips)."""
    import cv2
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    boxes=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(60,60))
    if len(boxes):
        x,y,w,h=max(boxes,key=lambda z:z[2]*z[3])
        # mouth typically occupies lower third of detected face; deliberate approximation.
        a=gray[y+int(h*.60):y+int(h*.91), x+int(w*.22):x+int(w*.78)]
        found=True
    else:
        h,w=gray.shape
        a=gray[int(.56*h):int(.85*h),int(.30*w):int(.70*w)]
        found=False
    if a.size==0:raise ValueError('Cannot extract lip ROI')
    return cv2.resize(a,(VWIDTH,VWIDTH)),found


def decode_mp4_bytes(mp4_bytes):
    import cv2,subprocess,tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'recording.mp4';p.write_bytes(mp4_bytes)
        from avse_unlearning import ffmpeg_decode
        # Earlier experiments use whole downscaled video; here actively crop lower face.
        audio,_=ffmpeg_decode(p)
        cap=cv2.VideoCapture(str(p))
        crops=[];i=0
        while len(crops)<NFRAMES and i<250:
            ok,im=cap.read()
            if not ok:break
            if i%3==0:
                crop,_=image_to_lips(im);crops.append(crop)
            i+=1
        cap.release()
        if not crops:raise ValueError('Cannot extract video frames')
        return audio,np.stack(crops)
