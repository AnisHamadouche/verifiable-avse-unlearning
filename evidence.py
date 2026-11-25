"""P3: hash-chained events and externally pinned certificate verification.
Chain detects mutation of observed history; only externally retained checkpoints defend truncation.
"""
import hashlib, json, os, secrets
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

def canonical(data): return json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(data): return hashlib.sha256(canonical(data)).hexdigest()
def chain(events):
    prev='0'*64;entries=[]
    for seq, ev in enumerate(events):
        payload={'seq':seq,'prev':prev,'event':ev}
        h=digest(payload);entries.append({**payload,'hash':h});prev=h
    return entries

def verify_chain(entries):
    prev='0'*64
    for i, item in enumerate(entries):
        expected={'seq':i,'prev':prev,'event':item['event']}
        if item.get('seq')!=i or item.get('prev')!=prev or item.get('hash')!=digest(expected):
            raise ValueError(f'Event chain corrupted at position {i}')
        prev=item['hash']
    return prev

def write_chain(root, events):
    root=Path(root);entries=chain(events)
    tmp=root/('event_chain.json.'+secrets.token_hex(4)+'.tmp')
    tmp.write_text(json.dumps(entries,indent=2));os.replace(tmp,root/'event_chain.json')
    return entries[-1]['hash'] if entries else '0'*64

def verify_certificate_pinned(envelope, pinned_public_key_hex, expected_chain_root=None):
    if envelope['public_key_ed25519_hex']!=pinned_public_key_hex:
        raise ValueError('Signer does not match external pinned public key')
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(pinned_public_key_hex)).verify(
        bytes.fromhex(envelope['signature_ed25519_hex']),canonical(envelope['report']))
    if expected_chain_root is not None and envelope['report'].get('event_chain_root')!=expected_chain_root:
        raise ValueError('Certificate chain root is not externally witnessed root')
    return True

def export_public_key(root, public_key_raw):
    p=Path(root)/'verifier_public_key.hex';p.write_text(public_key_raw.hex()+'\n');return p
