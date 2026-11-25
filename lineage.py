"""P1: explicit dependency graph for samples, models, and derived reports.
Local research scope: registered artifacts only; not an inventory of external copies.
"""
import hashlib, json, os, secrets
from pathlib import Path
from datetime import datetime, timezone

def utc(): return datetime.now(timezone.utc).isoformat()
def atomic(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.'+secrets.token_hex(4)+'.tmp')
    t.write_text(json.dumps(payload,indent=2,sort_keys=True));os.replace(t,path)

def register(root, artifact, kind, sources=(), sha256=None, **metadata):
    p=Path(root)/'lineage.json'; graph=json.loads(p.read_text()) if p.exists() else {'schema':'avse-lineage-v1','nodes':{}}
    graph['nodes'][artifact]={'kind':kind,'sources':sorted(set(sources)),'sha256':sha256,
                              'registered_utc':utc(),**metadata}
    atomic(p,graph)
    return graph['nodes'][artifact]

def impact(root, revoked):
    p=Path(root)/'lineage.json';graph=json.loads(p.read_text()) if p.exists() else {'nodes':{}}
    impacted=set(revoked)
    changed=True
    while changed:
        changed=False
        for name, node in graph['nodes'].items():
            if name not in impacted and any(s in impacted for s in node['sources']):
                impacted.add(name);changed=True
    return {k:graph['nodes'][k] for k in sorted(impacted) if k in graph['nodes']}

def report(root, revoked):
    root=Path(root)
    affected=impact(root,revoked)
    inventory={'schema':'avse-withdrawal-impact-v1','withdrawn_ids':sorted(set(revoked)),
               'affected_registered_artifacts':affected,
               'untracked_external_copies':'UNKNOWN',
               'limitations':'Registered local dependencies only. Original source files, plaintext exports, copies outside this registry and former model checkpoints may survive.'}
    atomic(root/'withdrawal_impact.json',inventory)
    return inventory
