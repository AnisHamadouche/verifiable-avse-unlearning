#!/usr/bin/env python3
"""Fail-closed, local, read-only audit of AVSE prototype state.
No data or keys are uploaded. Checks are bounded to the active local vault.
"""
import argparse, json, hashlib, sys
from datetime import datetime, timezone
from pathlib import Path
# Allow direct execution from repository root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_json(p, mandatory=False):
    if not p.exists():
        if mandatory: raise ValueError(f'Missing {p.name}')
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def audit(root, key_root, require_certificate=False, pinned_key=None, witnessed_root=None):
    from evidence import verify_chain, verify_certificate_pinned
    from secure_cache import EncryptedCache
    root, key_root = Path(root).resolve(), Path(key_root).resolve()
    checks=[]
    vault=EncryptedCache(root,key_root / "samples")
    def check(name, ok, note=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': note})
    records = read_json(root/'records.json', True)
    ids=[r['id'] for r in records]
    check('unique_record_ids',len(ids)==len(set(ids)))
    active=[r for r in records if r.get('status')=='active']
    revoked=[r for r in records if r.get('status')=='revoked']
    check('active_records_exist',len(active)>0)
    receipt_list=read_json(root/'revocation_receipts.json') or []
    receipts={r['sample_id']: r for r in receipt_list}
    for row in revoked:
        sid=row['id']; key=key_root/'samples'/f'{sid}.key'
        ct=root/'encrypted_cache'/f'{sid}.enc'; receipt=receipts.get(sid,{})
        check('revoked_key_absent',not key.exists(), sid)
        try:
            vault.load(sid)
        except FileNotFoundError:
            check('revoked_decryption_blocked',True,sid)
        except Exception as exc:
            check('revoked_decryption_blocked',False,f'{sid}: unexpected {type(exc).__name__}')
        else:
            check('revoked_decryption_blocked',False,sid)
        check('revocation_receipt_present',bool(receipt),sid)
        if receipt:
            check('revocation_receipt_claims_blocked',receipt.get('decrypt_blocked_by_missing_key') is True,sid)
            check('receipt_ciphertext_hash_matches',ct.exists() and hashlib.sha256(ct.read_bytes()).hexdigest()==receipt.get('ciphertext_sha256'),sid)
    for row in active:
        sid=row['id']; key=key_root/'samples'/f'{sid}.key'
        ct=root/'encrypted_cache'/f'{sid}.enc'
        check('active_key_ciphertext_present',key.is_file() and ct.is_file(),sid)
        if key.is_file() and ct.is_file():
            try:
                audio, frames = vault.load(sid)
                check('active_record_decrypts',bool(audio.size and frames.size),sid)
            except Exception as exc:
                check('active_record_decrypts',False,f'{sid}: {type(exc).__name__}')
    if revoked:
        policy=read_json(root/'model_serving_policy.json',True)
        check('original_model_retired',policy.get('original.pt')=='retired_after_withdrawal')
        impact=read_json(root/'withdrawal_impact.json',True)
        check('impact_includes_withdrawn_ids',set(r['id'] for r in revoked).issubset(set(impact.get('withdrawn_ids',[]))))
        lineage=read_json(root/'lineage.json',True)
        nodes=lineage.get('nodes',{})
        if (root/'original.pt').exists():
            check('original_model_in_lineage','original.pt' in nodes)
            check('original_model_depends_on_withdrawn',bool(set(nodes.get('original.pt',{}).get('sources',[])) & set(r['id'] for r in revoked)))
        for model in ['retrained_without_deleted.pt']:
            if (root/model).exists():
                check('retrained_model_registered',model in nodes)
                check('retrained_model_excludes_withdrawn',not (set(nodes.get(model,{}).get('sources',[])) & set(r['id'] for r in revoked)))
    chain=read_json(root/'event_chain.json')
    root_hash=None
    if chain is not None:
        try: root_hash=verify_chain(chain); check('chain_integrity',True)
        except Exception as exc: check('chain_integrity',False,str(exc))
    elif revoked: check('chain_available',False,'Issue a certificate to create event_chain.json')
    if require_certificate:
        envelope=read_json(root/'audit_certificate.json',True)
        check('pinned_key_configured',bool(pinned_key))
        if pinned_key:
            try:
                verify_certificate_pinned(envelope,Path(pinned_key).read_text().strip(),witnessed_root)
                check('certificate_signature_and_pinning',True)
            except Exception as exc: check('certificate_signature_and_pinning',False,str(exc))
        check('certificate_matches_supplied_chain',root_hash is not None and envelope.get('report',{}).get('event_chain_root')==root_hash)
        check('externally_witnessed_root_configured',bool(witnessed_root))
    summary={'schema':'avse-privacy-audit-v1','created_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'local registered derived data, active key store, registered model serving states and supplied evidence',
             'active_records':len(active),'revoked_records':len(revoked),'passed':all(c['pass'] for c in checks),
             'checks':checks,'limitations':['No proof of missing external copies or backup key destruction.',
                 'No formal model-forgetting guarantee; use privacy attacks and retraining comparisons separately.',
                 'Locally signed certificates do not prove completeness without externally witnessed checkpoints.']}
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--key-root',required=True)
    p.add_argument('--output',default='privacy_audit_summary.json')
    p.add_argument('--require-certificate',action='store_true')
    p.add_argument('--pinned-key')
    p.add_argument('--witnessed-root')
    a=p.parse_args()
    try:
        result=audit(a.root,a.key_root,a.require_certificate,a.pinned_key,a.witnessed_root)
    except Exception as exc:
        result={'passed':False,'checks':[{'check':'audit_execution','pass':False,'detail':str(exc)}]}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for c in result['checks']:
        print(('PASS' if c['pass'] else 'FAIL'),c['check'],c.get('detail',''))
    print('RESULT:', 'PASS' if result['passed'] else 'FAIL')
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
