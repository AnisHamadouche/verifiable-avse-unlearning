"""Independent verifier, requiring previously pinned public key and optionally chain root."""
import argparse,json
from pathlib import Path
from evidence import verify_certificate_pinned,verify_chain

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--certificate',required=True)
    p.add_argument('--pinned-public-key',required=True,help='Path to externally retained .hex key')
    p.add_argument('--chain',help='Event-chain JSON copied separately')
    p.add_argument('--witnessed-root',help='Externally retained SHA256 chain root (not a file in untrusted project directory)')
    a=p.parse_args()
    cert=json.loads(Path(a.certificate).read_text())
    key=Path(a.pinned_public_key).read_text().strip()
    verify_certificate_pinned(cert,key,a.witnessed_root)
    if a.chain:
        root=verify_chain(json.loads(Path(a.chain).read_text()))
        if root!=cert['report'].get('event_chain_root'):raise ValueError('Chain and signed report disagree')
        if a.witnessed_root and root!=a.witnessed_root:raise ValueError('Chain and external witness disagree')
    print('PASS: authentic certificate under pinned key; supplied chain consistent.')
    print('NOT PROVEN: missing events, external copies, key sanitisation or certified model forgetting.')
if __name__=='__main__':main()
