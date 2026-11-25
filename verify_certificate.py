import json,sys
from privacy_backend import verify_certificate
if len(sys.argv)!=2:raise SystemExit('Usage: python verify_certificate.py path/to/audit_certificate.json')
cert=json.load(open(sys.argv[1]))
assert verify_certificate(cert)
print('Signature VALID. Does not prove real-world deletion or unlearning.')
