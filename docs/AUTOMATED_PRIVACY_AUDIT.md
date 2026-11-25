# Automated healthcare privacy checks

## CI workflow
- `ci.yml`: compile, tests, local security scans, source guard, verifier smoke check.
- `privacy-audit.yml`: weekly, PR and push synthetic deletion regression checks.
- `release.yml`: blocking tests and synthetic audit prior to source packaging.

## Run locally
From the project root, execute:

```bash
python -m pytest tests -q
python scripts/generate_audit_fixture.py /tmp/avse-fixture /tmp/avse-keys
python scripts/privacy_audit.py --root /tmp/avse-fixture --key-root /tmp/avse-keys --output /tmp/avse-audit.json
streamlit run app.py --server.address 127.0.0.1
```

Select **8 · Automated privacy audit** to audit an existing experiment. Certificate
verification can be enabled using an independently pinned Ed25519 public key plus a
previously witnessed chain root. Enter those verifier-controlled values on that page.

## What passes mean
- Local revocation checks validate that missing active DEKs prevent access through
  this specific encrypted cache and intact ciphertext receipts are present.
- Active checks confirm a non-revoked ciphertext can be decrypted using its key.
- Model checks verify declared retirement and registered training-data lineage.
- Evidence checks validate the internally consistent hash chain and, optionally,
  the signature under a pinned key and an externally witnessed root.

None of these provide cryptographic proof of global erasure, log completeness,
medical regulatory compliance, or formal machine unlearning. For real deployments,
use user-controlled KMS/HSM, externally witnessed transparency checkpoints,
production security reviews, and validated machine-unlearning evaluation.
