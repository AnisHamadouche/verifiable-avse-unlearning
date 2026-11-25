# P1–P5 Implementation and Reproducibility

**Scope:** Local, single-operator, consent-based research prototype only. Authentication was intentionally excluded by project request. Never expose the Streamlit service publicly or ingest patient records into this prototype.

## P1: Data lineage and withdrawal propagation

- `lineage.py` registers encrypted derived samples, model checkpoints and metrics reports as nodes, with explicit dependency edges.
- Every original-model dependency includes all audio/video training samples and the babble-speech pool. A deleted sample can therefore affect other training mixtures.
- `withdrawal_impact.json` identifies *registered* downstream artefacts affected by a deletion request, including the original checkpoint.
- `model_serving_policy.json` retires original checkpoints after any withdrawal. The playback page excludes the original checkpoint when samples have been revoked. Historical checkpoints still physically exist for research comparison.
- Not captured: LRS3 source file copies, RAM, browser downloads, backups, or any artefact not registered in the graph. **Do not treat the report as proof of complete deletion.**

## P2: Unlearning and empirical privacy evaluation

- `privacy_backend.py`: exact fresh training procedure on retained clips (retraining baseline); optional fine-tuning from original contaminated model (approximation only).
- `unlearning_audit.py`: per-sample L1 losses for retained vs held-out test clips under *each* checkpoint; resulting loss-threshold membership-inference proxy AUC, train/test loss, latency, counts in `unlearning_privacy_metrics.csv`.
- Uses active training samples only to construct noise after withdrawal. Original checkpoint contains pre-withdrawal influence and is retired for playback, not securely destroyed.
- **Not implemented:** deleted-sample MIA once keys are revoked, exact distributional retraining equivalence, certified removal or proof that a model ceased learning from withdrawn examples. The attack AUC should not be advertised as such.

## P3: Verification and trust boundaries

- `evidence.py` creates SHA-256 chained event entries in `event_chain.json`, and checks ordering, parent pointers and hashes.
- `privacy_backend.py` signs an audit certificate with Ed25519, including the event-chain root.
- `verify_independent.py` verifies the signature against a **separately pinned public key**; optional `--witnessed-root` requires a previously saved checkpoint outside operator control. Export/pin the public key before accepting a certificate; otherwise the operator can substitute a new key.
- Hash chains do **not** prove omitted events, and certificates do **not** certify key wiping or machine forgetting. No third-party witnessing infrastructure or trusted KMS/HSM is shipped.

Example:

```bash
# Once, independently obtain and retain the verification key in a trusted location:
mkdir -p ~/trusted-avse
cp app_data/demo/verifier_public_key.hex ~/trusted-avse/pinned_public_key.hex
# At issuance, keep the current hash root separately as an independently witnessed checkpoint.
python verify_independent.py \
  --certificate app_data/demo/audit_certificate.json \
  --pinned-public-key ~/trusted-avse/pinned_public_key.hex \
  --chain app_data/demo/event_chain.json
```

**Threat-model warning:** If the operator also controls the `~/trusted-avse` directory, the public key is *not* independent. Use a physically separate device or independently trusted party for actual external pinning. Reissuing a certificate after future events creates a new chain root.

## P4: AVSE quality and architecture

- `visual_features.py`: optional MediaPipe FaceMesh lip-landmark crop with OpenCV Haar lower-face fallback. Install compatible `mediapipe` separately and confirm it has `solutions.face_mesh` available.
- `avse_unlearning.py`: original `TinyAVSE` baseline and new `CrossAttentionAVSE` with audio queries attending to visual feature tokens; both use a shared inference path and the checkpoint weights identify the architecture.
- Streamlit capture displays a basic capture-quality report (face detection fraction, audio/video frame count and estimated span). Sync remains approximate; timestamps are not hard-synchronised.
- The selectable architecture is a research alternative, **not an established SOTA implementation**. Requires LRS3 training/evaluation on real recordings to establish performance.

## P5: CI/CD and evidence integrity

- `.github/workflows/ci.yml`: Python 3.11 CPU PyTorch, syntax checks, package tests, synthetic training→withdrawal→retraining→metrics integration test, certificate verification against pinned key, and tracked-file leakage check.
- `.github/workflows/release.yml`: release of **source-only** Git archive with GitHub build provenance attestation. A signed software build is not a signed deletion certificate.
- `scripts/check_tracked_files.py`: blocks common sensitive binary formats and local data paths from tracked Git files. This does not replace secret scanning or a rigorous code review.
- Security scanners run as informational checks pending version-pinned requirements and baseline triage. No continuous deployment to a patient-data environment is provided.

## Running

```bash
conda create -n avse-app python=3.11 -y
conda activate avse-app
# For NVIDIA 535 driver choose the CUDA 12.1 wheel; CPU users install CPU PyTorch.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-app.txt
sudo apt-get install -y ffmpeg
OMP_NUM_THREADS=2 python -m pytest tests -q
streamlit run app.py --server.address 127.0.0.1
```

Use **Capture & Import → Train & Unlearn (baseline) → Records & Deletion → Train & Unlearn (after withdrawal) → Metrics → Certificates → Assurance & Provenance**. A 10% selective deletion is illustrative; preserve some training clips.

## Source limitations that remain before real production

Critical gaps remain: authentication/authorisation (deliberately omitted), consent-management backend, KMS and key backups, independent log witness, all-copy discovery/deletion, old-checkpoint sanitisation, certified-unlearning proof, repeated user-withdrawal guarantees, measured adversarial deleted-sample attacks, robust timestamp alignment, scalable transactional metadata storage, deployment hardening, formal clinical governance and penetration testing. **Do not use for real health data in production.**
