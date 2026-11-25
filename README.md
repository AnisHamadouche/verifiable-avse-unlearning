# AVSE Privacy Lab — live capture, encrypted storage, selective deletion, unlearning and certificates

Local **Streamlit research demonstration** building on the LRS3-TED PyTorch AVSE experiment. The app records simultaneous webcam and microphone streams, extracts *approximate* lower-face/mouth crops, creates **per-clip AES-256-GCM encrypted** derived data, trains a small audio-visual speech-enhancement model, withdraws selected clips, retrains without those clips and displays speech metrics and operator-signed audit statements.

**Not production-ready or a medical product.** Neither key unlinking nor a signature proves that every copy has been deleted, or that a model has forgotten a user's information. Audio-video sync uses approximate callback times; face detection uses Haar rectangles, not a precision lip-landmark tracker. Real LRS3 access and licensing are independent of this repository.

## Quick start (Ubuntu 22.04)

Clone your existing repository and add this folder, or extract this package into a separate directory:

```bash
cd ~/lrs3_avse_unlearning
# Copy this package's files into this repository, retaining your existing code as needed.
conda create -n avse-app python=3.11 -y
conda activate avse-app

# For NVIDIA driver 535, use the CUDA 12.1 wheel, not the CUDA 13.2 build.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-app.txt
sudo apt-get install -y ffmpeg

python -m pytest tests/test_backend.py -q
streamlit run app.py --server.address 127.0.0.1
```

Open **http://localhost:8501** on the same machine. Allow camera and microphone access. If your Linux/Firefox WebRTC stream cannot be negotiated, import an MP4 recording instead. Remote webcam access generally needs **HTTPS**, firewall/STUN/TURN configuration and stronger server controls. Never expose this unprotected prototype publicly.

### Optional: existing LRS3 dataset

The dataset is **not bundled**. If already installed at `~/LRS3-TED`, select the **Capture & import → Import LRS3 subset** expander, set that path and choose any number of MP4s, or enter **0** for the complete train/test split (no fixed upper limit). Imports process one sample at a time, show progress, and skip already imported path-tracked samples on subsequent runs. Large datasets require substantial disk space, and the app currently performs synchronous imports. Your local directory should contain `trainval/**/*.mp4` and `test/**/*.mp4`. The app reads the first 2 seconds of each clip using FFmpeg and OpenCV. It keeps the original MP4 files unchanged. To obtain the dataset, follow the official LRS3 access/licensing instructions; do not commit its media to GitHub.

### Running the interactive workflow

1. On **Capture & import**, press START and grant webcam/microphone access. Record speech for at least 2 seconds, press STOP, then **Save encrypted stream clip**. You can also upload MP4s or import local LRS3 samples. Clips are marked `train` or `test`.
2. Record/import **at least 8 train clips and 2 test clips**. For a quick functional demonstration, add **12 synthetic train + 3 synthetic test** examples, which are not real speech.
3. Go to **Train & unlearn** and train the original AVSE model. Start with 1–3 epochs for a demo. Use your supported CUDA-enabled PyTorch install for GPU execution.
4. In **Records & deletion**, select **one or several training clips**, but not all of them, and request selective deletion. Each selected key is removed from the demo key store and the selected clip becomes inaccessible through the active vault. **Ciphertext and original source files may remain.**
5. In **Train & unlearn**, run retraining from scratch using active records; optionally compare fine-tuning from the original checkpoint. Full retraining is a reference procedure; approximate fine-tuning provides **no forgetting guarantee**.
6. **Metrics** shows per-model SI-SNR improvement, STOI and ESTOI on held-out test examples, along with validation loss. The same model architecture uses a visual CNN fed mouth crops. STOI/ESTOI are intelligibility metrics, **not privacy metrics**.
7. In **Certificates**, inspect local revocation receipts, issue a signed JSON audit statement and download it. Verify its signature independently with:

```bash
python verify_certificate.py app_data/demo/audit_certificate.json
```

## Privacy and scope of evidence

- **Data at rest**: each derived clip has an independent random AES-256-GCM key, stored separately from its encrypted `.enc` file. Ciphertext has authenticated encryption.
- **Partial withdrawal**: selected **train** samples lose their active local keys; untouched samples remain accessible. Revocation removes key files via normal filesystem unlinking. This is **not assured secure sanitisation** of backups, snapshots, filesystem blocks, copied plaintext or external sources.
- **ML impact**: the *original* model checkpoint can retain learned information. Retraining trains fresh weights without revoked samples, assuming the dataset iterator and data lineage are complete. Fine-tuning is not guaranteed to forget. No membership-inference audit, certified unlearning, cryptographic proof of training or remote attestation is implemented.
- **Certificates**: local Ed25519 signed statements can be checked for integrity and issuer identity. The signing key belongs to the operator running this demonstration, so it **does not make an event independently verifiable**. An external witnessing service, trusted KMS and proof of executed unlearning would be required for a stronger claim.
- **Recorded participants**: use consented recordings only. Webcam frames and microphone audio are transmitted from the browser to the local Streamlit process. The interface is **single-user**, lacks authentication, and is **not suitable for clinical patient data**.
- **Reset**: for a new experiment use a **new experiment directory and key directory**. Do not delete keys or data without knowing the consequences.

## GitHub CI

`.github/workflows/ci.yml` runs Python syntax and non-media encryption/deletion tests on a GitHub-hosted CPU runner. **Do not commit** private key directories, model weights, recorded media or `app_data/`. For real GPU training run the Streamlit app locally or add a carefully isolated private GPU runner.

## Project contents

```text
app.py                    Streamlit UI including live WebRTC capture
privacy_backend.py        Vault, partial deletion, training, metrics, signatures
avse_unlearning.py        Existing tiny AVSE model and evaluation pipeline
secure_cache.py           Per-sample AES-256-GCM encrypted store
speech_metrics.py         SI-SNRi, STOI, ESTOI, optional PESQ calculation
verify_certificate.py     Independent *signature* verification CLI
requirements-app.txt      App dependencies (install torch separately)
tests/test_backend.py     Partial-deletion and signature tamper tests
.github/workflows/ci.yml GitHub Actions checks
```

## Limitations for a research paper

Report retained-data utility (STOI, ESTOI, SI-SNRi), verified **local** key absence, and number of selected samples. Clearly state that this is **an exploratory application**, not a verifiable unlearning scheme. For journal-quality results, add speaker-disjoint splits, audio/video sync validation, mouth landmark tracking (e.g., MediaPipe Face Mesh), membership-inference attacks, repeated trials and confidence intervals, persistent verified data lineage, device-held keys, an independently witnessed append-only audit log and cryptographically verifiable model-update proofs.

## Playback & Preview tab

After recording, importing an MP4, or generating synthetic clips, open **▶️ Playback & preview**. Select an **active** record to listen to its decrypted 16 kHz audio and watch the synchronised **48×48 lower-face crop sequence**. The video player muxes the audio and lip frames into a temporary in-memory MP4 using FFmpeg. An audio-only player and selectable individual lip-frame gallery are also provided. Full-face frames were not stored and cannot be played back.

The app never caches decrypted samples in Streamlit session state or intentionally writes plaintext preview files to persistent disk. The 2-second preview necessarily exists temporarily in RAM and the user's browser. Keep the app on `localhost`, use it only with recordings you have permission to process, and never distribute private previews. **Revoked records are excluded**: removing their per-sample key blocks preview through the active vault. This does not prove erasure from other caches, backups, browsers, or earlier downloaded copies.

Requires `ffmpeg` for combined audio/video playback (`sudo apt install ffmpeg`).


## Six-page Streamlit workflow (Next / Back)

Run `streamlit run app.py`. Use **Next →**, **← Back**, or the sidebar page selector. The screens are Capture & Import, Playback & Preview, Train, Records & Deletion, Metrics, and Certificates. Train the **original** model before revoking training samples.

Playback displays three independent audio players: **original captured audio**, **noisy audio** (deterministically generated two-speaker babble at 0 dB using currently available training records), and **enhanced audio** from a selectable saved checkpoint. A checkpoint is required for the last player. Since noise pools change after withdrawal, this interactive example may not exactly reproduce historical test-set metrics. Revoked samples are not playable. The synchronized lip-region MP4 contains original audio.

Metrics include SI-SNR, SI-SNR improvement, waveform SNR and SNR improvement, segmental SNR, RMSE, MAE, STOI/ESTOI input/output and improvements, and optional PESQ-WB if actually computed. New metrics are recorded by new training runs; older CSV files will only display metrics that exist in those files. PESQ requires an optional `pesq` install and enabling it in backend evaluation. This prototype does not claim any privacy/unlearning guarantee.


### Selecting evaluation metrics

On **5 · Metrics**, tick the measures to show under the grouped checkboxes:
intelligibility (STOI/ESTOI), quality (PESQ-WB, if computed), enhancement
(SI-SNR/SNR/segmental SNR), and reconstruction error (RMSE/MAE).
**Select all available metrics** and **Clear metric selection** are provided.
The selections filter the model summary, chart choices, individual-clip table,
and downloadable CSV reports. Unavailable metrics are not offered.

## P1–P5 research extensions (November 2026)

The project now includes dependency lineage and deletion-impact reporting (P1); retraining/fine-tuning plus a **retained-vs-test** loss-membership proxy (P2); signed reports with independently pinnable Ed25519 verification keys and hash-chained events (P3); optional MediaPipe mouth landmarks and audio/visual cross-attention (P4); and extended synthetic GitHub CI with a source-only attested release (P5).

The UI includes an additional **7 · Assurance & provenance** page. The original model is marked retired after withdrawal and cannot be selected for audio playback. Run `python -m pytest tests -q` to exercise synthetic integration tests. For scope, exact verification commands, limitations, and CI/CD, see [`docs/P1_P5_IMPLEMENTATION.md`](docs/P1_P5_IMPLEMENTATION.md).

**Important:** These features do not turn the app into a production or clinical service. The local signing operator remains able to omit events; neither missing DEKs nor empirical membership proxy scores certify universal deletion or unlearning.


## Automated privacy audit (P1–P5 + CI gate)

The UI now has **8 · Automated privacy audit** as a separate Next/Back page. It displays
pass/fail checks and exports `privacy_audit_summary.json` from the *local* experiment.
The audit reads the experiment's registered records, encrypted sample keys and ciphertext,
withdrawal receipts, model-serving policy, dependency lineage and event chain. It tests
that active samples decrypt and revoked samples cannot be accessed through the active vault.

From the repository root:

```bash
python -m pytest tests -q
python scripts/generate_audit_fixture.py /tmp/avse-fixture /tmp/avse-keys
python scripts/privacy_audit.py --root /tmp/avse-fixture --key-root /tmp/avse-keys \
  --output /tmp/avse-privacy-audit.json
```

To audit a *real local research experiment* (never put it on public GitHub):

```bash
python scripts/privacy_audit.py --root app_data/demo \
  --key-root "$HOME/.local/share/avse-streamlit-keys" \
  --output /tmp/avse-local-audit.json
```

To verify the certificate independently, first pin its public key and a witnessed event
chain root *outside the application operator's control*, then provide:

```bash
python scripts/privacy_audit.py --root app_data/demo \
  --key-root "$HOME/.local/share/avse-streamlit-keys" \
  --require-certificate --pinned-key "$HOME/trusted-avse/key.hex" \
  --witnessed-root YOUR_PREVIOUSLY_WITNESSED_64_HEX_ROOT \
  --output /tmp/avse-verified-audit.json
```

The GitHub Actions workflow `.github/workflows/privacy-audit.yml` runs only synthetic
records for pull requests, `main`, weekly scheduled audits and manual runs. The
source-release workflow now executes the same regression suite and synthetic deletion
audit before creating an attested archive. Security checks fail the job on findings rather
than ignoring them. Configure a **repository ruleset** requiring the `synthetic-privacy-audit`
job and `privacy-ci` before merging into `main`.

The prototype **does not** independently establish complete erasure of unknown copies,
backups, clipboard/browser caches, or learned model information. Application certificates
are not medical/regulatory compliance certificates; local event chains can be rewritten
without an external witness. Keep clinical and volunteer data off GitHub.

## Publication plots, tables and verification logs (CLI)

To generate IEEE-style **title-free** PDF/EPS/PNG figures, CSV/LaTeX tables, a reporting log and SHA-256 manifest without opening Streamlit:

```bash
python -m pip install numpy pandas matplotlib
python scripts/ieee_paper_report.py --run-dir app_data/demo \
    --output paper_outputs/demo --bootstrap 2000 --dpi 600
```

For the earlier LRS3 command-line experiment, set `--run-dir runs/avse_experiment_01` instead. Full instructions, supported metrics and scientific limitations are in [`docs/IEEE_PAPER_REPORTING.md`](docs/IEEE_PAPER_REPORTING.md). The GitHub workflow `.github/workflows/report-smoke.yml` tests reporting on synthetic measurements without uploading volunteer data.
