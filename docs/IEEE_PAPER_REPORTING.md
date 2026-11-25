# Reproducible IEEE-style publication reporting

The standalone CLI reads **existing experiment outputs** without starting Streamlit, accessing AVSE media, decrypting recordings or loading any key material. It works with both Streamlit `metrics_original.csv`, `metrics_retrained.csv`, `metrics_finetuned.csv` and the original CLI `per_clip_metrics.csv`.

## Setup

```bash
conda activate avse-app
python -m pip install numpy pandas matplotlib
```

## From the Streamlit experiment

```bash
python scripts/ieee_paper_report.py \
  --run-dir app_data/demo \
  --output paper_outputs/demo \
  --bootstrap 2000 --seed 42 --dpi 600
```

## From the earlier LRS3 training CLI

```bash
python scripts/ieee_paper_report.py \
  --run-dir runs/avse_experiment_01 \
  --output paper_outputs/lrs3_run_01 \
  --bootstrap 2000 --seed 42 --dpi 600
```

Use the actual run directory, which must contain at least some experiment CSV or audit outputs. Missing metrics are **omitted** (not set to zero). The reporter never computes WER, PESQ, privacy attack scores, inference latency, formal deletion proof costs or other metrics if they were not measured during training/evaluation.

## Output files

```text
paper_outputs/demo/
├── figures/
│   ├── fig_training_curves.pdf / .eps / .png     (if training history exists)
│   ├── fig_si_snri_comparison.*                  (if measured)
│   ├── fig_output_stoi_comparison.*
│   ├── fig_output_estoi_comparison.*
│   ├── fig_si_snri_ecdf.*
│   ├── fig_local_revocation_checks.*             (if receipts exist)
│   └── fig_membership_proxy_auc.*               (if measured)
├── tables/
│   ├── table_speech_summary_long.csv / .tex
│   ├── table_speech_performance.csv / .tex
│   ├── table_paired_model_differences.csv / .tex
│   ├── table_local_key_revocation.csv / .tex
│   ├── table_membership_proxy.csv / .tex         (if measured)
│   └── table_privacy_audit_checks.csv / .tex     (if available)
├── logs/report.log
└── evidence_manifest.json
```

All figure outputs **have no titles or captions**; use `\caption{...}` in your IEEE LaTeX manuscript. Figure sizes are approximately IEEE one-column width (3.5 inches), serif fonts, monochrome-friendly line styles, and vector output PDF/EPS; raster PNG defaults to 600 dpi. LaTeX uses `booktabs`: include `\usepackage{booktabs}` in the IEEE preamble.

CSV contains numerical results, while LaTeX tables round values to three decimal places. The bootstrapped 95% confidence intervals are **per-clip resampling**, not speaker-level generalisation intervals; for a strong paper use speaker-grouped resampling and repeated independent training seeds. A paired comparison is only made when sample IDs match across models; output data are summary tables without sample IDs.

## Important verification limitations

- `fig_local_revocation_checks` only summarises flags from **local receipts**. It cannot prove external backup/key destruction or that the operator was truthful.
- `fig_membership_proxy_auc` is **retained-versus-held-out** loss discrimination, not a deleted-sample membership-inference test and not a certified forgetting guarantee.
- `evidence_manifest.json` lists SHA-256 checksums of the generated report files; it is **unsigned** and does not independently establish source provenance.
- Reporting does not upload private media, keys, sample IDs, original video or volunteer recordings. Audit log details may still be sensitive: review tables before publishing.

## Local and GitHub CI tests

```bash
python -m pytest tests/test_ieee_paper_report.py -q
```

`report-smoke.yml` executes the test using synthetic metrics in GitHub Actions. It deliberately does not download LRS3 or publish actual experiment outputs.
