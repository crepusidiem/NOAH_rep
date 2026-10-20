# Architecture audit and CPU verification

- Date/timezone: 2026-10-19, America/New_York.
- Stage: local PDF/source audit, implementation fixes, CPU regression and synthetic training smoke test.
- Repository: `/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/NOAH_rep` (shell resolves to `/home/wey371/rajpurkarlab/home/wey371/NOAH_rep`).
- Source baseline: `1d21fd6f85d4fabbf6cdf0022e865db82cc52dc7`; dirty tree. Pre-existing changes in `scripts/train_pilot.py`, `scripts/evaluate_clinical_probes.py` and untracked reconciliation/probing/ablation scripts were preserved. Audit edits remain uncommitted.
- Interpreter: `/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python`; PyTorch `2.14.1+cu130`, inspected directly. Muon available.
- Slurm jobs / allocated GPU / queue: NOT_RUN / no GPU requested / NOT_MEASURED. Commands ran on CPU with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
- Dataset scope: no clinical tables/media decoded for this audit. Supplied ECHO1.0.1 directory existence verified; video count, readable-video rate and encoding NOT_MEASURED. No clinical content copied into reports.
- Paper: local50-page `paper/NOAH.pdf`; SHA-256 `0f3a7bb049d6f7eb6cdf61c2aa9692d0695037851f95b73711a26ed803576730`; `pdftotext -layout` used to inspect Methods and Table16.

## Results

The existing specification and traceability matrix describe the architectural target and substantially match the PDF. The new [architecture audit](../../../docs/ARCHITECTURE_AUDIT.md) maps requirements to current source, tests, selected assumptions and remaining gaps.

Source fixes: linear specifics projection; input temporal encoding; latent-width cascade; residual drop path; ECHO reconstruction loss; valid shifted mask intersection; category/modality/specifics-presence weights; Muon/AdamW role partition; separate update/epoch schedule clocks; accumulation tail scaling; actual text-content targets; strict unresolved-target/CSV handling; checkpoint-strict evaluation; full-history, support-masked prior-only rollout with latent temperature and outgoing-gap semantics. Default free rollout preserves simultaneous events; forced time and per-head top-p sampling remain explicit deviations.

ECG extraction previously instantiated OTIS without loading pretrained weights. Current source requires a compatible checkpoint, strict state loading, frozen parameters, expected lead/rate/length checks, finite192-D output and cache identity covering checkpoint, code, recipe and source waveform/header bytes. Legacy unverified caches are not reused. Compatible pretrained weights were not obtained or validated here; this gate remains OPEN.

Candidate full-dimension parameter inventory:148,957,844 versus paper150,322,970. This discrepancy remains explicit; the count test does not certify equality. New architecture changes are incompatible with silently partial loading of old checkpoints.

## Regression command

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -m pytest tests/test_architecture_audit.py tests/test_temporal_oracle.py tests/test_variational_oracle.py tests/test_window_builder.py tests/test_noah_model.py tests/test_rollout.py tests/test_transformer_scaling.py tests/test_paper_scale_model.py tests/test_dynamic_demographics.py tests/test_clinical_probes.py -q
```

Result:39 passed,1 skipped. The skip is the existing real-curated-batch fixture requiring absent `data/processed/pilot_train_windows.pt`. Coverage includes independent temporal/KL references, ownership, full-dimension model forward/backward, L2048 candidate attention, ECHO loss/gradients, future-content invariance, outgoing rollout timing/support/temperature, optimizer role/state checks and legacy-ECG-cache rejection. Full CXR encoder tests were not run: they require real media and encoder inference. No new scientific performance metrics are claimed.

## Synthetic trainer smoke test

A temporary synthetic fixture under `/tmp/noah_architecture_smoke/` contains one70-event journey, three40-token windows, overlap8 and six prefixes. Category LAB_TEST/numeric values are fictional. Training and validation intentionally use the same synthetic fixture to exercise code paths; this is not generalization evidence.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python scripts/train_pilot.py --train_data /tmp/noah_architecture_smoke/windows.pt --val_data /tmp/noah_architecture_smoke/windows.pt --output_dir /tmp/noah_architecture_smoke/checkpoints --device cpu --epochs 1 --batch_size 2 --d_model 32 --d_latent 16 --n_layers 1 --n_heads 2 --ffn_ratio 2 --config configs/pilot_config.yaml --wandb_mode disabled
```

Exit0. Muon/AdamW training, validation, constrained prior rollout and checkpoint save completed. Three windows, two microbatches, accumulation2; optimizer clock advances once. Model179,108 parameters (head width64 inherited explicitly from pilot YAML). Artifacts remain outside Git. Measured training-loop wall time approximately0.07s, excluding process/import/setup; no GPU throughput inference is warranted. Checkpoint random states/optimizer partition/support/weights were saved. Exact CLI resume and data-order replay remain NOT_RUN.

## Limitations and next action

Corrected real-data extraction/training is NOT_RUN. ICU charted-vital aggregation, ECHO DICOM/video path, text/CXR content/recipe cache provenance, complete full-record inference/evaluation and exact resume remain OPEN. Existing clinical metrics reflect previous implementations and protocols. Verify assets/source coverage, regenerate affected artifacts, then run corrected small real-data integration before expanding cohort or claiming paper fidelity. See [project status](../../../docs/PROJECT_STATUS.md).
