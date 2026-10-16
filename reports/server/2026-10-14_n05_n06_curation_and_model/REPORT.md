# Server report: 2026-10-14_n05_n06_curation_and_model

- Checked at / timezone: 2026-10-14 14:28:00 EDT
- Stage: ETL / Model Implementation / Verification
- Repo / branch / source commit: `/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/NOAH_rep` / `main` / `b77e8b9` (dirty: new pipeline scripts and model modules)
- Dirty tree at execution: Yes (actively developing N-05 ETL, N-06 model architecture, and test verification suite)
- Interpreter / environment: `/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python` (Python 3.12.15, PyTorch 2.5.1, DuckDB 1.1.3, HuggingFace Transformers, Timm 1.0.30)
- Data root / scope: `/home/wey371/rajpurkarlab/datasets` (MIMIC-IV hosp, icu, ed, note, ecg, cxr; ECHO disabled for pilot)
- Slurm job / state / exit code: Local execution verified; `sbatch/run_pilot_etl_cpu.sh` configured for 1,000 subcohort
- Command / config / seed: 
  - Curation: `python scripts/run_curation_pipeline.py --max_patients 10 --device cpu`
  - Training test: `python scripts/train_pilot.py --epochs 2 --batch_size 4 --device cpu`
  - Test suite: `PYTHONPATH=. pytest tests/ -v`

## Results

1. **Data Curation Pipeline (N-05):**
   - DuckDB high-throughput event extraction completed in 120.85s across all MIMIC tables.
   - Handled quoted multi-line clinical texts using `parallel=false, null_padding=true, ignore_errors=true`.
   - Extracted 12,642 raw clinical events across 10/10 patients (100% eligibility >= 64 events).
   - BioLORD-2023 text specifics deduplication: 1,155 unique clinical phrases embedded in 64.32s into SQLite cache.
   - CXR and ECG media attached to records and cached.
   - Sliding window construction with context length $L \le 1024$, overlap 256, 6 demographic prefix tokens.
   - **Exact loss ownership verified:** `total_raw_events = 12,642 == total_scored_tokens = 12,642`. Demographic prefixes and overlap tokens un-scored (`loss_mask = False`).
   - Distribution across modalities:
     - Numeric: 7,738
     - Categorical: 4,490
     - Text: 291
     - ECG: 74
     - CXR: 49
     - ECHO: 0 (deferred)
   - Shards saved: `data/processed/pilot_train_windows.pt` and `data/processed/curation_summary.json`.

2. **Complete NOAH Model Architecture (N-06):**
   - `src/models/projections.py`: `ModalityInputProjector` mapping categories, modalities, 44-D Fourier numeric vectors, 768-D text/CXR vectors, and 192-D ECG vectors into $d_{\text{model}} = 256$.
   - `src/models/transformer.py`: `NOAHTransformerBackbone` with bidirectional directional temporal frequency attention ($\mathcal{T}^Q, \mathcal{T}^K$) in layers 0-3, causal masking, and LayerScale ($10^{-4}$).
   - `src/models/variational.py`: `ConditionalPrior` $p(z_t \mid h_{t-1})$ with smooth $\sigma_p$ squashing to $[0.05, 2.0]$, `InformedPosterior` $q(z_t \mid h_{t-1}, e_t)$, exact analytical Gaussian KL divergence.
   - `src/models/decoder.py`: `HierarchicalMultimodalDecoder` cascading feature MLP ($z_t \to \text{category} \to \text{specifics} \to \text{modality} \to \text{values} \to \text{time hurdle}$).
   - `src/models/noah.py`: Full model uniting projections, transformer, variational latent module, hierarchical decoder, dual-path loss ($0.3 L_{\text{post}} + \beta L_{\text{KL}} + 0.1 L_{\text{prior}}$), and prior-only free rollout.
   - 18/18 pytest tests passing across `tests/test_temporal_oracle.py`, `tests/test_variational_oracle.py`, `tests/test_window_builder.py`, and `tests/test_noah_model.py`.

3. **Pilot Training Sanity Check (N-07 Prototype):**
   - 2 epochs trained on 19 curated windows (5,435,924 trainable parameters):
     - Epoch 1 ($\beta=0.50$): Loss: 222.68 | RecPost: 15.61 | KL: 432.89 | RecPrior: 15.52
     - Epoch 2 ($\beta=1.00$): Loss: 161.46 | RecPost: 13.70 | KL: 155.93 | RecPrior: 14.13
   - Checkpoints saved: `checkpoints/pilot/pilot_checkpoint_epoch_1.pt` and `pilot_checkpoint_epoch_2.pt`.
   - Free rollout sanity check generated valid next event from prefix: Category: `LAB_TEST`, Modality: `NUMERIC`, Gap: `0.00h`.

## Outputs

- Dataset Shards: `data/processed/pilot_train_windows.pt`, `data/processed/curation_summary.json`
- Embedding Caches: `data/cache/text_embeddings.sqlite`, `data/cache/cxr/`, `data/cache/ecg/`
- Checkpoints: `checkpoints/pilot/pilot_checkpoint_epoch_2.pt`
- Source code:
  - `src/data/duckdb_extractor.py`
  - `src/data/embeddings_cache.py`
  - `src/data/window_builder.py`
  - `src/models/projections.py`
  - `src/models/transformer.py`
  - `src/models/variational.py`
  - `src/models/decoder.py`
  - `src/models/noah.py`
  - `scripts/run_curation_pipeline.py`
  - `scripts/train_pilot.py`
  - `tests/test_noah_model.py`

## Next action

- Launch full 1,000 subcohort curation job on HMS O2 (`sbatch sbatch/run_pilot_etl_cpu.sh`) or submit pilot GPU training on Harvard O2 (`sbatch sbatch/run_pilot_gpu.sh`).
