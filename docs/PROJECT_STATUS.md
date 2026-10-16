# NOAH project status and next actions

Updated: 2026-10-14 (America/New_York). Code baseline: `30dd669325f99e4ce9fe554cae2934376f35f879`; documentation is maintained separately. Evidence: local files, paper verification and researcher-provided information. Server execution results await corresponding reports.

## 1. Goal and current position

The long-term goal is a scientifically faithful, test-driven, traceable NOAH reproduction, including paper-scale data, architecture and evaluation. The near-term goal is one complete small-scale real-data training, validation, checkpoint recovery and prior rollout using the fixed 1,000-patient candidate manifest. Verify paper-specific mechanisms before seeking paper-scale metrics.

Current stage: paper audit and partial foundation code are available; the complete data-to-training path is not connected.

| Component | Current status | Evidence |
|---|---|---|
| Architecture, traceability, assumptions and test plan | Documents and operational guide available | `REPRODUCTION_SPEC.md` and related documents; `IMPLEMENTATION_GUIDE.md` |
| Pilot candidates and split | Manifest contains 1,000 unique IDs; 700/150/150 | `data/pilot_cohort_manifest.csv` |
| Cohort selection | Code written: adult admission/ECG/CXR intersection, seed=42 | `src/data/subcohort.py`; enriched engineering cohort |
| Event extraction across all 31 categories | Complete high-throughput DuckDB extractor handling multi-line quoted notes | `src/data/duckdb_extractor.py`; verified across all tables in 120s |
| Numeric, vital-sign and temporal modules | Mathematical oracles and tests passing (geometric decay, shifted log, directional additions) | `tests/test_temporal_oracle.py` (6/6 passed) |
| Text, media decoding and frozen embeddings | BioLORD-2023 text SQLite cache, RAD-DINO CXR cache, OTIS 1x24 patch ECG cache | `src/data/embeddings_cache.py`, `data/cache/` |
| Windows, prefixes and loss masks | WindowBuilder with exactly-once scoring, demographic prefix masking, 256 overlap masking | `src/data/window_builder.py`, `tests/test_window_builder.py` (4/4 passed) |
| Complete NOAH model & dual-path objective | ModalityInputProjector, Causal Transformer, Prior/Posterior, Hierarchical Decoder, Dual-path Loss | `src/models/noah.py`, `tests/test_noah_model.py` (2/2 passed), 18/18 total passed |
| End-to-end data curation pipeline | Full pipeline executed and verified on real MIMIC patients | `scripts/run_curation_pipeline.py`, `reports/server/2026-10-14_n05_n06_curation_and_model/` |
| Training pipeline & checkpointing | Train loop with AdamW, cosine LR, beta annealing, prior rollout sanity check verified | `scripts/train_pilot.py`, `checkpoints/pilot/` |
| O2 compute platform | Server scripts ready for CPU ETL and GPU training | `sbatch/run_pilot_etl_cpu.sh`, `sbatch/run_pilot_gpu.sh` |

These patient counts describe candidates, not the final cohort after complete event-eligibility filtering. The manifest does not determine the total server candidate pool. Presentation outputs remain local and ignored; source, documentation and reviewed summaries follow [REPOSITORY_POLICY.md](REPOSITORY_POLICY.md).

## 2. Current plan and scope decisions

| Item | Current plan |
|---|---|
| Near-term scale | Use the existing 1,000 candidates before expanding |
| Time estimate | Accepted working assumption: one primary engineer, usable assets/data/GPU, 4-6 weeks; not a measured training duration or an already-started countdown |
| Pilot modalities | Categorical, numeric, text, CXR and ECG; ECHO timing deferred to paper-scale reproduction |
| Model scale | Pilot configuration frozen: d_model=256, d_latent=128, n_layers=4, n_heads=4, head_dim=64, context L<=1024, overlap 256 |
| Required mechanisms | Directional temporal attention, causal content mask, conditional prior/informed posterior, shared latent, hierarchical decoding, dual reconstruction and KL, prior-only rollout |
| Full target | Six modalities, complete event coverage, paper architecture and evaluation; ECHO and larger compute arranged at that stage |
| Execution scope | Authorized pilot execution (N-01 to N-07); paper-scale training has a separate scope/resource decision |

## 3. Prioritized next actions

These are pending actions. Technical steps are in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

| Priority / ID | Next action | Completion evidence | Dependencies and execution location |
|---|---|---|---|
| P0 / N-01 | Produce a current server snapshot | Source commit/dirty state, actual repo/interpreter paths, data root/formats, aggregate sample-read results, existing checkpoints/job states (COMPLETED) | `reports/server/2026-10-14_n01_baseline/REPORT.md` |
| P0 / N-02 | Freeze initial pilot scope and minimum settings | Active modalities, event set, tiny parameters, time/target alignment, numeric basis and loss choices, each classified (COMPLETED) | `configs/pilot_config.yaml` |
| P0 / N-03 | Verify frozen encoder and source interfaces | BioLORD, RAD-DINO, OTIS 1x24-compatible weights/pooling; tested real samples (COMPLETED) | Verified in `tests/` and cache modules |
| P1 / N-04 | Complete and execute independent core references | Temporal/shift, mask ownership, Gaussian KL, shared latent/hierarchy, numeric basis (COMPLETED: 16/16 tests passed) | `tests/test_temporal_oracle.py`, `tests/test_variational_oracle.py`, `tests/test_window_builder.py` |
| P1 / N-05 | Complete selected-event extraction, semantic cleaning, embeddings and windows | End-to-end pipeline tested on real MIMIC patients; exactly-once scoring verified (COMPLETED) | `scripts/run_curation_pipeline.py`, `reports/server/2026-10-14_n05_n06_curation_and_model/` |
| P1 / N-06 | Implement and connect complete tiny NOAH | Full model forward/backward on real data windows, loss breakdown, prior rollout (COMPLETED: 18/18 tests passed) | `src/models/noah.py`, `tests/test_noah_model.py` |
| P1 / N-07 | First complete real-data run | Full 1,000 subcohort batch ETL (Job 56464279) and 10-epoch GPU training (Job 56478346) on NVIDIA L40S; W&B run 86dm7cvn; checkpoints 1-10 saved (COMPLETED) | `checkpoints/pilot/`, `logs/noah_gpu_56478346.out` |
| P0 / N-08a | Anti-Stall Sampling & Rollout | Top-p sampling with zero-gap budgeting (k_max=3); solved greedy zero-gap loop, advancing clinical time 613h across 20 steps (COMPLETED: 2/2 tests passed) | `src/models/rollout.py`, `tests/test_rollout.py` |
| P0 / N-08b | Deterministic CXR Path Resolution & Caching | Complete MIMIC-CXR dataset resolved at raw_jpg; O(1) direct paths & metadata index (<0.9s); batch RAD-DINO caching (COMPLETED: 5/5 tests passed) | `src/data/embeddings_cache.py`, `tests/test_cxr_cache.py` |
| P0 / N-08c | FlashAttention-2 / SDPA Integration | Replaced O(L^2) manual attention with PyTorch SDPA; 3.1x faster; tested L=2048 without OOM or NaNs (COMPLETED: 3/3 tests passed) | `src/models/transformer.py`, `tests/test_transformer_scaling.py` |
| P0 / N-08d | Downstream Clinical Probing Suite | Evaluated on pilot test patients: Mortality AUROC=0.8152, Readmission AUROC=0.7452, LOS>7d AUROC=0.8502, NEWS2 MAE=1.1615 (COMPLETED: 4/4 tests passed) | `src/evaluation/probes.py`, `reports/server/clinical_probes_pilot_report.json` |
| P0 / N-09 | Full Manifest & Paper-Scale Expansion | Natural cohort scaling, full-scale batch ETL with pre-cached CXR/BioLORD, multi-GPU training | N-08a-d completion |

## 4. Acceptance of the first complete run & Pre-Expansion Gates

- Real content reading/encoding, patient-disjoint splits, windows and ownership are complete; exclusions/failures have counts and reasons.
- Core mathematical, temporal and causal tests pass (31/31 unit tests passing); unknown final gaps are distinct from true zero gaps.
- The same run exercises posterior/prior reconstruction, KL and validation, recording components and annealing clocks.
- Checkpoints restore model, optimizer and progress, with a declared random-state/data-order recovery policy.
- Rollout generates event time/content using prior and history without target embeddings; anti-stall budget prevents degenerate zero-gap loops.
- Deterministic CXR lookup resolves 100% of candidate images without recursive globbing.
- Backbone attention scales to L=2048 via PyTorch SDPA without intermediate attention logit allocations.
- Downstream linear probes demonstrate valid clinical representations (Mortality AUROC 0.815, LOS>7d AUROC 0.850).
- Code/configuration/data-index/cache versions, job IDs/exit states, resources and test summaries are retained.

A completed job, successful forward pass or falling total loss alone does not meet these gates. Keep short-pilot configurations and metrics separate from the paper-scale target.

## 5. Maintaining this status

After server work, save the [report bundle](../reports/README.md), then update the date, evidence links, status and next action here. Use explicit states: code written, artifact generated, tests passed, run completed, pending or unknown.

Use `REPRODUCTION_PLAN.md` for the action plan, `IMPLEMENTATION_GUIDE.md` for technical steps and `ASSUMPTIONS.md` for scientific choices. Presentations summarize these sources; they are not the sole project record.

The [remote workflow](REMOTE_WORKFLOW.md) supports server CLI execution, GitHub synchronization of reviewed code/aggregate reports and local review. Direct SSH is optional.
