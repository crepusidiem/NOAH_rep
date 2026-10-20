# M1 executed category/heart-rate mortality integration

Executed2026-10-19 (America/New_York), existing O2 allocation56795452, CPU with2 threads, no new jobs or GPU kernels. Branch: `development/clean-baseline`. The implementation runs the complete source event -> provenance/window -> candidate training -> checkpoint/recovery -> single landmark -> encounter label -> probe -> prediction -> metric chain. No historical source, raw records, old labels/models, embedding caches or prior metric artifacts were changed. No alpha sweep, external encoder, NEWS2, MC or large-cohort experiment ran.

## Implementation and protocol

[configs/m1.json](../../../configs/m1.json) records the effective bounded protocol. [m1/data.py](../../../packages/noah_baseline/m1/data.py) extracts versioned hospital and ED source records and constructs provenance-aligned windows. [m1/training.py](../../../packages/noah_baseline/m1/training.py) supplies one bounded train/save/load path; [m1/verification.py](../../../packages/noah_baseline/m1/verification.py) supplies independent source and objective references. [m1/pipeline.py](../../../packages/noah_baseline/m1/pipeline.py) executes exact-state extraction, development-only probing and gated metric replay. The existing model, temporal, prior/posterior, decoder and loss files retain their B0 migration hashes.

Inputs are the original verified1000-patient cohort, MIMIC-IV2.2 hospital admissions/patients and MIMIC-IV-ED2.2 stays/vitalsign. The cohort's missing materialized birth-year column is handled through its actual anchor-year/age schema and direct patients-table parity. Raw CSV record identities remain unique even for identical source observations. Unit mappings use the official [ED vital-sign documentation](https://mimic.mit.edu/docs/iv/modules/ed/vitalsign.html). Only heart rate in bpm is used, avoiding mixed unlabeled numerical variables without specifics encoders. Source charttime supplies the occurrence coordinate; observed availability is absent and retained as unknown. The occurrence-time diagnostic cannot be promoted to strict prospective prediction.

The first hospital admission is selected by `(admittime, hadm_id)` before eligibility, not by mortality or last-recorded-stay status. Require alive and still in that same encounter at admission+24h; unknown or contradictory mortality/timing is excluded. No later admission replaces an ineligible first index. Context uses only that encounter's events through the cutoff, including its linked pre-admission ED observations. Birth year is a source-derived numeric prefix; five other demographic attributes are explicitly Unknown/unconditioned, with no text targets or hidden encoder. Retained gaps are recomputed; chunk-terminal next-gap0 has an unknown mask. One last-event hidden state in the final bounded context window is used per index, never pooled states.

The candidate pilot dimensions remain256/128,4 layers,4 heads, FFN4; all candidate heads remain present. Count:4,760,468 parameters, a new clean revision rather than a legacy checkpoint architecture. Context64, overlap16, minimum-events1, omitted specifics and limited sources are engineering deviations. Weighted component means and the fixed train-only inverse-frequency candidate remain unchanged; this is not a weighting-method experiment. Training uses2 microbatches per update, fixed source order, separate LR/KL clocks, candidate Muon/AdamW, clipping10 and four retained updates. The exact-resume comparison adds one duplicate reference update. These few warmup operations are not meaningful converged pretraining.

Checkpoint restore refreshes cached optimizer parameter-group aliases so the resumed LR reaches actual optimizer groups; uninterrupted/resumed traces, all model tensors and optimizer buffers agree bitwise. Checkpoints bind architecture/configuration/source/data/windows, RNGs and cursor. Repeated loading with changed data/config/revision fails. An independent reconstruction/normal-KL oracle agrees with the real-data objective exactly (3.800625801086426 for the checked example); shifted prior/posterior indexing is checked directly.

StandardScaler and logistic-regression coefficients are fit on train only. Validation log loss selects C from0.01/0.1/1; C=0.1 was selected. There is no train+validation refit or test tuning. Perturbing all real held-out test features and labels leaves preprocessing, selected C and coefficients unchanged. All probabilities, representations, labels and metrics are saved; float64 metric reductions and reloaded probe probabilities replay exactly.

## Executed acceptance

| ID | Source-like fixtures and real evidence | Result |
|---|---|---|
| M1-01 | Actual cohort/source CSV schemas; unique source records, explicit units/unknown availability, cross-encounter join rejection. Independent SQL matches Python index/eligibility/labels for1000 patients;11 real numeric source records match raw CSV. | PASS |
| M1-02 | Future suffix, delayed/unknown availability, terminal-gap and demographic tests. Real feature/state suffix invariance for surviving and deceased encounters in all3 splits. | PASS |
| M1-03 | Immutable patient split membership, train-only target/weight/probe fitting, first-before-eligibility index selection; short index never replaced by a later fatal admission. | PASS |
| M1-04 | Independent shifted owner ledgers and masks for every saved window; padding/overlap tests and direct prior/posterior indexing. Additional real surviving/deceased ownership stresses each span5 windows and22 unique targets, with no duplicate loss ownership. | PASS |
| M1-05 | Independent weighted CE/BCE/MSE/KL reference and per-microbatch accumulated gradients; bounded real forward/backward/optimizer operations, finite losses and separate clocks. Candidate means are not claimed equivalent to arbitrary global-event normalization. | PASS |
| M1-06 | Fixture and real checkpoint restore: exact data/config identity, RNG/cursor, bitwise model/optimizer/reference trace and final checkpoint reload. | PASS |
| M1-07 | Discordant-encounter mortality and risk-set fixtures; all743 real representations equal independent exact `h[last]`, with encounter/event identity in predictions. | PASS |
| M1-08 | Development-only fitting/selection, real and fixture test-perturbation invariance, saved-probe/prediction metric replay, actual release pass for retrospective diagnostic and rejection of prospective/paper-faithful claims. | PASS |

Fixture suite:8 passed in8.58s. Full clean suite:74 passed in20.45s, no failures/skips. Real canonical run measured21.34s through metric replay; final evidence hashing/release checks follow. Independent artifact replay verifies32 saved files and exact serialized metrics. Additional short-window stress is verification only; it does not alter the context64 training/evaluation results. See [fixture_results.json](fixture_results.json), [artifact_replay.json](artifact_replay.json), [summary.json](summary.json) and [verify_artifacts.py](verify_artifacts.py).

## Real-data demonstration

The1000 source patients have4,129 hospital admissions;666 have multiple admissions. Extraction produced25,108 events, all with unknown availability. Eligibility excludes255 stays ending by24h,1 contradictory mortality record and1 death timestamp after the encounter end. There are743 eligible patients/windows and526 train windows. Short/overlapping-window behavior is additionally exercised in fixtures and real window stress tests.

| Split | Eligible encounters | In-hospital deaths | AUROC | AUPRC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Train |526|9|0.979798|0.522409|0.012510|0.047393|
| Validation |102|1|0.950495|0.166667|0.009406|0.041514|
| Test |115|2|0.907080|0.175000|0.016175|0.068394|

At threshold0.5 the test balanced accuracy is0.5: all test predictions are negative. High accuracy and a ranking AUROC based on only2 test deaths do not establish clinical utility, calibrated risk, a benefit from training, or reproduction of the paper. This is an occurrence-time engineering demonstration after four retained updates. No clinical-performance or strictly prospective claim is supported.

## Artifacts and reproducibility

Private run: `artifacts/reduced_pilot/runs/m1-real-20261019-02/`, ignored by Git. It contains source tar, actual configuration/runtime, full source/cohort manifests, events, provenance-aligned windows, checkpoint/recovery, feature/label identities, fitted probe, validation selection,743 prediction rows, metrics, executed source/objective/suffix/probe checks, release evidence and SHA-256 inventory. Do not publish patient-level artifacts. `experiment_manifest.json` binds all important artifacts and the scientific interpretation. The source base commit was `ac25d36ab51de73dfa7b802bfc97701a1c3a00c1` plus the preserved implementation snapshot; the source tar SHA-256 is `872c530e3dae832192f5915bff80e9562fec857de425d8f78b29f7b3be7545a2`. Future code commit IDs must not be substituted for this executed source identity. Model checkpoint SHA-256: `204ed4c38b1e078191eec986bc7dd6cb0ef2076255fbd3573ecaa514b9869571`.

Commands, from this worktree with the recorded interpreter:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py m1-check --output artifacts/reduced_pilot/evidence/m1-fixtures-v7.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py m1 --hosp /home/wey371/rajpurkarlab/datasets/mimic-iv/physionet.org/files/mimiciv/2.2/hosp --ed /home/wey371/rajpurkarlab/datasets/mimic-iv-ed/physionet.org/files/mimic-iv-ed/2.2/ed --cohort /n/data1/hms/dbmi/rajpurkar/lab/home/wey371/NOAH_rep/data/pilot_cohort_manifest.csv --run-id m1-real-20261019-02 --fixture-evidence artifacts/reduced_pilot/evidence/m1-fixtures-v7.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q -o cache_dir=/tmp/noah_m1_final_test_cache
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I reports/server/2026-10-19_m1/verify_artifacts.py artifacts/reduced_pilot/runs/m1-real-20261019-02 artifacts/reduced_pilot/evidence/m1-real-artifact-replay-v1.json
```

Executed paths are immutable run identifiers. Repeating requires fresh run/evidence names and fixture results bound to the active package/test hashes. An earlier source-header attempt was preserved as a pre-training failure without checkpoints; successful metrics come only from run02. No outcome/prediction artifacts were overwritten.

There are no remaining blockers for the stated M1 engineering acceptance. Larger training needs separate authorization. Unknown availability and small event counts restrict interpretation and remain explicit; they do not block the permitted occurrence-time demonstration. External modalities, NEWS2, readmission, MC, author architecture choices and alpha remain outside M1.
