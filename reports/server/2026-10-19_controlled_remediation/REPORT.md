# Controlled validity remediation: preservation and evaluation guards

- Date/timezone: 2026-10-19, America/New_York.
- Repository: `/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/NOAH_rep`.
- Baseline: `1d21fd6f85d4fabbf6cdf0022e865db82cc52dc7`, dirty tree including pre-existing user work. Changes remain uncommitted; preserved source inventory/patch identify the pre-remediation tree separately from the remediation code.
- Execution: existing O2 allocation56795452 on `compute-b-16-177.o2.rc.hms.harvard.edu`; no new jobs submitted. Checkpoint copying/hashing, metadata inspection and synthetic tests used CPU/filesystem I/O. No GPU kernels, clinical-table scans, encoder inference, alpha-sweep or retraining were launched. GPU allocation details/peak memory/queue conditions NOT_MEASURED for this CPU-only stage.
- Interpreter: `/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python`; Python3.12, PyTorch2.14.1+cu130. Exact package versions are in the private preservation supplement.

## Preserved evidence

Private ignored snapshot: `output/reproducibility/2026-10-19_controlled_remediation_5b9bac05/`.

- Historical Git source archive; current pre-remediation source/config/report tree including untracked scripts and OTIS code; binary dirty patch/status; prior full audit and synthetic evidence.
- Fifty repository checkpoints plus the separately identified current synthetic smoke checkpoint:51 independent copies,55,663,961,469 bytes. Each copy was read back and SHA-256 verified; configs, epochs, module counts and state-shape hashes were inspected without model training. Copies and preserved evidence are read-only.
- Thirty checkpoints:152,655,380 parameters; twenty:5,435,924; one synthetic checkpoint:179,108. Current full-dimension assembly:148,957,844, with no corresponding completed real-data checkpoint identified.
- Current pre-remediation source inventory SHA-256: `3fcd5e8574d3edc4b261d159afd0137fd1711003c13129b46c5442048174bb31`. `manifest.json`, `manifest.sha256` and `snapshot_integrity.json` record byte identities, archives and preservation recipes.
- Cohort manifest preserved privately and verified equal to the historical HEAD version; package versions and proof that model files, trainer and numeric YAML files remained unchanged during this phase are in `preservation_supplement/`.
- The resulting remediation source is preserved separately in `remediation_code/`: 59 files, 449,414 bytes; inventory SHA-256 `af9d6042e62c13bfd543bd9db76ba7de063206973cea1f03dbd765d318815343`. This snapshot covers source, scripts, configurations, tests, launchers and repository agent instructions. It does not overwrite the pre-remediation snapshot or identify a new Git commit.

Actual prediction/fold artifacts for historical aggregate reports were not found in the reviewed report bundle. Report/checkpoint associations are explicitly filename/report claims, not replay-verified lineage. Historical source/runtime/encoder generation provenance remains unknown where absent. A byte-preservation snapshot cannot retroactively establish those facts or corrected metric validity. No historical metric values were recomputed or overwritten.

## Implemented changes

- Explicit experiment manifests and actual file/evidence hash checks; prediction-provenance inspection, required-check and methodology approval gates; no unsupported retrospective-to-prospective promotion, architecture mixing, partial report overwrites or release based on missing/skipped evidence.
- New single-landmark extraction and encounter-label primitives with explicit inputs. They reject full-record pooling, future events, unknown/delayed prospective availability, future demographic snapshots, wrong event/encounter identity and incompatible prepared gap features. No index-encounter or unknown-gap scientific default is selected.
- Legacy pooled/subject-max clinical evaluation, old verification/reconciliation/aggregation paths and alpha launcher held before execution. Existing scripts are preserved in the snapshot. Real-data evaluation is intentionally not replaced by a partially implemented clinical method.
- Unknown aggregate source availability remains unknown; missing/nonfinite core vitals cannot become a score; nonpositive aggregation span/stride and missing patient split membership fail explicitly. Complete NEWS2 label/scoring/MLP methodology remains pending.
- Current checkpoint loader rejects unversioned legacy layouts and conflicting architecture YAML. Model definitions, training code, loss normalization and numeric YAML configuration are unchanged from the pre-remediation snapshot.

## Verification

Synthetic regression command:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -m pytest tests/test_evaluation_contracts.py tests/test_clinical_probes.py tests/test_temporal_oracle.py tests/test_variational_oracle.py tests/test_window_builder.py tests/test_dynamic_demographics.py -q -o cache_dir=/tmp/noah_remediation_test_cache
```

Result: **61 passed in 7.86 seconds**, with no failures or skips. Python syntax parsing and `git diff --check` also passed. The execution result and safe preservation summary are recorded in [summary.json](summary.json). Full-size training, corrected real-data extraction, source/view suffix-invariance, clinical label oracles, prediction replay, exact resume and paper-scale fidelity checks remain NOT_RUN or incomplete. Passing synthetic checks do not close those gates.

## Required decisions and next actions

[ARCHITECTURE_AUDIT.md](../../../docs/ARCHITECTURE_AUDIT.md#controlled-remediation-preserved-evidence-and-revision-separation) contains all15 findings grouped by confirmed defects, missing methodology, paper underspecification and reporting deficiencies; each has required changes, dataset/checkpoint impact, tests and experiment blockers. M01–M12 give alternatives and recommendations, all pending where not specified by the paper. The same section records the dependency order and historical metric interpretation.

Approve endpoint/information/probe contracts, implement provenance-complete source views, connect corrected encounter labels and landmarks, verify clinical labels/prediction replay and freeze one comparison recipe before any alpha-sweep. The launcher currently exits78. No retraining or new alpha/resource authorization is inferred from this remediation phase.
