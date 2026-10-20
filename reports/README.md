# Clean-baseline reporting context

Reports copied from earlier dated audits are historical evidence about their recorded revisions. Their private source/artifact paths belong to the legacy checkout, not this worktree. Current evidence is grouped by milestone:

| Evidence | Report directory | Interpretation |
|---|---|---|
| Legacy architectural/validity audit | `server/2026-10-19_architecture_audit/`, `server/2026-10-19_controlled_remediation/` | Historical revision findings and preserved checkpoints |
| Source readmission investigation | `server/2026-10-19_readmission_intervals/` | Read-only source/ETL/label investigation; clinical cases remain private |
| B0 isolated initialization | `server/2026-10-19_clean_baseline/` | Preservation, isolation and component migration |
| M1 real-data engineering chain | `server/2026-10-19_m1/` | Executed bounded pipeline and acceptance checks |
| M1.5 preparation | `server/2026-10-20_m15_preflight/` | Read-only target inventory and resource proposal |
| M1.5 bounded learning | `server/2026-10-20_m15_learning/` | Actual fixed-budget teacher-forced, time-conditioned category prediction |

No historical clinical score is a new baseline result. The M1.5 report records executed source `b7a848f`; its results commit is `6435b3d`. Later documentation/synchronization commits do not replace either identity.

# Server execution report convention

This document defines how server execution evidence is saved. Dated report bundles and aggregate experiment reports are present under `server/`; each has its own execution evidence and scope. Published content follows [REPOSITORY_POLICY.md](../docs/REPOSITORY_POLICY.md). All repository reports and reusable agent prompts must be in English.

## Location

```text
reports/
  README.md
  server/
    YYYY-MM-DD_<run_id>/
      REPORT.md                 Commands, scope, results, failures and next action
      environment.json         Selected environment/version fields (optional)
      summary.json             Aggregate counts, shapes, tests and resources (optional)
      sanitized_job_log.txt     Reviewed log safe to synchronize (optional)
```

One `REPORT.md` is enough to start. These names/formats are engineering conventions, not paper requirements. Keep code/schema/configuration and reports aligned on the same branch. Record the source commit at execution time, not the later commit containing the report.

## Required information

1. Actual date/timezone, run ID, stage (inspection/tests/ETL/encoding/training/rollout) and executor.
2. Actual absolute repository path, branch, source commit and dirty state. For uncommitted code, attach a reviewed diff or equivalent version record.
3. Actual interpreter/environment/package versions; node type, Slurm job ID, state, exit code and requested versus obtained resources when relevant.
4. Data root/resolved location, versions, formats such as `.csv`/`.csv.gz`, and table/media scope. Distinguish path existence from successful reading.
5. Exact sanitized command/entry point, configuration path/hash, seed, sample size and sampling method.
6. Measured aggregate results: candidate/retained patients, splits, events/modalities, successes/failures, embedding shapes, windows/scored tokens and test outcomes.
7. Server artifact locations, relative links to safe summaries and data/cache/configuration versions.
8. Measured wall time, memory and throughput where available. Separate queue, encoding and training durations.
9. Issues and affected milestones; next action and corresponding N-ID in `PROJECT_STATUS.md`.
10. Use `NOT_RUN` for unexecuted checks, `NOT_MEASURED` for unavailable measurements and `UNKNOWN` for unconfirmed facts. Requested resources are not obtained resources.

Synchronize reviewed code and aggregate summaries, not clinical text, raw media, embedding caches or new patient-level tables. Remove patient content and credentials from logs. Record selected environment fields rather than publishing a complete `env` dump.

## Minimal REPORT.md template

```markdown
# Server report: <run_id>

- Checked at / timezone: <actual time>
- Stage: <inspection | tests | ETL | encoding | training | rollout>
- Repo / branch / source commit: <actual values>
- Dirty tree at execution: <yes/no and explanation>
- Interpreter / environment: <actual path and versions>
- Data root / scope: <actual paths, versions, formats, sample scope>
- Slurm job / state / exit code: <actual values or NOT_RUN>
- Command / config / seed: <exact sanitized values>

## Results
<Measured aggregate values. Use NOT_MEASURED for unmeasured fields.>

## Outputs
<Server artifact locations and links to safe summary files.>

## Failures and limitations
<Observed failures and untested scope.>

## Next action
<Action, dependency and PROJECT_STATUS N-ID.>
```

See [REMOTE_WORKFLOW.md](../docs/REMOTE_WORKFLOW.md) for synchronization and [PROJECT_STATUS.md](../docs/PROJECT_STATUS.md) for current progress. Documentation describes the current state; Git retains version history.

## Experiment manifests and scientific release

Every future evaluation requires an applicable experiment manifest. Clinical evaluations use the structure in `configs/evaluation_manifest.template.json`, validated by `packages/noah_baseline/evaluation/contracts.py`; the archived implementation used `src/evaluation/contracts.py`. The template intentionally contains unresolved/null fields and cannot pass a gate. Record code commit and source artifact hash, architecture revision/module counts, checkpoint identity, dataset versions and mapping/cache manifests, encoder identities/recipe hashes, random seeds, patient split artifact, cutoff and next-gap/demographic policies, encounter/endpoint/horizon/censor definitions, availability handling, probe population/validation, private predictions, metric/uncertainty implementation and all approved assumptions/deviations.

Two primary modes are supported: `paper_reproduction` and `strict_prospective`. The mode is not itself a conformance certificate. A paper profile may disclose true next-gap timing as a literal reconstruction/retrieval information condition; a strict prospective profile rejects it and requires observed source availability and cutoff-bound metadata. A full-record mean is never a landmark representation. Existing provenance-free shards do not acquire these properties by adding a mode string.

Use `python -I tools/run.py release` before publishing a requested clinical label. `write_validated_report` performs the same checks and refuses to overwrite an existing experiment report. Actual source/checkpoint/split/prediction files and check artifacts are hashed. Relevant required checks and methodology decisions must pass; unknown, skipped, absent, mismatched or pending evidence blocks the claim. Paper-faithful claims also reject deviations and the known candidate parameter mismatch. Report-to-checkpoint associations inferred from names remain unverified until execution lineage/prediction replay is established.

Prediction records remain private. Landmark records include `representation_kind=single_landmark`, a `LandmarkRequest`, ordered per-token `EventEvidence` (including demographic evidence), finite label/prediction and their source identities. Release checks examine these records, not only a self-declared PASS flag. The reporting gate prevents unsupported label promotion; it does not prove a fabricated evidence file is an authentic execution. Preserve executable test output and the exact source snapshot that produced it.

Historical pooled mortality/LOS metrics are exploratory retrospective subject-label diagnostics, not prospective or paper landmark results. Historical readmission/NEWS2 targets need correction before clinical endpoint interpretation. Posterior reconstruction remains a conditional reconstruction diagnostic with its exact masks/recipe/provenance, not free forecasting. See the [remediation matrix and historical metric classification](../docs/ARCHITECTURE_AUDIT.md#historical-metric-interpretation).
