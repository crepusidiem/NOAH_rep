# Server execution report convention

This document defines how server execution evidence is saved. No actual report bundle is currently present. Published content follows [REPOSITORY_POLICY.md](../docs/REPOSITORY_POLICY.md). All repository reports and reusable agent prompts must be in English.

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
