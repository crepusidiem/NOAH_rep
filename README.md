# NOAH clean development baseline

The clean baseline preserves reusable candidate mathematics and evaluation guards while the legacy checkout remains intact. M1's bounded occurrence-time mortality pipeline and the single approved M1.5 learning experiment are complete. M1.5 demonstrates restricted teacher-forced, time-conditioned category learning, with common-category and numeric failures recorded explicitly. Neither milestone establishes exact paper reproduction or strict prospective prediction. Read [current status](docs/PROJECT_STATUS.md), [M1.5 results](reports/server/2026-10-20_m15_learning/REPORT.md) and [migration decisions](docs/MIGRATION_BASELINE.md).

`main` is the current clean baseline; `development/clean-baseline` is its development track. `archive/legacy-2026-10-19` preserves the reviewed legacy/audit snapshot, and `legacy/preserved-2026-10-19` is checked out in the original server directory for compatible historical work. Keep these tracks separate: do not merge legacy implementations back into the clean model. Milestone tags identify B0, M1 and M1.5; executed source commits and private artifact hashes remain recorded in their reports.

The authoritative documents remain [the paper](paper/NOAH.pdf), [reproduction specification](docs/REPRODUCTION_SPEC.md), [assumptions](docs/ASSUMPTIONS.md), [traceability](docs/PAPER_TRACEABILITY.md) and [test plan](docs/TEST_PLAN.md). No parallel scientific specification is introduced.

Use the recorded interpreter with Python isolation:

```bash
/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q
/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py status
```

Runtime uses `noah_baseline`; legacy `src`/`scripts` imports are blocked. There is no legacy cache loader or alpha launcher. The `m1` command supports CPU-bounded integration; M1.5's separate one-run approval is consumed and its permanent run claim blocks a second execution. Further training requires separate authorization. New artifacts belong under ignored `artifacts/<profile>/`; the executed runs pin their source/data manifests and read sources in place. Checkpoints, clinical records, predictions, caches and rendered plots remain private on the server. Git contains reviewed source, configurations, tests, specifications and aggregate reports only. Profiles are disabled templates, with evaluation mode and experiment scale recorded separately. The migration inventory records exact source identities and the namespace-only changes.
