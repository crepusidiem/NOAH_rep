# NOAH clean development baseline

This isolated worktree preserves reusable candidate mathematics and evaluation guards while the legacy checkout remains intact. The bounded M1 occurrence-time mortality pipeline is implemented; it is not a claim of exact paper reproduction or strict prospective prediction. Read [migration decisions and milestones](docs/MIGRATION_BASELINE.md) and [current status](docs/PROJECT_STATUS.md).

The authoritative documents remain [the paper](paper/NOAH.pdf), [reproduction specification](docs/REPRODUCTION_SPEC.md), [assumptions](docs/ASSUMPTIONS.md), [traceability](docs/PAPER_TRACEABILITY.md) and [test plan](docs/TEST_PLAN.md). No parallel scientific specification is introduced.

Use the recorded interpreter with Python isolation:

```bash
/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q
/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py status
```

Runtime uses `noah_baseline`; legacy `src`/`scripts` imports are blocked. There is no legacy cache loader, inherited dataset path or alpha launcher. The explicitly authorized `m1` command supports CPU-bounded integration only; generic substantial training stays blocked. New artifacts belong under ignored `artifacts/<profile>/`; raw sources will be read in place only after an explicit source manifest and methodology contract exist. Profiles are disabled templates, with evaluation mode and experiment scale recorded separately. The migration inventory records exact source identities and the namespace-only changes.
