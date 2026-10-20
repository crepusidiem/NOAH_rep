# M1.5 executed preflight

Date:2026-10-20, America/New_York. Branch `development/clean-baseline`; base commit `f5e0f58c6f414c191385a1eb6e0e062d08387b93` plus the new, hashed preflight script. This is executed read-only inventory/inference evidence, not a training report. See [aggregate measurements](aggregate.json), [execution proposal](../../../docs/M15_EXECUTION_PROPOSAL.md) and [proposed configuration](../../../configs/m15.json).

Working directory: `/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/NOAH_rep/output/worktrees/clean-baseline`.

Executed command:

```bash
/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m15_preflight.py --m1-run artifacts/reduced_pilot/runs/m1-real-20261019-02 --output artifacts/reduced_pilot/m15/preflight-20261020-01.json
```

Successful exit0. An initial invocation used the display name `validation` instead of source split `val` and failed before writing output; the corrected successful script maps the display label explicitly. No M1 files changed and neither invocation performed optimizer updates. The successful script hash is `939a6634f2f1f793f1cc7a83be883c2ada70f37cee8eca2d009dd9a68defce8d`; its output hash is `84882a78c8d9d9b478b2d8b30de96592f46531179823b07b10109f542bb64d53`. The proposed configuration hash is `0c8f96e8d937224e1c859e900bded3233c7ad006473bea72d7d3e62d066d4105`.

All32 M1 inventory entries pass content SHA256 and read-only permission checks. The preflight counted2335 train and436 validation owned targets, verified uniqueness and shifted-mask counts, checked time support conservation and patient disjointness. It inspected no test targets or mortality predictions. All four observed categories have at least100 train targets; no1–99-support category exists. Positive/zero/unknown train gaps are1784/25/526; validation330/4/102. Full counts and ledger hashes are in the aggregate report.

A fresh, untrained candidate has4,760,468 parameters. Inference-only timing:64 training windows following four warmups, two Torch CPU threads, mean7.070ms, median7.117ms, p957.436ms, peak process RSS1002.8MiB. This is not measured backward/optimizer throughput. No GPU or new scheduled job was used. M1.5 optimizer updates:0. No learning or clinical validity conclusion is available. The proposed60-minute CPU resource cap still requires approval.

Regression command `/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q` completed with exit0:74 passed in12.73s. These are the existing isolated synthetic/contract tests, including bounded training/recovery checks; they do not execute the proposed M1.5 training. Additional local assertions passed for configuration/preflight/source hashes, disabled execution, update-budget arithmetic and byte-identical private/public aggregate evidence. `git diff --check` passed. No model, data-recipe or M1 training source files changed.
