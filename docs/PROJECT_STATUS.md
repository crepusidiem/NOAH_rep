# Clean baseline status

Updated2026-10-19 (America/New_York). Branch: `development/clean-baseline`. Legacy source archive: `edf6a06267a571ce53a3d3718d091e106b7799dd`. The original checkout remains unchanged on `main`.

B0 initialization preserves candidate mathematics and engineering guards in the isolated `noah_baseline` namespace. Full checkpoint snapshot reverification passed for51 independent files (55,663,961,469 bytes). Clean validation:66 tests passed in54.74s, with no skips; execution evidence is in [the initialization report](../reports/server/2026-10-19_clean_baseline/REPORT.md). No real-data training, encoder extraction, clinical evaluation or alpha sweep has run in this worktree.

The package contains migrated candidate model/window components and explicit encounter/landmark/release primitives. There is no clinical ETL, embedding cache loader, historical trainer, pooled evaluator or alpha launcher. Data provenance serialization and real-data end-to-end integration remain unimplemented. The full-size candidate remains148,957,844 parameters versus paper150,322,970; no claim of exact architecture or existing-checkpoint compatibility is made.

Use [MIGRATION_BASELINE.md](MIGRATION_BASELINE.md) for module decisions, artifact boundaries, M1 acceptance tests and dependency order. Existing specification, assumptions, traceability and test-plan documents remain the scientific authorities. Historical audit reports are reference evidence about their recorded revisions, not current clean-baseline results.

Next: approve applicable M01–M12 decisions and the first reduced-scope protocol, then implement provenance-preserving source/views and the bounded M1 pipeline. Do not begin retraining or alpha work from initialization alone. Disabled profiles preserve paper reproduction, prospective evaluation and pilot scale as distinct choices.
