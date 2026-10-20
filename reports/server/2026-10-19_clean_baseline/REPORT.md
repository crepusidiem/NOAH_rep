# Clean baseline initialization evidence

Executed2026-10-19, America/New_York, existing O2 allocation56795452. The original checkout remains on `main`; no legacy source, configuration, documentation, original index, dataset, cache, checkpoint or historical metric was overwritten. No new Slurm job, real-data training, clinical evaluation or alpha-sweep ran. Unit tests use synthetic tensors and temporary artifacts only; optimizer update tests are engineering checks, not training experiments.

Local archive branch `archive/legacy-2026-10-19`: `edf6a06267a571ce53a3d3718d091e106b7799dd`, parent `1d21fd6f85d4fabbf6cdf0022e865db82cc52dc7`, tree `df2f26cea7f23fc53969336fa961239a77562bdd`. A separate Git index captured111 reviewed tracked/untracked files. Original staged/unstaged patches, original index/status, current-source copies and verified Git history bundle are under the original checkout's ignored `output/baselines/2026-10-19_clean_init/`. Source inventory SHA-256: `622166c2b8380e81ddf2829f42f807933dd360eaffb0829b71690ef19ef9f22c`. The inherited private cohort remains only in legacy/history, not the clean active tree. No publication or push occurred.

Full independent checkpoint-snapshot reverification passed:51 files,55,663,961,469 bytes; all SHA-256 values match the preserved manifest, independent original/copy inodes verified where originals remain, and all8 snapshot-integrity entries passed. This required111.24s of CPU/filesystem I/O. It does not establish missing historical source/encoder/runtime lineage. Checkpoint and dataset bytes were not duplicated for development.

The clean branch is `development/clean-baseline`; worktree is `<legacy-root>/output/worktrees/clean-baseline`. Development code uses `packages/noah_baseline/`, explicit local artifact namespaces, an isolated Python launcher and disabled paper/prospective/pilot profile templates. Canonical scientific documents are retained in place. The legacy `src`, trainer, clinical ETL, embedding loaders, pooled evaluator and alpha launcher are absent from the active runtime. Historical dated audit reports remain reference evidence, with original private paths interpreted in the legacy checkout.

Migrated component text is unchanged except for the import namespace; `migration_inventory.json` binds source and destination hashes and lists test adaptations/exclusions. Source, cache/symlink, profile and package-origin checks are newly implemented. No likelihood, loss reduction, architecture, chronology, encoder recipe or clinical methodology was silently changed. Conditional candidate components retain their documented assumptions; passing tests is not exact paper fidelity.

Verification command, from the clean worktree:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q -o cache_dir=/tmp/noah_clean_baseline_pytest_cache
```

Result: **66 passed in54.74s**, no failures or skips. Coverage includes independent temporal and KL oracles, prior isolation, causal attention/masks, ECHO gradients, optimizer routing/state, zero-gap and prior-only rollout mechanics, overlap/window ownership, demographic candidates, landmark/manifest/report gates, and adversarial source/cache isolation. Meta-device module counts reconcile to148,957,844, not the paper's150,322,970. Python syntax checks passed. The tested package environment is recorded in `environment.baseline.json`; it is not an author environment or complete portable legacy lock.

[Migration decisions, pending approvals, ordered milestones and M1 acceptance tests](../../../docs/MIGRATION_BASELINE.md) define the next work. M1 real-data provenance/training/landmark integration remains NOT_IMPLEMENTED. NEWS2, clinical readmission policy, author assets/count, loss normalization, prospective availability and alpha scope remain open as applicable. The supported launcher has no training or sweep command. [summary.json](summary.json) records safe aggregate identities/results; final Git/worktree/bundle and untouched-legacy receipt is stored privately after the initialization commit.
