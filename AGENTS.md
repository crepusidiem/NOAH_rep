# NOAH reproduction instructions for coding agents

This repository reproduces NOAH (arXiv:2609.09140v1). The objective is a scientifically faithful, test-driven reproduction. Read these instructions before changing the data pipeline, model, training, or evaluation.

## Read first

Read [current project status](docs/PROJECT_STATUS.md) before choosing work or describing progress. It is a dated local evidence snapshot; refresh it from actual files and server reports.

1. [Technical implementation guide](docs/IMPLEMENTATION_GUIDE.md): operational steps, verified feature extraction, upstream interfaces, current code status and acceptance checks.
2. [Full reproduction specification](docs/REPRODUCTION_SPEC.md) and [paper traceability](docs/PAPER_TRACEABILITY.md): exact mechanisms, shapes, formulas and paper locations.
3. [Assumptions](docs/ASSUMPTIONS.md) and [test plan](docs/TEST_PLAN.md): independently configurable unresolved choices and scientific tests.
4. [Dependency-aware plan](docs/REPRODUCTION_PLAN.md) when choosing the next stage.

For remote O2 work, read [REMOTE_WORKFLOW.md](docs/REMOTE_WORKFLOW.md) and [reports/README.md](reports/README.md). Keep execution reports tied to source commits, actual paths, commands, measured aggregate results and job status. A report template is not an executed report. Sync reviewed code and safe summaries, not clinical content or embedding caches. Update project status only to the evidence level actually attained.

The authoritative paper is [paper/NOAH.pdf](paper/NOAH.pdf). Its printed/PDF page numbers coincide. Audit equation labels E01-E16 are local locators, not equation numbers printed in the paper.

## Authorization and working scope

Follow the researcher's active task: a documentation request stays within documentation; an implementation or execution request authorizes the named phase. Preserve existing code and proceed through N-01 to N-07 in the current plan when starting the pilot. Paper-scale training remains a separate resource and scope decision.

A later explicit researcher request to implement, test or run a phase authorizes work within that scope. Apply that authorization without asking for the same approval again. Routine choices within an authorized phase can proceed with explicit configuration and recorded assumptions. These instructions do not require a new permission question for every unresolved detail. Do not infer authorization to contact authors, publish data or start costly remote runs from a documentation task.

## Evidence and fidelity

- Treat supplied NOAH text/figures/tables as paper requirements. Code tied to the same NOAH version can clarify them. Encoder upstream code supplies candidate implementations, not proof of NOAH's exact checkpoint or settings. Existing repository code is not an independent source of paper truth.
- Classify each decision as EXPLICIT, DERIVED, ASSUMED or DEVIATION. Separately record its evidence source. Keep unresolved configuration keys separate rather than hiding defaults behind one fidelity flag.
- Implement specified mechanisms directly after authorization. For missing details, check linked upstream sources, select and record a justified pilot candidate where appropriate, and test its consequences. Escalate material scope/scientific choices when necessary; do not block unrelated work on missing settings for later phases.
- Preserve the distinction between synthetic prototype, real small-scale pilot, paper-scale reproduction and strict causal audit. Record reduced dimensions, event subsets and missing ECHO as scope differences.
- Refresh current files before reporting progress. Code written, an artifact generated, a local test passed and a server job completed are different claims.

## Feature extraction rules

- Follow the paper's ECG patch-mean extraction and ECHO preprocessing/CLS extraction; record the remaining unspecified settings separately.
- Freeze external encoders and train NOAH projections. Pin checkpoint, processor/tokenizer, output selection and preprocessing revisions. Keep original content-space dimensions: text/CXR 768, ECG 192, ECHO 512.
- Verify long-text length/pooling/normalization. The public BioLORD sentence settings are not automatically the exact NOAH settings.
- Verify the resized CXR copy and its transformations before applying another processor. Do not double-rescale images.
- NOAH's OTIS patch is 1x24. Upstream linear-probe examples use 1x100 by default. Do not silently drop/reinitialize patch weights to make a checkpoint load; do not substitute classification/CLS outputs for final patch means.
- EchoPrime's video output is 512-D. Its study interface adds view information and can be 523-D. Do not use that interface as NOAH's video embedding. Preserve NOAH's DICOM metadata-region step; the upstream pixel-mask helper is not the same operation. Record clip start, pixel range, normalization and padding stage.
- Numeric 22-scale Fourier features and temporal base-10000 features are distinct. The current 44-D numeric convention is a candidate, not a fully specified paper formula.
- Version embedding caches by content and the entire extraction recipe. Do not silently replace encoding failures or missing modalities with zero targets.

## Model, training and information boundaries

- Preserve causal content attention, asymmetric temporal Q/K features, target alignment and explicit gap units/boundaries. Do not replace them with standard RoPE or a symmetric time encoding.
- Prior reads historical h_(t-1). Posterior additionally reads target e_t during training. Use a learned conditional diagonal prior, one shared latent per event/decoder pass, hierarchical feature decoding and the paper's hard gates.
- Keep posterior reconstruction, prior reconstruction and KL. KL and LR warmups have separate clocks. Record unspecified likelihoods/reductions and optimizer settings rather than calling library defaults author settings.
- Ownership is defined for shifted targets. Prefix/overlap provide context but do not duplicate losses; all loss branches obey masks. Unknown final gaps differ from real zero gaps.
- Free rollout uses history and prior only. Oracle time control is a distinct mode. Generated numeric/content embeddings remain in their input representation spaces; no raw-media generator is implied.
- Use independent mathematical/reference tests and adversarial information-boundary tests, not only shape tests or a declining total loss. Follow relevant IDs in the test plan.

## Practical project context

O2 GPU access is available. Record actual GPU allocation, memory, throughput and queue conditions when running. The paper's 8x A100 80GB is a scale reference, not the pilot's minimum requirement. SLURM wall-time requests are not measured training durations.

## Repository and documentation hygiene

Use dollar-delimited math in all Markdown: `$...$` for inline formulas and `$$` on separate lines around display formulas. This convention is required for VS Code Markdown preview; do not use backslash-based math delimiters.

All versioned repository content intended for collaborators must be in English: documentation, reports, reusable agent prompts, code comments/docstrings, test descriptions, user-facing messages and commit/PR descriptions. Use English filenames. Conversation with the researcher may be in Chinese. Preserve technical identifiers, source citations and original data values; do not translate clinical data to satisfy a repository language check. Keep private Chinese explanations outside versioned content.

Follow [REPOSITORY_POLICY.md](docs/REPOSITORY_POLICY.md) before committing or pushing. Keep presentation tooling, renders, intermediate outputs, clinical data, embeddings and checkpoints in ignored locations. Stage explicit reviewed paths and inspect staged content and outgoing commits. Do not force-add ignored artifacts. Ignore rules do not untrack existing files.

Write documentation about the current state, technical requirements and next actions. Update it in place; Git retains history. Omit reactions to previous reports, historical correction sections and conversational task narration. Retain source provenance, scientific assumptions and dated execution evidence.

Evaluation and reporting must use the experiment manifest and release contracts in `src/evaluation/contracts.py`. Landmark extraction must reject future events, unavailable prospective evidence and full-record pooling. Do not relabel retrospective, posterior-reconstruction or oracle-time metrics as prospective. Preserve architecture-specific checkpoints/results before modifications, and obtain explicit approval for unresolved clinical methodology. The alpha-sweep remains on hold until its documented dependencies and authorization are satisfied.

Update source traceability, chosen assumptions and execution status when changing behavior. Keep documentation-only tasks documentation-only. Preserve existing user work and do not claim results for work that has not run.

## Clean development worktree

Use `python -I tools/run.py test` from the recorded interpreter. Runtime imports use `noah_baseline`, never `src` or legacy launchers. No embedding caches or clinical data are inherited. Read `docs/MIGRATION_BASELINE.md` and the current `docs/PROJECT_STATUS.md` first. Canonical paper requirements remain in the existing specification, assumptions and traceability documents. Migrated model code is a declared candidate, not exact paper fidelity. The named M1 categorical/numeric24h mortality integration is authorized through the bounded `m1` command; excluded phases and alpha remain gated. Read `configs/m1.json` and the current M1 implementation/status before execution.
