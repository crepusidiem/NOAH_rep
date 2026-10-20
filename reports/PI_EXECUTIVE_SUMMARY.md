# NOAH reproduction: current research evidence

Checked: 2026-10-19. Primary source: [NOAH.pdf](../paper/NOAH.pdf), especially Methods pp.15–25 and Supplementary Table16 p.50. Current code/evidence: [architecture audit](../docs/ARCHITECTURE_AUDIT.md) and [project status](../docs/PROJECT_STATUS.md).

The repository contains pilot ETL, model, training and evaluation code, and reports of completed small-cohort experiments. A scientifically faithful reproduction remains incomplete. Expanding the cohort is conditional on resolving current encoder, data and architecture gates.

The paper reports299,712 patients,559M events, six modalities,12 transformer layers, width768, latent384,150,322,970 trainable parameters, context2048 and overlap256. A50,000-patient cohort would be an intermediate scale, not the paper cohort. The corrected architecture candidate has148,957,844 parameters; the exact author block/head layout is unresolved.

Existing aggregate reports describe a CXR ablation, architecture scaling and demographic/class-weight experiments. Their metrics belong to those implementations and evaluation protocols. They do not establish prospective clinical prediction, complete paper evaluation, or validated frozen ECG representations. The inspected ECG implementation constructed OTIS without loading pretrained weights; dependent caches and results need provenance review and corrected extraction. Statistical or causal explanations of pilot metric differences remain hypotheses.

Current source fixes cover input temporal encoding, linear specifics projection, latent-width hierarchical decoding, stochastic depth, ECHO loss, all three specified class-weighted heads, optimizer routing/schedules, note-content attachment, strict missing-target checks and prior-only rollout with correct outgoing clocks and modality support. Old checkpoints require explicit architecture compatibility handling. Optional forced-time/top-p rollout is a deviation; it does not demonstrate learned time forecasting.

Local CPU verification produced39 passing tests and one skipped real-shard test. A separate synthetic training/validation/rollout/checkpoint smoke run completed. No corrected real-data retraining, new GPU allocation or paper-scale experiment was executed in this audit.

Next work: verified pretrained1x24 OTIS assets and regenerated representations; pinned text/CXR cache recipes; ICU vital aggregation and source coverage; ECHO video processing at the supplied1.0.1 location; full-record/cutoff/resume tests; then corrected small real-data training and separately defined evaluation. Existing resource measurements are run-specific and cannot guarantee memory or throughput after these changes.
