# NOAH project status and next actions

Updated: 2026-10-19 (America/New_York). Inspected baseline: `1d21fd6f85d4fabbf6cdf0022e865db82cc52dc7` plus preserved pre-existing changes and the current architecture audit edits. Evidence: local source, supplied PDF, existing aggregate reports and CPU tests. No new Slurm jobs or clinical-data retraining were launched.

## Current position

The project has a real-data pilot implementation and reports of completed 1,000-candidate experiments and a 12-layer architecture-scaling experiment. It is **not yet a verified faithful reproduction**. Current priority is controlled validity remediation: preserve revisions, approve clinical information/endpoint contracts, verify provenance-complete views and gate reporting before any alpha-sweep or cohort expansion. The [architecture audit](ARCHITECTURE_AUDIT.md) maps every requirement family to code, fixes, assumptions and remaining gaps; the [full specification](REPRODUCTION_SPEC.md) and [paper traceability](PAPER_TRACEABILITY.md) describe the target.

| Component | Current evidence and limitations |
|---|---|
| Architectural specification | Existing specification/traceability substantially match paper Methods and Table16; architecture is not completely determined by the publication. |
| Cohort | Existing manifest contains 1,000 candidates, assigned700/150/150; enriched admission/ECG/CXR intersection differs from natural paper cohort. |
| Event extraction | DuckDB readers exist across multiple sources. Complete31-category/lifetime coverage is not established; ICU `chartevents` and sliding vital aggregation are not connected. CSV parsing now fails on malformed rows instead of silently skipping. |
| Frozen content | Text/CXR paths implemented, but full pinned cache recipes remain incomplete. Actual text content is now attached separately from specifics; missing declared targets fail curation. |
| ECG | Earlier cache constructed OTIS without loading weights. Current extraction requires strict compatible pretrained weights and recipe-hashed cache namespace. A suitable checkpoint and existing cache provenance remain unverified; dependent runs need revalidation. |
| ECHO | Supplied1.0.1 directory exists at `/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iv-echo/1.0.1/`. Paper uses1.0. Event/video preprocessing and encoder verification remain unimplemented/unexecuted. |
| Model | Input temporal encoding, linear specifics, latent-width cascade, stochastic depth, ECHO loss and mask intersection implemented. Learned prior/posterior, directional causal attention and dual objective retained. |
| Paper dimensions | Width768, layers12, heads12, FFN8, latent384 configured; current candidate148,957,844 parameters differs from paper150,322,970. Config context2048 requires new shards; five-modality scope remains declared. |
| Optimization | Trainer routes hidden matrices to Muon and embeddings/heads/non2D to AdamW; fits all three specified class weights; honors YAML schedules/clip, separate update/epoch clocks and partial accumulation groups. Internals/reductions remain declared candidates. |
| Rollout | Full supplied history, prior-only shared latent with temperature, observed modality support, outgoing-gap alignment and generated embedding reinjection. Default preserves zero gaps; anti-stall/top-p are optional labeled deviations. |
| Preservation | Fifty repository checkpoints and one synthetic checkpoint independently copied, rehashed and inspected; historical152,655,380, pilot5,435,924 and current synthetic179,108 layouts recorded separately in the private snapshot. Generating-source/metric replay lineage remains unverified. |
| Evaluation guards | New explicit encounter/landmark primitives, manifest/artifact/report gates and adversarial tests; legacy pooling/subject-max/NEWS2 reporting paths and alpha launcher held. Real-data view/label/probe integration and methodology approval remain pending. |
| Local validation | Prior architecture phase:39 passed,1 skipped plus a synthetic trainer smoke run. Controlled remediation:61 passed in7.86s, with no skips, covering contracts and refreshed core references; no corrected real-data evaluation/training completed. |
| Checkpoint recovery | Optimizer/support/weights/random states saved; strict model loading prevents silent partial migration. Exact resume/data-order recovery remains unverified. Old architecture checkpoints require explicit compatibility handling. |
| Prior clinical metrics | Existing reports contain pilot/architecture-scaling/weighted-head results. They are measurements of their recorded implementations, not validation of corrected architecture or exact paper protocols. No jobs/checkpoints were independently re-executed here. |

## Next actions and acceptance gates

1. Review and approve applicable M01–M12 decisions in the [controlled G01–G15 remediation matrix](ARCHITECTURE_AUDIT.md#controlled-remediation-preserved-evidence-and-revision-separation). Recommendations are not active scientific defaults.
2. Implement approved source availability/encounter/contributor provenance and cutoff-specific evaluation views, with future-information invariance and patient split/ownership tests. Old shards alone cannot certify a prospective cutoff.
3. Connect explicit encounter labels and single-landmark states to frozen compatible revisions; validate actual predictions and metric replay. Keep paper reproduction and strict prospective modes separate.
4. Validate complete NEWS2 local-window labels/probe independently; historical triage summaries are not clinical NEWS2 outcomes.
5. Verify encoder assets/recipes, source coverage, loss/reduction/weighting scope and effective configuration/recovery before corrected training comparisons. No model, trainer or numeric YAML settings changed in the guard phase.
6. Alpha-sweep remains HOLD: its launcher exits78 before work. Fix alpha=0 propagation and freeze the approved weighting axis, one revision/data/label/probe protocol and manifest before any separately authorized launch.

The earlier N-01–N-09 execution reports retain their own dated evidence. They do not close the gates above. See [audit execution report](../reports/server/2026-10-19_architecture_audit/REPORT.md) for commands, environment, scope and test results.

Controlled-remediation evidence and the snapshot inventory are in [the execution report](../reports/server/2026-10-19_controlled_remediation/REPORT.md). No corrected clinical scores or newly trained checkpoints were generated. Historical metric eligibility is classified in the remediation matrix; prospective interpretations are invalid.
