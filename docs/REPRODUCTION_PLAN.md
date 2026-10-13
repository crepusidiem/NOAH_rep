# NOAH dependency-aware reproduction plan

Phase 0 audit, 2026-10-13. [Specification](REPRODUCTION_SPEC.md), [traceability](PAPER_TRACEABILITY.md), [decision register](ASSUMPTIONS.md), and [test plan](TEST_PLAN.md) are the deliverables of this phase.

**Stop condition:** this phase writes documentation only. It does not implement the model, data pipeline, tests, training, or evaluation. The next phase requires explicit user approval.

## 1. Reproduction levels and dependency graph

Maintain distinct labels:

- **Synthetic prototype:** verifies mathematical, temporal, latent, hierarchy, masking, and rollout behavior on known data. No claim of MIMIC or paper-result replication.
- **Real small-scale experiment:** actual frozen encoders and a patient-disjoint subset, with named reduced dimensions/schedules where necessary. Tests assumptions and training behavior, not published metric equivalence.
- **Paper-scale reproduction candidate:** full reported architecture/corpus/evaluation, with exact evidence for missing settings or prominently declared unresolved choices. The word “exact” is withheld if core settings remain assumed.
- **Strict causal audit experiment:** an independently named alternative for as-of metadata, aggregation availability, and future-gap handling. Never merge these results with literal paper-profile numbers.

The schedule follows dependencies, not calendar deadlines:

~~~mermaid
flowchart TD
    P0["Phase 0: audit documents"] --> G0["Explicit implementation approval"]
    G0 --> D["Freeze decision and evidence manifests"]
    D --> M["Math, tensor, temporal and loss oracles"]
    D --> A["Data access, source mapping, frozen assets"]
    M --> P["Synthetic model and rollout prototype"]
    A --> Q["Event/encoder/window data QA"]
    P --> G1["Synthetic correctness gate"]
    G1 --> S["Real small-scale training"]
    Q --> S
    S --> X["Assumption sensitivity and inference audit"]
    X --> G2["Freeze paper-scale protocol and resource estimate"]
    G2 --> F["Full corpus preprocessing and pretraining"]
    F --> E["Frozen-model scientific evaluations"]
    E --> R["Table/figure comparison and discrepancy report"]
~~~

Data-access/encoder work and mathematical specification can proceed independently after approval; no sub-agent work is assumed or dispatched. Full pretraining depends on both.

## 2. Schedule and completion gates

Effort ranges below are planning estimates for one engineer, not paper training durations or guaranteed delivery dates. Data approval, checkpoint access, author's clarification, storage, and available GPUs can dominate elapsed time. Missing total epochs/batch size prevent a defensible paper-scale training ETA.

| Stage | Dependencies | Work after approval | Planning estimate | Completion evidence |
|---|---|---|---|---|
| 0: audit | Supplied paper | Read all 50 pages, inventory evidence/unknowns, write five documents | Completed in this audit | Five coherent documents, source hash, full appendix ledger; no model implementation |
| 1: decision freeze | Approval | Resolve/select critical A-IDs; archive evidence; define literal/causal/tiny configs and evaluation estimands | 2-5 engineer-days, plus external waiting | Every core knob has classification, rationale, manifest and sensitivity plan |
| 2A: independent oracles | Stage1 | Write temporal/numeric/Gaussian/KL/masking/metric reference tests before modules | 2-4 days | Analytical and adversarial oracles pass against hand calculations |
| 2B: data/asset inventory | Stage1 | Verify access to versioned MIMIC sources and frozen assets; source-row maps, preprocessing hashes, staged count ledger | 3-7 days for inventory/sample QA; full preprocessing unestimated | Real sample transforms/dimensions/provenance audited |
| 3: synthetic prototype | 2A | Implement paper-specific tensor graph and losses; scripted outgoing-gap rollout; known-rule synthetic generator | 4-8 days | Unit/integration gates, known temporal/joint behavior, prior-only replay, no hidden future input |
| 4: small real-data training | Stage3 gate +2B QA | Actual encoder caches, fixed patient subset, patient split, optimizer/schedule, diagnostic training, frozen-state evaluation | 5-10 days + measured compute | Stable prior/posterior training, real-data provenance, information-boundary tests and sensitivity report |
| 5: scale decision | Stage4 | Freeze unrevised scientific protocol, full-architecture count ledger, corpus QA, benchmark throughput/storage, estimate compute | 2-4 days | Approved engineering config; documented remaining fidelity gaps; measured resource plan |
| 6: paper-scale run | Stage5 +full data QA | Full data processing/caches and full-size dual-path pretraining with resumable checkpoints | Derive from actual throughput and chosen/confirmed duration | Reproducible manifests, completed objective/schedule logs, validation-selected frozen checkpoint |
| 7: scientific evaluations | Frozen checkpoint; endpoint/metric oracles | Forecasting, zero-shot, interventions, probes, NEWS2, survival, surprise/explainability; all figures/tables covered | 5-10 engineer-days + rollout/HPO compute | Protocol-matched estimates/CIs, coverage/calibration, comparison ledger and unresolved discrepancies |
| 8: reproduction report | Stage7 | Separate reproduced facts, assumed mechanisms, deviations, negative results and non-executed analyses | 2-3 days | Complete report with assets/config hashes and claim boundaries |

Stages2A/2B may overlap. Stage3 cannot wait for licensed clinical data because synthetic fixtures suffice; stage4 cannot begin without actual frozen encoders/source provenance. Stage6 is not entered merely because a toy run succeeds.

## 3. Stage 1: turn audit uncertainty into reviewable configurations

Critical decisions before any model prototype:

1. A-10 numeric basis, A-12 temporal units, A-13 input-time fusion, A-14 outgoing/incoming gap alignment and seed policies.
2. A-11 event shift/h0/posterior input stage; A-15/16 attention scaling/layers; A-17..20 backbone/MLPs/log-sigma squash.
3. A-23 value routing and continuous feature cascade; A-25/26 gates/gap transform; A-27..30 objective likelihood/scaling/masks/anneal clock.
4. A-06 context/prefix accounting; A-35 representation cutoff gap; A-37 cache policy; A-61 boundary time labels.

Proposed reviewable artifacts after approval: a decision manifest; symbol-to-tensor diagram; event/target index examples; module parameter budget; independently configurable run profiles. Configurations must fail fast on unresolved critical values rather than instantiate library defaults.

A coherent provisional implementation can follow the outgoing-gap scheduling candidate, but must preserve its ASSUMED status until corroborated. Choose full bounded recomputation as a test reference before attempting caching. Do not add raw image/text generators.

## 4. Synthetic-data prototype

Order of implementation after test oracles:

1. Event schema, exact target ownership and six-field prefix/windowing.
2. Distinct numeric-dyadic and temporal-geometric bases.
3. Asymmetric causal attention with selected layer range and approved block layout.
4. Shifted prior/posterior, smooth prior log-sigma, diagonal KL and shared reparameterized latents.
5. Cascading category/specifics/modality/value heads, hard presence/time gates, compatibility mask.
6. Dual reconstruction objective, component masks, KL schedule, optimizer partition/scheduler.
7. Prior-only decode/re-encode/append with an explicit clock and generation validity states.
8. Toy forecasting/endpoint/probe metrics.

The synthetic generator must expose:

- Horizon-dependent retrieval of history, including equal-time events and very long gaps.
- Predictable patterns plus independent innovations with a known distribution.
- Correlated categories/modalities/specifics/values, including illegal pairings.
- Missing versus zero values, procedure ends, metadata changes, overlap/tails, and long records.
- Known clinical target clocks, unscorable rollouts, and competing events.

Reduced dimensions must still satisfy \(2F\mid d_h\). A proposed tiny profile is d=128, H=4 (head width 32), two layers, k=64, nominal FFN ratio 8, shorter explicit context: **DEVIATION for prototype scope**, not paper settings. All module formulas, gates, and dual loss paths remain exercised. A full-size model-construction test separately verifies the reported dimensions; parameter count remains an evidence gate.

Exit: U/I suites relevant to prototype plus S-01..04/06/09 pass; literal versus strict information tests demonstrate the expected distinction. Report posterior reconstruction, prior reconstruction, unweighted and weighted KL, sigma quantiles/bounds, zero-gap rates, class compatibility, elapsed forecast horizon, and invalid ratios. Low posterior loss alone is insufficient.

## 5. Real-data preparation and small-scale training

Choose and freeze a subject-disjoint subset before outcomes are inspected. Suggested engineering range is 1,000-5,000 patients as an **ASSUMED planning choice**; ensure nonempty CXR/ECG/ECHO cases, or disclose enriched sampling and separately evaluate naturally distributed patients. Patient metadata and label derivation remain grounded in source rows.

Required preprocessing outputs:

- Source versions/checksums, subject links, event category/modality dictionaries, unit/time conventions, and original/available-at timestamps.
- Raw -> joined -> eligible -> encoder-success -> aggregate -> windowed -> scored counts, with reasons for each exclusion.
- Frozen asset checksums, real encoder transformations/output selection, deduplicated text/specifics cache where possible.
- Patient split IDs, 64-event inclusion rule, window ownership proof, missing/terminal gap masks.
- Modality/type statistics, within-stay density/gaps/composition compared with ED Fig.5, and train/val/test cohort summaries.

Run a one-batch gradient audit before an extended job. Then a tiny real-batch overfit, followed by a predeclared small training budget. Use full reported d/layers where resources permit; reduced architecture/schedule gets a separate experiment name. Keep posterior/prior sampling and \(.3,.1,\beta\) weights; do not simplify to a deterministic autoencoder to obtain a working run.

Sensitivity priority:

1. Literal window-end metadata versus as-of alternative.
2. Actual event-time aggregation versus available-at cutoff handling.
3. Temporal units/input forward-feature choices and probe last-gap sanitization.
4. Outgoing-gap alignment and initial seed/horizon sensitivity.
5. Continuous likelihood/dimension reduction; numeric phase/scalar transform.
6. Decoder training routing/class-weight/support scope; block/MLP choices affecting count.

Holding cohort, random seed, token budget, and metric implementation fixed is essential for meaningful comparisons. No test-set-informed architecture or checkpoint tuning. Small-scale science diagnoses behavior; it cannot be judged by matching paper-scale AUROC.

## 6. Paper-scale readiness and compute planning

Readiness is conditional on legitimate data access, exact asset availability, and resolving or declaring missing training settings. The paper says code will be provided upon publication; this audit did not inspect author code and makes no claim about its current availability.

Target corpus: ~559M events, ~391M training, ~84M validation, ~83M test; 299,712 patients. Author hardware: eight A100 80 GB GPUs. Paper-scale configuration target: 150,322,970 parameters, 12 layers, hidden width 768, 12 heads, latent width 384, context 2048, 22 numeric dyadic scales, reported regularizers/optimizers/schedules.

Before provisioning/starting a long job, measure:

- Real encoder/cache throughput by modality, failures/retries, and required raw-data storage.
- Full-size training tokens/s with *both* decoder passes, memory at actual effective batch, and distributed overhead.
- Validation cost, checkpoint storage/load/resume, and rollout cost at 128/1024-token budgets.
- DeepSurv TPE cost (100 trials per tuning setting; six settings with outer-fold evaluation) separately from NOAH pretraining.

Engineering estimates must use measured quantities:

\[
\text{pretraining wall time}\approx
\frac{\text{epochs}\times\text{attended training token positions per epoch}}
{\text{measured aggregate token positions/s}}
+\text{validation/checkpoint overhead}.
\]

With an assumed total budget of 2048 tokens, six prefix tokens, 256 clinical tokens of overlap, and stride 1786 clinical events, interior attended positions are approximately \(2048/1786\approx1.147\) times newly scored events. This estimate ignores first/tail windows and filtering; exact counts come from the window manifest. Report scored tokens and attended tokens separately.

A cache-size estimate can sum \(\sum_m n_mD_m\cdot\text{bytes per scalar}\), plus separately cached specifics, IDs, and timestamps. Under a *candidate* 44-dimensional numeric bank and fp16 storage, per-event text/value caches without deduplication are roughly 0.34 TB for Table 15's whole corpus, excluding specifics/raw data/indexing. This is an engineering estimate, not a paper storage claim. Deduplicated repeated text/specifics can materially reduce it; storage precision is an assumption needing numeric validation.

Total epochs/global batch/accumulation/cosine floor/Muon internals are not stated. Obtain corroboration or preregister explicit assumptions and compute ceilings. The audit cannot honestly supply author GPU-hours, wall time, or training cost.

Exit gate before a paper-scale fidelity claim: no silent critical defaults; exact configuration and dataset manifests; parameter-count discrepancies explained by modules; full-size numerical/causal tests; recorded information-profile differences. A count mismatch is not repaired by adding unused parameters.

## 7. Frozen-model evaluation schedule

Freeze a checkpoint using the predeclared validation criterion. Extract h under separately named representation-time policies. For cost control, develop metric/endpoint code first on fixtures and the small subset, then freeze it before test evaluation.

| Evaluation | Dependencies and required output |
|---|---|
| Surprise/state geometry | Frozen h plus posterior/prior on observed events; unweighted KL by 31 types, arclength/cosine lags, controlled projection settings; Fig.2/S Table1 |
| Forecasting | Cohort/anchor manifest, clock alignment and oracle schedule; 13,551 patients if reproduced, M=50, eight horizons, fixed budget; macro AUROC/Brier/count MAE/ECE, baselines, 1,000 patient bootstraps; Fig.3a-i/ED Table1/S Tables2-3 |
| Zero-shot outcomes | Target dictionaries, label orientation, initial gap seeds and validity logic; LOS/mortality 2,000 patients/M=50/128 tokens; readmission 1,000/M=25/1024 tokens, tau=1; coverage and calibration; ED Table2/ED Fig.1 |
| Linear probes | Frozen 768-dimensional h, patient/fold labels, per-fold scaling; 15 ICD chapters/29 comorbidities/outcomes, seven positions, five folds x three seeds, subgroup tables; Fig.5/ED Fig.2/S Tables7-10 |
| NEWS2 | Validated label windows/scorer, separate 70/15/15 patient split within NOAH test set; 512-256 MLP, three seeds, holdout/bootstrap and baselines; Fig.5/ED Fig.3/S Tables11-12 |
| Survival | Landmark/risk-set/censor policy, nested CV/HPO; 26,476 patients, six fractions, TPE 100 trials, D-cal 20 bins; td-CI, IBS and D-cal p-values; Fig.5/S Tables13-14 |
| Intervention simulations | Sepsis/fluid maps and numeric inverse for renal outcome; 1,085 patients, M=25, shared noise, 1 h seed, arm constraints; factual calibration, paired effects/coverage and both mortality denominators; Fig.4/ED Table3/S Tables4-6 |
| Attention/attribution/history | Head relation groups/lift/IG baseline; four-term ablation, history windows and age >=75, M=25; Fig.6/ED Fig.4 |

NEWS2 training precedes applying the probe to counterfactual generated states. Renal scalar recovery precedes MAKE-30; event-embedding rollouts themselves do not require scalar inversion. Survival tuning and linear/NEWS2 probe fitting are separate training tasks; “frozen NOAH” does not mean the entire evaluation is zero-shot.

## 8. Discrepancy triage and final deliverables

If results differ:

1. Check data eligibility, split, mappings, timestamps/units, frozen checkpoint/preprocessing, and coverage.
2. Check input/target shift, forward gap origin, prefix snapshot, representation cutoff and endpoint denominators.
3. Check likelihood/reduction/class weights, prior/posterior gap, optimizer groups/schedules and numerical precision.
4. Check scientifically declared assumptions and compare matched-budget sensitivities.
5. Only then inspect stochastic/statistical differences. Do not tune against test benchmarks to force agreement.

Archive final code/configs/tests only in later approved phases, with environment/data/encoder/run hashes, resume evidence, losses and diagnostics, evaluation predictions/coverage, bootstrap/CV outputs, and figure/table comparison ledger. Respect source data access constraints; reproducibility manifests must not imply redistribution of MIMIC.

The final report must state: what was reproduced exactly; what was reconstructed with assumptions; what was intentionally changed; what remains unresolved/unexecuted; and how much evidence supports each numerical or scientific claim.

Phase 0 document checks completed: the source SHA-256 was reconfirmed; all 61 requirement IDs have traceability rows; all 65 A-IDs have decision records; the test plan contains 103 unit, 10 integration, and 16 scientific experiment IDs without duplicates; local document links and Markdown table column counts were checked. These checks validate document consistency, not the model or any scientific result. Temporary PDF extraction/rendering artifacts were removed, and the supplied paper was preserved.

**Wait for explicit approval before beginning any implementation stage.**
