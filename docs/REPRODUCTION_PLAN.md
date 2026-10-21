# NOAH dependency-aware reproduction plan

Clean-worktree context: operational paths and results below describe the archived implementation unless mapped by [MIGRATION_BASELINE.md](MIGRATION_BASELINE.md). Active imports use `noah_baseline`; current stage and supported commands are in [PROJECT_STATUS.md](PROJECT_STATUS.md). Scientific requirements and paper citations are retained unchanged.

Current code verification and selected implementation candidates (2026-10-19): [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md). Paper requirements below remain distinct from implementation/test status.

Updated 2026-10-14. This is the action plan for the current pilot and the later paper-scale reproduction. See [current progress](PROJECT_STATUS.md), [operational implementation guide](IMPLEMENTATION_GUIDE.md), [specification](REPRODUCTION_SPEC.md), [traceability](PAPER_TRACEABILITY.md), [decision register](ASSUMPTIONS.md), and [test plan](TEST_PLAN.md).

Pilot entry: N-01 server inventory, N-02 minimal configuration and N-03 encoder compatibility. Implementation and execution follow the researcher’s active task; paper-scale training is a separate scope and resource decision.

## 0. Current milestone

B0, M1 and the single approved M1.5 run are accepted/frozen. The active scope is the revised M2: scalable cohort/source inventory, all six modality pipelines, unified provenance-preserving events/windows and statistical readiness. See [current status](PROJECT_STATUS.md), [migration milestones](MIGRATION_BASELINE.md) and the [executed integration report](../reports/server/2026-10-20_m2_integration/REPORT.md). The N-01–07 dependency details below remain reference requirements, not an instruction to restart completed phases.

M2-A fixes15,000 assigned patients (10,500/2,250/2,250); B has actual encoder evidence for all six modalities, with unresolved preprocessing/mappings labeled PARTIALLY VERIFIED. A bounded C chain of208 events/7 patients/10 windows passes target, cache, replay, suffix and model-routing checks. Full canonical counts/≥64-target eligibility and production coverage are not yet established. The existing model/loss candidate and M1/M1.5 evidence remain unchanged.

Next dependency order:

1. C1 source indexing is accepted through job57044273 plus a separate CXR post-validation attestation, under the authorized2 CPUs/8GiB/60min/8GiB output and256MB validation budget. The original job remains FAILED/INCOMPLETE; no further C1 submission is needed. Preserve job57006471 and its accepted indexes; no automatic retry, resource increase or extra job. Actual terminal evidence belongs in project status and its execution report.
2. Reuse the exercised six-modality implementations for a provisional occurrence-time engineering dataset. Keep source-backed longitudinal events even when encounter attribution is unknown. Record unresolved material exceptions in [OPEN_SCIENTIFIC_ISSUES.md](OPEN_SCIENTIFIC_ISSUES.md); do not repeat broad encoder feasibility studies or require all future clinical decisions before engineering work.
3. Prepare canonical mappings, numeric representations, content task queues and bounded manifest accounting. Full-cohort preprocessing and bulk encoding each require a concrete bounded resource proposal before execution. Reuse compatible verified caches only with source/content/checkpoint/recipe evidence.
4. Materialize the fixed15,000-patient event/cache/window dataset, preserving all assigned members and reporting actual canonical, encoded and owned-target populations. No density resampling;10,500 assigned training members are not10,500 qualified sequences. Independent ownership, split, provenance and artifact checks govern promotion.
5. Propose one meaningful multimodal learning experiment after usable dataset evidence. Training, strict prospective evaluation, paper-fidelity claims and the alpha sweep remain separately gated; C1 success leaves full-M2 readiness HOLD.

## 1. Reproduction levels and dependency graph

Maintain distinct labels:

- **Synthetic prototype:** verifies mathematical, temporal, latent, hierarchy, masking, and rollout behavior on known data. No claim of MIMIC or paper-result replication.
- **Real small-scale experiment:** actual frozen encoders and a patient-disjoint subset, with named reduced dimensions/schedules where necessary. Tests assumptions and training behavior, not published metric equivalence.
- **Paper-scale reproduction candidate:** full reported architecture/corpus/evaluation, with exact evidence for missing settings or prominently declared unresolved choices. The word “exact” is withheld if core settings remain assumed.
- **Strict causal audit experiment:** an independently named alternative for as-of metadata, aggregation availability, and future-gap handling. Never merge these results with literal paper-profile numbers.

Dependencies determine readiness. The accepted 4-6 week planning window in section 2 is conditional, not a fixed completion date:

~~~mermaid
flowchart TD
    P0["Audit docs + existing cohort/helpers"] --> G0["Agreement on pilot scope and implementation"]
    G0 --> D["Freeze settings needed for the next stage"]
    D --> M["Math, tensor, temporal and loss oracles"]
    D --> A["Data access, source mapping, frozen assets"]
    M --> P["Synthetic model and rollout prototype"]
    A --> Q["Selected-event/encoder/window QA"]
    Q --> Q2["Two-patient real path, then fixed candidate cohort"]
    P --> G1["Synthetic correctness gate"]
    G1 --> S["Real small-scale training"]
    Q2 --> S
    S --> X["Assumption sensitivity and inference audit"]
    X --> G2["Freeze paper-scale protocol and resource estimate"]
    ECHO["ECHO + full event/corpus scope"] --> F["Full corpus preprocessing and pretraining"]
    G2 --> F
    F --> E["Frozen-model scientific evaluations"]
    E --> R["Table/figure comparison and discrepancy report"]
~~~

Data-access/encoder work and mathematical specification can proceed independently within the authorized phase. Full pretraining depends on both.

## 2. Schedule and completion gates

The researcher accepted **4-6 weeks** as a working estimate for one primary engineer with substantial availability and usable data, required assets and GPU time. Count from agreement on scope/start of implementation, not from the first planning conversation. Aim for a first complete run in week 4; weeks 5-6 cover integration/correctness repairs and resource profiling. Access, checkpoint incompatibility, part-time availability or queue delays can extend the calendar. This includes implementation and data integration, not just GPU time. The scripts' 4h/8h wall-time requests are not measured durations.

| Working window | Focus | Milestone evidence |
|---|---|---|
| Week 1 | N-01 server snapshot, N-02 minimal run choices, N-03 encoder compatibility and source mapping | Versioned report, scope/config decisions, readable sample sources/assets |
| Weeks 2-3 | N-04 oracles and N-06 core model; N-05 event/encoding/window work where independent | Synthetic correctness plus two-patient real path, then candidate-cohort samples |
| Week 4 target | N-07 integration and short real-data run | Train/validate/save/resume/prior-only rollout with component logs |
| Weeks 5-6 buffer | Repair observed failures, complete missing acceptance checks and profile resources | Reproducible first run and measured expansion plan |

Detailed effort ranges below remain estimates, not numbers to sum into an unconditional promise. Independent work can overlap. Missing total paper epochs/batch size prevents a defensible paper-scale training ETA.

| Stage | Dependencies | Work after approval | Planning estimate | Completion evidence |
|---|---|---|---|---|
| 0: specification | Supplied paper | Architecture, traceability, assumptions, tests and schedule | Documents available | Coherent requirements, source hash and full appendix coverage |
| 1: pilot decision freeze | Scope agreement | Select the next-stage critical A-IDs; record active modalities/event set and literal/causal/tiny profiles | 2-5 engineer-days, plus external waiting | Required knobs have classification and rationale; future evaluation choices stay open |
| 2A: independent oracles | Required Stage1 choices | Review/reuse existing tests; add temporal/numeric/Gaussian/KL/masking reference tests before affected modules | 2-4 days | Analytical and adversarial results tied to actual code/config |
| 2B: data/asset inventory | Required Stage1 choices | Produce N-01 report, verify source reads and encoder compatibility; source-row maps and staged counts | 3-7 days for inventory/sample QA; full preprocessing unestimated | Real transforms/dimensions/provenance, including OTIS 1x24 and CXR source |
| 3: synthetic prototype | 2A | Implement paper-specific tensor graph and losses; scripted outgoing-gap rollout; known-rule synthetic generator | 4-8 days | Unit/integration gates, known temporal/joint behavior, prior-only replay, no hidden future input |
| 4: small real-data run | Stage3 gate + sample QA | Complete two-patient path, then fixed 1,000-candidate samples; tiny-batch checks and short diagnostic training | 5-10 days + measured compute | Train/validate/save/resume/prior-rollout evidence, component losses and resource profile |
| 5: scale decision | Stage4 | Freeze unrevised scientific protocol, full-architecture count ledger, corpus QA, benchmark throughput/storage, estimate compute | 2-4 days | Approved engineering config; documented remaining fidelity gaps; measured resource plan |
| 6: paper-scale run | Stage5 +full data QA | Full data processing/caches and full-size dual-path pretraining with resumable checkpoints | Derive from actual throughput and chosen/confirmed duration | Reproducible manifests, completed objective/schedule logs, validation-selected frozen checkpoint |
| 7: scientific evaluations | Frozen checkpoint; endpoint/metric oracles | Forecasting, zero-shot, interventions, probes, NEWS2, survival, surprise/explainability; all figures/tables covered | 5-10 engineer-days + rollout/HPO compute | Protocol-matched estimates/CIs, coverage/calibration, comparison ledger and unresolved discrepancies |
| 8: reproduction report | Stage7 | Separate reproduced facts, assumed mechanisms, deviations, negative results and non-executed analyses | 2-3 days | Complete report with assets/config hashes and claim boundaries |

Stages2A/2B may overlap. Stage3 cannot wait for licensed clinical data because synthetic fixtures suffice; stage4 cannot begin without actual frozen encoders/source provenance. Stage6 is not entered merely because a toy run succeeds.

## 3. Stage 1: turn audit uncertainty into reviewable configurations

Resolve or select explicit, configurable candidates for the components currently being implemented. Different modules do not have to wait for unrelated choices. The initial model/data choices include:

1. A-10 numeric basis, A-12 temporal units, A-13 input-time fusion, A-14 outgoing/incoming gap alignment and seed policies.
2. A-11 event shift/h0/posterior input stage; A-15/16 attention scaling/layers; A-17..20 backbone/MLPs/log-sigma squash.
3. A-23 value routing and continuous feature cascade; A-25/26 gates/gap transform; A-27..30 objective likelihood/scaling/masks/anneal clock.
4. A-06 context/prefix accounting and A-61 boundary time labels for window/rollout tests. Resolve A-35 before frozen-state analysis. Start A-37 with bounded recomputation as the reference; optimized cache behavior can follow correctness.

Proposed reviewable artifacts after implementation authorization: a decision manifest; symbol-to-tensor diagram; event/target index examples; module parameter budget; independently configurable run profiles. A runnable profile must supply the critical values for its stage explicitly. Do not hide library defaults, but do not make missing future clinical endpoint/HPO settings prevent data readers or synthetic tests.

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
8. Minimal forecasting/validity diagnostics. Clinical endpoint/probe tooling belongs to the later evaluation stage and need not delay the first functional run.

The synthetic generator must expose:

- Horizon-dependent retrieval of history, including equal-time events and very long gaps.
- Predictable patterns plus independent innovations with a known distribution.
- Correlated categories/modalities/specifics/values, including illegal pairings.
- Missing versus zero values, procedure ends, metadata changes, overlap/tails, and long records.
- Known event/rollout clocks and unscorable/stopping states. Detailed clinical competing-endpoint fixtures can be added when their evaluation is implemented.

Reduced dimensions must still satisfy $2F\mid d_h$. A proposed tiny profile is d=128, H=4 (head width 32), two layers, k=64, nominal FFN ratio 8, shorter explicit context: **DEVIATION for prototype scope**, not paper settings. All module formulas, gates, and dual loss paths remain exercised. A full-size model-construction test separately verifies the reported dimensions; parameter count remains an evidence gate.

Exit: relevant mathematical/unit/integration tests and synthetic temporal/generative correctness experiments pass with a named configuration. Where a scientific test requires clinical labels or a trained real model, defer it and record that dependency rather than treating it as a synthetic-model prerequisite. Literal versus strict information tests demonstrate their expected difference. Report posterior/prior reconstruction, KL, sigma bounds, zero gaps, compatibility and rollout validity. Low posterior loss alone is insufficient.

## 5. Real-data preparation and small-scale training

Use the **existing fixed 1,000-patient candidate manifest** with its 700/150/150 assignments. It is an enriched adult admission+ECG+CXR engineering cohort, not the paper's naturally distributed cohort. Target the proposed five modalities after scope agreement; obtain ECHO in parallel for the later full target, without dummy ECHO vectors. Declare the event categories actually included and any incomplete source coverage.

Apply the 64 scored-event rule only after complete extraction of the selected pilot event scope and correct masks. Report eligible/scorable counts and split counts without implying the candidate count is the retained count. Keep surviving patients in their original split. Do not silently replace exclusions or reshape the test cohort using results; a replacement cohort protocol must be declared before outcomes are inspected. Later scientific evaluation needs a natural-cohort design.

Before processing all candidates, finish an actual two-patient path with real media/text decoding, frozen embeddings, windows and ownership tests. The existing print-only extraction entry is a useful starting reader check, not that completed path. Encoder recipes and cache hashes follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

Required preprocessing outputs:

- Source versions/checksums, subject links, event category/modality dictionaries, unit/time conventions, and original/available-at timestamps.
- Raw -> joined -> eligible -> encoder-success -> aggregate -> windowed -> scored counts, with reasons for each exclusion.
- Frozen asset checksums, real encoder transformations/output selection, deduplicated text/specifics cache where possible.
- Patient split IDs, 64-event inclusion rule, window ownership proof, missing/terminal gap masks.
- Modality/type statistics, within-stay density/gaps/composition compared with ED Fig.5, and train/val/test cohort summaries.

Run a one-batch gradient audit before an extended job. Then a tiny real-batch overfit, followed by a predeclared small training budget. Use full reported d/layers where resources permit; reduced architecture/schedule gets a separate experiment name. Keep posterior/prior sampling and $.3,.1,\beta$ weights; do not simplify to a deterministic autoencoder to obtain a working run.

Sensitivity priority:

These comparisons follow the first complete run. They do not require finishing all clinical evaluations during the 4-6 week functional pilot. The pilot already includes core mathematical and information-boundary correctness checks; select the most consequential matched-budget sensitivities next, and defer endpoint-dependent comparisons until their labels/metrics exist.

1. Literal window-end metadata versus as-of alternative.
2. Actual event-time aggregation versus available-at cutoff handling.
3. Temporal units/input forward-feature choices and probe last-gap sanitization.
4. Outgoing-gap alignment and initial seed/horizon sensitivity.
5. Continuous likelihood/dimension reduction; numeric phase/scalar transform.
6. Decoder training routing/class-weight/support scope; block/MLP choices affecting count.

Holding cohort, random seed, token budget, and metric implementation fixed is essential for meaningful comparisons. No test-set-informed architecture or checkpoint tuning. Small-scale science diagnoses behavior; it cannot be judged by matching paper-scale AUROC.

## 6. Paper-scale readiness and compute planning

Readiness is conditional on data access, usable pinned encoder assets, and resolving or declaring missing training settings. Exact author checkpoint identity is required for an exact-assets claim, not for starting a clearly labeled assumption-based run. Use author code tied to this paper version when available; record asset identity in the run manifest.

Target corpus: ~559M events, ~391M training, ~84M validation, ~83M test; 299,712 patients. Author hardware: eight A100 80 GB GPUs. Paper-scale configuration target: 150,322,970 parameters, 12 layers, hidden width 768, 12 heads, latent width 384, context 2048, 22 numeric dyadic scales, reported regularizers/optimizers/schedules.

Before provisioning/starting a long job, measure:

- Real encoder/cache throughput by modality, failures/retries, and required raw-data storage.
- Full-size training tokens/s with *both* decoder passes, memory at actual effective batch, and distributed overhead.
- Validation cost, checkpoint storage/load/resume, and rollout cost at 128/1024-token budgets.
- DeepSurv TPE cost (100 trials per tuning setting; six settings with outer-fold evaluation) separately from NOAH pretraining.

Engineering estimates must use measured quantities:

$$
\text{pretraining wall time}\approx
\frac{\text{epochs}\times\text{attended training token positions per epoch}}
{\text{measured aggregate token positions/s}}
+\text{validation/checkpoint overhead}.
$$

With an assumed total budget of 2048 tokens, six prefix tokens, 256 clinical tokens of overlap, and stride 1786 clinical events, interior attended positions are approximately $2048/1786\approx1.147$ times newly scored events. This estimate ignores first/tail windows and filtering; exact counts come from the window manifest. Report scored tokens and attended tokens separately.

A cache-size estimate can sum $\sum_m n_mD_m\cdot\text{bytes per scalar}$, plus separately cached specifics, IDs, and timestamps. Under a *candidate* 44-dimensional numeric bank and fp16 storage, per-event text/value caches without deduplication are roughly 0.34 TB for Table 15's whole corpus, excluding specifics/raw data/indexing. This is an engineering estimate, not a paper storage claim. Deduplicated repeated text/specifics can materially reduce it; storage precision is an assumption needing numeric validation.

Total epochs/global batch/accumulation/cosine floor/Muon internals are not stated. Obtain corroboration or preregister explicit assumptions and compute ceilings. The audit cannot honestly supply author GPU-hours, wall time, or training cost.

Exit gate before a paper-scale fidelity claim: no silent critical defaults; full modalities/event coverage; configuration and dataset manifests; parameter-count discrepancies explained by modules; full-size numerical/information-boundary tests; recorded assumptions/profile differences. Exact author assets may remain unavailable; using a declared public candidate is permissible for an assumption-based reproduction, but cannot be described as exact checkpoint fidelity. A count mismatch is not repaired by adding unused parameters.

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

For every subsequent run, store its safe execution report under [reports/](../reports/README.md) and update [PROJECT_STATUS.md](PROJECT_STATUS.md) to the attained evidence level. Remote sync and optional SSH setup are in [REMOTE_WORKFLOW.md](REMOTE_WORKFLOW.md).

## Controlled remediation before alpha experiments

The current dependency order is preservation -> narrow guards -> approved clinical/methodology contracts -> source availability and cutoff views -> encounter labels and landmark state integration -> independent clinical label/metric verification -> frozen comparison recipe -> separately authorized experiments. The detailed G01–G15 matrix, M01–M12 recommendations and historical metric eligibility are in [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md#controlled-remediation-preserved-evidence-and-revision-separation).

The alpha launcher is on an explicit hold and exits before training. In particular, define whether alpha varies category only or all weighted heads, propagate alpha=0 explicitly, freeze loss reductions and one architecture/source/data/probe revision, and pass applicable real-data release checks before submitting an array. No existing historical headline AUROC closes these dependencies.


## Current named M1 milestone

The approved categorical/numeric24h encounter-mortality integration is implemented and executed: all eight acceptance groups passed, with74 clean-suite tests and a four-update real-data demonstration. See `docs/PROJECT_STATUS.md`, `configs/m1.json` and `reports/server/2026-10-19_m1/REPORT.md`. This closes the bounded engineering chain only. It does not authorize substantial pretraining, external modalities, full paper evaluation, prospective claims or alpha execution. Do not repeat a broad audit or treat the earlier B0 approval holds as a request to re-approve the same M1 scope.
