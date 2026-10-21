# Open scientific issues

This is a compact exception register, not a specification or execution ledger. [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md), [ASSUMPTIONS.md](ASSUMPTIONS.md) and [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md) remain authoritative. Classification describes the unresolved issue, not whether an engineering component works. Implemented candidates are distinguished from proposed production choices; neither establishes author settings or PI approval of an alternative. Source indexing is not canonical event construction.

## Questions for the PI — Decision Summary

| Decision | Current provisional choice | Meaningful alternative | Why it matters | Can work proceed while awaiting feedback? |
|---|---|---|---|---|
| **1. Which events belong to a patient versus an encounter?** SCI-01 | Preserve explicit hospital IDs; keep missing/ambiguous attribution unknown and retain patient-level records. | A separate, validated episode/interval attribution with declared ambiguity rules. | Missing-ID labs/media must not acquire a fabricated encounter or disappear from longitudinal learning. | Yes: source indexing and unambiguous patient-level event/embedding work. Only encounter-dependent use of unresolved records waits. |
| **2. How should imprecise clocks enter the sequence?** SCI-02/03 | Keep occurrence and raw storage clocks separate; retain date-only/untimed records outside precise timelines; do not invent midnight. | An explicitly retrospective date/encounter ordering proxy, or a coarse-time representation. | Ordering changes next-event targets and gaps; storage is not necessarily clinical availability. | Yes: precise occurrence-time engineering data and encodings. Do not claim strict prospective validity or silently time unresolved records. |
| **3. Which medication actions and source mappings define the event vocabulary?** SCI-04/05 | Preserve source actions separately; the implemented adapters retain ambiguous prescriptions/actions for reconciliation. | Prescriptions-primary with reconciled pharmacy support, versus distinct order/dispensing/administration events. | Counting two records as two doses or an order as administration changes learned trajectories. | Yes: indexing, content encodings and mappings supported by source semantics. Ambiguous canonical identities remain explicitly deferred. |
| **4. Which numeric/ICU aggregation candidate should be frozen?** SCI-06 | Implemented 44-D Fourier candidate; C2 freezes a 2h span/1h stride as a declared candidate; bulk execution is pending, and this is not an author setting. | Other declared spans/strides and per-item unit-normalization policies. | Changes extrema, event density, eligible patients and class statistics. | Yes: retain source values/units and encode verified compatible values. Aggregated outputs need a versioned candidate before generation; no PI decision is needed for an evidence-backed unit lookup. |
| **5. Should full notes use the tested 128-token candidate or content-preserving chunks?** SCI-08 | Implemented BioLORD 128-token masked mean/L2; all four tested notes truncate. | Chunk and combine all note content with a declared aggregation. | Truncation discards clinical content; chunking changes embeddings and cost. | Yes: cache validation and recipe-specific preparation. Either clearly declared candidate can be proposed for a bounded bulk run; a later change rebuilds note embeddings. |
| **6. Freeze existing media recipes provisionally, or compare a specific alternative first?** SCI-07/09/10 | Pinned public encoders; tested CXR recipe, both separately cached OTIS position variants, and a tested B-mode ECHO recipe. The C2 queue recipe selects existing candidates explicitly; bulk GPU execution remains unapproved. | OTIS analytic versus upstream interpolation; a broader validated ECHO mask/clip policy; exact author settings if obtainable. | The two OTIS choices produce materially different embeddings; unsupported ECHO content must be reported, not silently replaced. | Yes: verified existing components/caches and source alignment work. Routine identity checks continue; only incompatible or unsupported inputs wait. Bulk recipes must be declared in the proposed job. |
| **7. Which interpretation is intended for the next training result?** SCI-11/12/13 | Existing reduced candidate and declared loss, teacher-forced time-conditioned prediction; no new training authorized. | A separately configured free-forecasting or paper-dimension experiment. | True next-gap timing, unresolved architecture totals and loss scaling prevent paper-exact or unconditional claims. | Yes: event/embedding preparation. Freeze one architecture/loss/information condition in the separately authorized training protocol; do not restart encoder feasibility. |
| **8. Which population will later clinical claims represent?** SCI-19/20 | Keep all 15,000 fixed members and natural modality absence. Distinguish assigned, canonical, fully encoded/owned-target eligible, and landmark-eligible populations. | Separately reported landmark-defined and full-trajectory-density-conditioned estimands; modality-stratified analyses without complete-case resampling. | Future record density can select on survival/follow-up;48,130 ECHO records represent only 358 patients. | Yes: C2 preserves the cohort and reports actual supports. Do not use future full-record eligibility to define an early prediction risk set or infer population-wide ECHO performance. |

Nonblocking uncertainty is documented rather than converted into a blanket stop. A dataset may be an explicitly provisional occurrence-time engineering dataset while prospective or paper-fidelity claims remain unavailable. The freeze priorities below identify decisions to record before declaring an affected artifact final; they do not require stopping unrelated work. Routine checks of hashes, deterministic joins and authoritative dictionary fields are engineering work, not PI questions.

## Detailed exception register

### SCI-01 — Longitudinal records and uncertain encounter/media attribution

**Category:** DATA_LIMITATION. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** C1 measured 4,460,269/9,781,921 lab rows without `hadm_id`; indexed media frequently identify patient/study rather than hospital encounter. [C1 evidence](../reports/server/2026-10-20_m2_c1/REPORT.md); A-01/02/09.

**What remains uncertain or different:** Which patient-associated observations can validly be assigned to a hospital account/episode, and how ambiguous linkage affects encounter-specific analyses. File existence is not proof of clinical encounter attribution or event timing.

**Example:** A genuine outpatient lab between admissions belongs to the patient but not automatically to either hospital stay.

**Current provisional handling:** IMPLEMENTED: raw missing IDs remain unknown; explicit source IDs are preserved. Bounded adapters can label a supported unique interval match as inferred. IMPLEMENTED C2 candidate, execution pending: retain longitudinal-only events without nearest-stay or interval-attribution fallback; use only same-patient explicit ED stay links.

**Scientific consequence:** Encounter labels, landmark eligibility and modality support differ if unlinked events are dropped or guessed.

**Alternative choices:** Patient-only attribution; explicit unique-interval inference with uncertainty flags; approved clinical episodes.

**Question for the PI:** Should unlinked observations remain patient-level events with encounter-dependent analyses restricted to verified attribution?

**Affected artifacts:** Event/encounter manifests, landmark views and labels; content embeddings need rebuilding only if the underlying content identity or recipe changes.

### SCI-02 — Occurrence, recording, availability and demographic snapshots

**Category:** DATA_LIMITATION. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** Paper p.15 describes event chronology and window-end metadata. Source occurrence/store/verification clocks differ;148,894 lab rows lack storetime. Strict M2 windows reject unknown demographic availability. [A-02/05/35](ASSUMPTIONS.md), [integration evidence](../reports/server/2026-10-20_m2_integration/REPORT.md).

**What remains uncertain or different:** Whether a recorded clock establishes actual clinical availability; when source-derived age/demographic attributes were knowable. Discharge summaries and retrospectively assigned diagnoses can reveal an encounter outcome despite an earlier associated clinical date. Literal window-end prefixes differ from cutoff-safe prediction.

**Example:** A sample collected before 24h but stored after 24h cannot automatically enter a 24h prospective representation. A discharge summary sharing a hospital ID must never be moved to admission, and its content is not evidence of earlier availability.

**Current provisional handling:** IMPLEMENTED: preserve raw clocks/unknowns, occurrence diagnostics and strict guards separately; M1 uses static birth year with five Unknown slots. PROPOSED: clearly labeled occurrence-time pretraining, not strict prospective certification.

**Scientific consequence:** Leakage can enter through late results or demographics even with causal attention; strict claims may remain infeasible for affected records.

**Alternative choices:** Occurrence-time reproduction profile; observed-availability cutoff profile; separate literal/window-end versus as-of demographic recipes.

**Question for the PI:** Should the next engineering dataset retain occurrence chronology while requiring a separate evidence-supported view for prospective evaluation?

**Affected artifacts:** Temporal/demographic event views, windows, representations and prospective metrics; raw content embeddings usually remain reusable.

### SCI-03 — Imprecise clocks, simultaneous events and cross-dataset alignment

**Category:** PAPER_UNSPECIFIED. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** All 57,381 indexed hospital procedure rows have date-only syntax; diagnoses lack a precise event clock. DATE-named fields can contain midnight-formatted strings without subday precision. Paper does not provide executable tie/proxy rules. A-01/02; [C1 evidence](../reports/server/2026-10-20_m2_c1/REPORT.md).

**What remains uncertain or different:** How imprecise records enter a precise sequence; how equal-time observations should be ordered; and whether all linked datasets preserve identical patient-specific deidentification shifts/time conventions. Deterministic serialization does not establish a clinical causal order.

**Example:** A retrospective diagnosis cannot be treated as known at admission merely because its stay ID is known. Two measurements with the same timestamp can be serialized lexically while their actual clinical order remains unknown; a shared subject ID alone does not independently verify cross-dataset clock alignment.

**Current provisional handling:** IMPLEMENTED C1: raw precision and source identities are preserved. IMPLEMENTED C2 candidate, execution pending: retain date-only/untimed records in explicit deferrals; sort precise events by subject, occurrence hour and stable event ID; retain supplied cross-dataset clocks without a guessed shift correction. Retrospective ordering proxies remain unimplemented. Equal-time order is a serialization convention, not causal evidence.

**Scientific consequence:** Category coverage, adjacency, zero-gap transitions and temporal attention change. Excluding imprecise records makes the timed stream incomplete relative to the paper; incorrect cross-dataset clock alignment could corrupt multimodal sequence order. Prospective use also requires availability, not just a deterministic ordering.

**Alternative choices:** Separate coarse/untimed records; documented stable ties versus evidence-backed source precedence; separately verified dataset alignment; a declared retrospective proxy or coarse-time scheme in a different dataset revision.

**Question for the PI:** For later fidelity comparisons, should unresolved within-time order and cross-dataset clock alignment be tested as explicit sensitivities while imprecise records remain separate?

**Affected artifacts:** Event ordering/shards, target counts, windows and trained checkpoints; source-linked content embeddings can generally be reused.

### SCI-04 — Medication actions and overlapping source representations

**Category:** PAPER_UNSPECIFIED. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** Prescriptions/pharmacy/eMAR/pyxis/reconciliation represent different processes. Indexed pharmacy/prescriptions have 1,178,787/1,338,360 rows, not proven record equivalence. A-01; [source adapters](../packages/noah_baseline/m2/adapters.py).

**What remains uncertain or different:** The paper's executable mapping of order, scheduled start, dispensing, observed administration and stop, including duplicate representation.

**Example:** A planned prescription stop is not an observed infusion stop; dispensing is not proof of administration.

**Current provisional handling:** IMPLEMENTED: separate source identities, not-given actions rejected as administration, unresolved primary Rx source deferred. IMPLEMENTED C2 candidate: prescriptions/pharmacy remain separately deferred; pyxis dispensing is not mislabeled administration. Prescriptions-primary reconciliation remains an unselected alternative.

**Scientific consequence:** Drug frequencies, dose/action identities, event transitions and downstream simulations can be misleading.

**Alternative choices:** Distinct source-action events; reconciled prescriptions-primary; reconciled pharmacy-primary, with explicit action semantics.

**Question for the PI:** Which medication actions should be distinct canonical targets, and should duplicate order representations be reconciled before training?

**Affected artifacts:** Medication event shards, specifics queues, class statistics/windows/checkpoints; identical verified text-content vectors may be reused.

### SCI-05 — Executable category and specific-item ontology

**Category:** UNVERIFIED_ASSUMPTION. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** Paper lists 31 categories but not a complete source/item ETL. The local registry covers the vocabulary; actual adapter checks cover selected rows, not every clinical mapping. A-01/08/24; [traceability DATA/DEC](PAPER_TRACEABILITY.md).

**What remains uncertain or different:** Chart-item Vitals/Other Event assignment, qualitative payload representation and event/specifics granularity; complete registry coverage is not validated event support.

**Example:** A microbiology organism/result should not silently be encoded as a numeric laboratory measurement.

**Current provisional handling:** IMPLEMENTED: dictionary labels required and unclassified/qualitative cases explicitly deferred. IMPLEMENTED C2 candidate: version existing source-backed adapters, an explicit 11-item ICU vital whitelist, and deferred reasons; no density or modality resampling. Unmapped categories remain unsupported rather than fabricated.

**Scientific consequence:** Different ontology choices change category/modality targets and meaningful patient eligibility.

**Alternative choices:** Conservative source-backed subset with declared unsupported cases; a reviewed expanded mapping including qualitative/categorical payloads.

**Question for the PI:** Is a declared conservative canonical subset acceptable while unsupported semantic mappings remain visible, or are particular missing categories essential to the next experiment?

**Affected artifacts:** Canonical events, specifics/content queues, support masks, weights, windows and checkpoints.

### SCI-06 — Numeric units, Fourier convention and ICU aggregation

**Category:** PAPER_UNSPECIFIED. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** The paper specifies 22 dyadic scales and vital extrema aggregation but not the complete numeric formula, unit harmonization or span/stride. A-04/10. The 44-D raw-value candidate works;1,443,946 lab source rows lack units, and a bounded HR sample was hourly.

**What remains uncertain or different:** Item-specific unit conversion, missing/qualitative measurements, Fourier phase/range, extrema token identity and contributor timing under overlapping windows.

**Example:** The same physiological quantity in two units yields different Fourier values unless its representation recipe explicitly harmonizes units.

**Current provisional handling:** IMPLEMENTED: source values/units retained, numeric zero preserved, unknown/nonfinite units/results deferred by adapters;44-D candidate retained. IMPLEMENTED C2 candidate, execution pending:2h span/1h stride, per-patient/hospital/stay/item/unit grouping, hour-floored first-observation origin, left-closed/right-open windows, two extrema at contributor-mean time, and complete contributor edges. The whitelist is explicit in `configs/m2_c2.json`; unsupported chart items are deferred, not relabeled. Source units remain separate, with no population-fitted conversion. Numeric specifics append the exact native unit to the source label, so differing units are not hidden from model inputs; this is a declared candidate lexical change, not an author setting. Strict views must apply cutoff safety before aggregation or use verified closed-window outputs.

**Scientific consequence:** Changes numeric embeddings, event density, missingness, class statistics and ≥64-target support; future contributors can leak through aggregates.

**Alternative choices:** Source-unit-specific tokens; validated canonical-unit conversions; explicit alternative span/stride/extrema recipes with recorded consequences.

**Question for the PI:** Which provisional unit/aggregation recipe should define the first frozen multimodal dataset, with unresolvable cases retained as explicit failures?

**Affected artifacts:** Numeric encodings, aggregate event IDs/times, all affected support/window manifests and checkpoints.

### SCI-07 — Public encoder identity versus NOAH's exact assets

**Category:** UNVERIFIED_ASSUMPTION. **Priority:** BEFORE_PAPER_FIDELITY_CLAIM.

**What is actually known:** All four public checkpoints are pinned and genuinely exercised; dimensions/output contracts match the implemented candidates. Their exact identity as NOAH's author checkpoints is not established. [Assets](../configs/encoder_assets.json), A-09; [integration evidence](../reports/server/2026-10-20_m2_integration/REPORT.md).

**What remains uncertain or different:** Author revisions/processor settings and encoder-pretraining overlap. For RAD-DINO, original JPEG/RGB/pinned processor/CLS is a verified candidate, not proof of author preprocessing.

**Example:** Two 768-D CXR embeddings from different checkpoint/processor recipes are not interchangeable merely because shapes match.

**Current provisional handling:** IMPLEMENTED: frozen public assets and content/checkpoint/recipe-hashed caches; incompatible legacy caches rejected. PROPOSED: scale these verified candidates with explicit fidelity limits rather than repeat feasibility.

**Scientific consequence:** Affects exact reproduction and attribution of encoder benefits, not the established feasibility of the six-modality path.

**Alternative choices:** Continue pinned public candidates; use exact version-matched author assets if independently obtained and verified.

**Question for the PI:** Is the pinned-public-encoder reproduction claim acceptable unless exact author assets become available?

**Affected artifacts:** Only embeddings whose content/checkpoint/preprocessing changes, and dependent manifests/windows/checkpoints; no blanket re-encoding of compatible caches.

### SCI-08 — Long-note truncation and pooling

**Category:** PAPER_UNSPECIFIED. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** Public BioLORD sentence settings use 128 tokens and masked mean; local candidate applies L2. All four actual note examples truncate, with 436–1,802 original tokens. Paper long-note handling is not executable from its text. A-09; [resource evidence](../reports/server/2026-10-20_m2_integration/resources.json).

**What remains uncertain or different:** How much note content NOAH retained and which aggregation produced one note representation; whether retrospective content was masked or excluded at earlier paper landmarks. Truncation is not a protection against outcome information in discharge summaries.

**Example:** A discharge outcome can occur in the first 128 tokens or later in a summary. Neither truncation nor a shared admission ID establishes availability at admission or 24h. Retaining all content requires the same availability boundary as truncation.

**Current provisional handling:** IMPLEMENTED:128-token masked mean/L2 with truncation metadata. The C2 candidate retains source chart/store clocks and exact note-content hashes, labels discharge documentation retrospective, and never substitutes admission time; unknown availability remains unknown. IMPLEMENTED C2 queue recipe retains this tested candidate with explicit truncation. Chunk aggregation remains an alternative; no cohort-scale note encoding has run.

**Scientific consequence:** Content coverage, note retrieval and training targets change; this is distinct from availability leakage.

**Alternative choices:** Declared native truncation; deterministic all-content chunking/aggregation; independently supported longer-input recipe.

**Question for the PI:** Should the first frozen note cache retain the tested native candidate or pay the cost of chunked content coverage?

**Affected artifacts:** Note embeddings and dependent event/window/checkpoint manifests; raw note hashes/identities remain reusable.

### SCI-09 — OTIS position adaptation and ECG normalization

**Category:** UNVERIFIED_ASSUMPTION. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** The trained 1×24/192-D model loads strictly and produces final patch means. Its saved 43-position bank needs adaptation for 209 positions. Both tested policies give cosine similarities 0.960–0.969 on two actual recordings. A-09; [OTIS implementation](../packages/noah_baseline/m2/otis.py).

**What remains uncertain or different:** NOAH's exact position adaptation, normalization and treatment of the 8 trailing samples per lead.

**Example:** The same recording yields different embeddings under linear interpolation and analytic extension despite unchanged learned weights.

**Current provisional handling:** IMPLEMENTED: canonical lead order, robust normalization, explicit tail discard and separate caches for both position policies. IMPLEMENTED C2 queue recipe: upstream interpolation is explicitly selected as the first candidate; analytic extension remains a separate tested sensitivity. No cohort-scale ECG encoding has run.

**Scientific consequence:** All ECG content targets depend on this recipe; smoke evidence does not prove population-wide validity.

**Alternative choices:** Tested upstream interpolation; tested analytic extension; independently verified author recipe.

**Question for the PI:** Should upstream interpolation be the provisional production candidate, retaining the measured alternative as a sensitivity?

**Affected artifacts:** ECG embeddings and dependent manifests/checkpoints; waveform identities and unrelated modalities remain reusable.

### SCI-10 — ECHO recipe coverage and source-version difference

**Category:** PAPER_IMPLEMENTATION_DEVIATION. **Priority:** MUST_DECIDE_BEFORE_DATA_FREEZE.

**What is actually known:** Paper lists ECHO 1.0; available source is 1.0.1. Metadata-first crop/annotation/AVI/frame/normalization/video 512-D path passes on two B-mode clips from one patient. Unsupported color-flow regions fail explicitly. A-09/60; [ECHO implementation](../packages/noah_baseline/m2/echo.py).

**What remains uncertain or different:** Cross-version content differences and generality of the candidate mask/clip rules, not basic encoder feasibility.

**Example:** A color-flow clip may lose clinically meaningful color if treated as a static colored annotation.

**Current provisional handling:** IMPLEMENTED: declared 1.0.1 source, metadata crop required, explicit unsupported-content failure, no fallback zeros/raw-frame substitution. IMPLEMENTED C2 queue recipe reuses the tested candidate; later GPU processing must report actual supported/failed clips. Exact-version fidelity remains unavailable without comparison.

**Scientific consequence:** Selected ECHO population and retained visual information may differ from NOAH.

**Alternative choices:** Declared 1.0.1/B-mode candidate with failure accounting; validated extension for additional clip classes; independently accessible source 1.0.

**Question for the PI:** Is a documented 1.0.1 candidate with explicit unsupported-clip coverage acceptable for the next dataset?

**Affected artifacts:** ECHO embeddings/manifests/checkpoints if source or preprocessing changes; no rewrite of preserved source files.

### SCI-11 — Next-gap conditioning and forecasting interpretation

**Category:** PAPER_UNSPECIFIED. **Priority:** BEFORE_PAPER_FIDELITY_CLAIM.

**What is actually known:** Paper Q-next-gap/K-previous-gap conditioning is explicit; retained true next gaps disclose target arrival time. General rollout seeds/indexing and unknown-tail handling are partly unspecified. A-12/14/35/36/61.

**What remains uncertain or different:** How author training, landmark extraction, free rollout and oracle-time control align incoming/outgoing/terminal gaps and initial horizons.

**Example:** Predicting the category given a future arrival gap is not unconditional prediction of what and when occurs next.

**Current provisional handling:** IMPLEMENTED: M1/M1.5 retain declared candidate conventions, teacher-forced time-conditioned reporting and masked unknown terminal gaps; landmark guards prohibit realized future suffix timing. No new free-rollout result exists.

**Scientific consequence:** Forecasting claims, time losses, generated trajectories and checkpoint comparability depend on the information condition.

**Alternative choices:** Declared time-conditioned reconstruction; separately configured unknown-horizon free rollout; explicit oracle schedule.

**Question for the PI:** Should the next training result remain a time-conditioned learning experiment, with free forecasting evaluated only under a separately frozen protocol?

**Affected artifacts:** Temporal windows/targets, representations, rollout predictions and potentially checkpoints; content caches generally remain reusable.

### SCI-12 — Candidate architecture and exact paper-size reproduction

**Category:** PAPER_IMPLEMENTATION_DEVIATION. **Priority:** BEFORE_PAPER_FIDELITY_CLAIM.

**What is actually known:** Paper reports 150,322,970 parameters; preserved legacy full models have 152,655,380, the audited full-dimension candidate 148,957,844, and the active clean engineering candidate 4,760,468. [Parameter evidence](ARCHITECTURE_AUDIT.md#candidate-parameter-inventory), A-06/07/17/19/58.

**What remains uncertain or different:** Exact block/head layouts, norms/biases, context/prefix budget and full-record chunk conventions; matching totals alone cannot resolve these.

**Example:** Adding unused parameters to match Table 16 would not reconstruct its model.

**Current provisional handling:** IMPLEMENTED: distinct revision-specific checkpoint families and unchanged reduced candidate. PROPOSED future multimodal training architecture must be frozen under its own resource authorization; no redesign is implied by this issue.

**Scientific consequence:** Exact architecture/scale claims and comparisons across revisions are unavailable without conformance evidence.

**Alternative choices:** Meaningful declared candidate experiment; version-matched author architecture if obtained; explicit sensitivity across justified layouts.

**Question for the PI:** Should the next authorized multimodal experiment retain a declared candidate architecture while exact author reconstruction remains a separate claim?

**Affected artifacts:** Model configurations, checkpoint/recovery states and metrics; content-space embeddings need not be rebuilt for a projection/backbone revision alone.

### SCI-13 — Objective reduction, weights and optimization assumptions

**Category:** PAPER_UNSPECIFIED. **Priority:** BEFORE_PAPER_FIDELITY_CLAIM.

**What is actually known:** Reconstruction/KL branches and published coefficients are specified, but likelihood scales, denominators, class-weight formula and several optimizer/stopping settings are not. A-27..34. M1/M1.5 fix inverse-square-root train-derived weights and declared component/microbatch reductions.

**What remains uncertain or different:** Whether candidate dimension/mask/weight reductions reproduce author scaling and how future weighting changes interact with loss mass and eligible-target counts.

**Example:** Averaging masked microbatch losses is not generally identical to one global weighted-event mean.

**Current provisional handling:** IMPLEMENTED: fixed candidate reductions/weights and recovery-tested optimizer clocks; no alpha changes. Future training protocol and alpha sensitivity are not authorized by dataset work.

**Scientific consequence:** Apparent rare-category improvements can reflect normalization, optimization budget or target population changes rather than the weighting exponent.

**Alternative choices:** Retain the declared candidate; a separately preregistered normalized objective/weighting comparison with fixed targets, architecture and resources.

**Question for the PI:** Should the first multimodal learning experiment retain the current declared loss/weights, reserving all normalization/alpha changes for a later controlled protocol?

**Affected artifacts:** Train-fitted statistics, optimizer states/checkpoints and predictions; no content-embedding rebuild for loss changes alone.

### SCI-14 — Paper landmark/outcome boundaries versus approved pilot mortality

**Category:** PAPER_UNSPECIFIED. **Priority:** DEFERRED.

**What is actually known:** M1 approved and verified first-admission+24h, alive/still-admitted eligibility and encounter-specific in-hospital mortality. Paper probe landmarks/LOS boundaries are different and incompletely operationalized. A-41/48/49; [M1 evidence](../reports/server/2026-10-19_m1/REPORT.md).

**What remains uncertain or different:** Full paper encounter selection, inclusive admission/discharge ties, LOS horizon comparisons and outcome ascertainment. The accepted pilot rule is not unresolved for that pilot and must not be silently generalized.

**Example:** A later fatal stay must not relabel an earlier surviving index encounter; discharge-time retrieval can include outcome information unavailable at 24h.

**Current provisional handling:** IMPLEMENTED: encounter-keyed labels/single-state guards and pilot protocol. PROPOSED paper-landmark and strict prospective protocols remain separate future evaluations.

**Scientific consequence:** Cohorts, labels, representation information and mortality/LOS comparisons change.

**Alternative choices:** Continue the approved pilot estimand; define paper-landmark retrieval separately; preregister a new prospective clinical endpoint.

**Question for the PI:** Which future endpoint/landmark should be primary after multimodal learning, keeping the approved 24h pilot separate from paper retrieval?

**Affected artifacts:** Cohort/label/landmark manifests, probe fits and metrics; no automatic backbone or content-cache rebuild.

### SCI-15 — Readmission account/episode, return and follow-up definition

**Category:** PAPER_UNSPECIFIED. **Priority:** DEFERRED.

**What is actually known:** The source investigation found genuine cross-account overlaps and cases a nonnegative immediate-successor fix misses. Paper's 30-day discharge-referenced MC task does not specify transfer/account merging, observation/planned admissions, competing death or complete follow-up. [Readmission source evidence](../reports/server/2026-10-19_readmission_intervals/REPORT.md), A-41..43.

**What remains uncertain or different:** The index unit and qualifying return, exact-zero/same-day boundary, conflicting death/disposition markers and outside-system ascertainment.

**Example:** An overlapping immediate admission may not qualify, while a later non-overlapping return within 30days does; no next recorded row is not proof of 30days without readmission.

**Current provisional handling:** IMPLEMENTED: preserve hospital identities/intervals and reject negative-gap primitives. Investigated corrected labels remain diagnostic, not approved production endpoints; no new readmission evaluation runs.

**Scientific consequence:** Label prevalence, censoring, competing risks and interpretation of prospective versus observed-system return differ.

**Alternative choices:** Hospital-account return with explicit overlap exclusions; approved continuous-care episodes; a clinically restricted unplanned-return endpoint with demonstrated follow-up.

**Question for the PI:** Which account/episode return definition and competing-death/follow-up policy should govern a later readmission experiment?

**Affected artifacts:** Readmission cohorts/labels/predictions/metrics; event identities change only if a separate approved episode construction is adopted.

### SCI-16 — Conditional statistical uncertainty and training-seed variation

**Category:** UNVERIFIED_ASSUMPTION. **Priority:** DEFERRED.

**What is actually known:** Patient-cluster and exactly aligned paired bootstrap utilities are implemented/tested. M1.5 has one training seed; paper task-specific folds/seeds/CI conventions vary. A-40/48; [uncertainty implementation](../packages/noah_baseline/evaluation/bootstrap.py).

**What remains uncertain or different:** Which CI estimand applies to each future comparison and how between-training-seed variation is reported; implemented bootstrap code does not estimate unobserved seed variation.

**Example:** A narrow patient-bootstrap interval for one fitted model is not evidence of reproducibility across training seeds.

**Current provisional handling:** IMPLEMENTED: explicit bootstrap seeds/budgets, patient clustering, paired alignment and degeneracy reporting. PROPOSED: report conditional patient uncertainty separately from seed-level results; no new seeds authorized.

**Scientific consequence:** Significance, claimed superiority and rare-outcome precision can be overstated.

**Alternative choices:** Conditional patient intervals plus separate seed dispersion; a justified hierarchical resampling design with additional authorized runs.

**Question for the PI:** Should future comparisons report patient-cluster paired intervals conditional on each model and training-seed variation separately?

**Affected artifacts:** Statistical protocols and metric/CI artifacts; extra model checkpoints only if additional training is authorized.

### SCI-17 — Observation/recording process and causal interpretation

**Category:** UNVERIFIED_ASSUMPTION. **Priority:** DEFERRED.

**What is actually known:** Recorded event sequences depend on clinicians' testing/treatment/documentation decisions, not physiology alone; NOAH's observational event model does not identify causal intervention effects by itself. A-57/64; [scientific interpretation register](ASSUMPTIONS.md).

**What remains uncertain or different:** Generalization under changed care/documentation policies and whether simulated event contrasts support any causal estimand.

**Example:** More frequent lab events may reflect clinician concern rather than a treatment-induced biological change.

**Current provisional handling:** IMPLEMENTED reporting treats results as conditional event-pattern learning/diagnostics, not causal efficacy or irreducible uncertainty. No correction for observation intensity has been implemented or proposed as a paper fact.

**Scientific consequence:** Forecast interpretation, cross-site transfer and intervention claims can exceed what the data/model establish.

**Alternative choices:** Explicitly observational trajectory learning; a separately designed observation-policy/causal analysis with additional assumptions and validation.

**Question for the PI:** Should the next experiment's claim remain observational clinical-event learning, excluding causal intervention conclusions?

**Affected artifacts:** Claim/analysis protocols; new model inputs/checkpoints only if a separately approved observation-process model is introduced.

### SCI-18 — NEWS2's unresolved label/scoring protocol

**Category:** PAPER_UNSPECIFIED. **Priority:** DEFERRED.

**What is actually known:** The paper describes local-window NEWS2 retrieval; the preserved legacy surrogate is not validated NEWS2. Exact source items, scoring-scale/oxygen/consciousness mappings and incomplete-window handling remain unresolved. A-50/51; [evaluation traceability](PAPER_TRACEABILITY.md#rollout-and-scientific-evaluation).

**What remains uncertain or different:** Which executable source/scoring/window recipe reproduces the reported endpoint and how rare/incomplete cases are handled.

**Example:** Filling missing observations with a plausible low score invents a clinical label.

**Current provisional handling:** IMPLEMENTED: missing-input rejection/reporting guards; old scores remain excluded from validated NEWS2 claims. No corrected NEWS2 extraction/evaluation is selected for the present dataset phase.

**Scientific consequence:** Future NEWS2 comparisons are unavailable until independent label validation; unrelated source indexing/embedding generation is unaffected.

**Alternative choices:** Independently validate the paper retrieval protocol; explicitly omit this benchmark from a reduced claim; a deterioration endpoint is a separate new methodology.

**Question for the PI:** Should NEWS2 remain deferred until an independently verified paper-style label protocol is ready?

**Affected artifacts:** NEWS2 labels/windows/probes/metrics; general encoder caches and clinical source indexes need no automatic rebuild.


### SCI-19 — Cohort density, pretraining eligibility and early-evaluation selection

**Category:** UNVERIFIED_ASSUMPTION. **Priority:** BEFORE_PAPER_FIDELITY_CLAIM.

**What is actually known:** The fixed cohort has 15,000 source patients and 10,500 train members, selected without new outcome/density inputs while preserving the enriched historical pilot. Paper minimum 64-target eligibility, actual canonical coverage and endpoint-specific eligibility are distinct. [Cohort evidence](../reports/server/2026-10-20_m2/REPORT.md), [C1 supports](../reports/server/2026-10-20_m2_c1_recovery_execution/source_counts.csv), A-01/48/49.

**What remains uncertain or different:** The author's exact density denominator and whether full longitudinal eligibility preceded early-landmark cohort selection. Final canonical/fully encoded/owned-target counts are not implied by cohort assignment; downstream risk sets also require index encounter, survival, follow-up and cutoff-specific evidence.

**Example:** Requiring 64 events over a full future trajectory can preferentially retain long survivors, recurrent admissions or intensively observed patients in an admission+24h analysis.

**Current provisional handling:** IMPLEMENTED cohort: patient-disjoint memberships remain fixed. IMPLEMENTED C2 candidate, execution pending: count assigned, canonical, ≥64-candidate, feature-complete and ≥64-valid-owned-target patients separately; retain patients with zero outputs. PROPOSED future clinical evaluation: define eligibility using information available at its landmark, and disclose any paper-style full-record-conditioned comparator separately. Accepted M1 eligibility is not reopened.

**Scientific consequence:** Selection on survival/future observation affects risk, apparent discrimination, calibration and comparability to the paper or a deployment population.

**Alternative choices:** Pretraining density eligibility plus an independent landmark risk set; a separately labeled density-conditioned retrospective comparison with coverage/selection reporting.

**Question for the PI:** Which estimand should a later paper comparison report alongside the independently landmark-eligible clinical population?

**Affected artifacts:** Eligibility/support/window manifests and later clinical cohorts, labels, probes and metrics; fixed splits and content embeddings need no automatic rebuilding.

### SCI-20 — Uneven modality support and generalization

**Category:** DATA_LIMITATION. **Priority:** BEFORE_PAPER_FIDELITY_CLAIM.

**What is actually known:** C1 has 48,130 ECHO DICOM records from 358 patients (253/60/45 train/validation/test), versus 65,137 ECG records from 10,824 patients and 32,133 CXR images from 4,893 patients. These are source supports, not successful encoding or canonical-event supports. [Executed split/table supports](../reports/server/2026-10-20_m2_c1_recovery_execution/source_counts.csv).

**What remains uncertain or different:** How successful encodings, clinical selection and repeated-record density concentrate representation learning; whether subsequent performance generalizes beyond the modality-observed subgroup.

**Example:** Many clips from one ECHO study can yield many targets without adding independent patients; a complete-case six-modality cohort would select a different clinical population.

**Current provisional handling:** IMPLEMENTED cohort: preserve all members and natural modality absence. IMPLEMENTED C2 candidate, execution pending: count events and contributing patients by split/category/modality, deduplicate content work without erasing event bindings, and inventory failures. No complete-case requirement, implicit modality balancing or replacement of missing content with zeros is permitted.

**Scientific consequence:** Effective support and uncertainty can be overstated by record totals; modality-rich patients can dominate and complete-case analysis can change the estimand.

**Alternative choices:** Natural-coverage learning with conditional support reporting; later authorized patient/modality-balanced sensitivity under a separate protocol, without changing the primary fixed cohort.

**Question for the PI:** Should later claims report modality-observed subgroup performance and natural-cohort performance separately, with patient-level uncertainty?

**Affected artifacts:** Support/encoding manifests, sampling/weighting protocols and later metric populations; existing source indexes remain unchanged, and only separately approved weighting changes require new training.
