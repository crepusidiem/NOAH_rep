# NOAH technical implementation guide

Clean-worktree context: operational paths and results below describe the archived implementation unless mapped by [MIGRATION_BASELINE.md](MIGRATION_BASELINE.md). Active imports use `noah_baseline`; current stage and supported commands are in [PROJECT_STATUS.md](PROJECT_STATUS.md). Scientific requirements and paper citations are retained unchanged.

Checked: 2026-10-14. This guide connects paper requirements to preprocessing, feature extraction, data windows, the model and training. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for current completion status.

## 1. Established requirements and unresolved settings

The paper specifies ECG input dimensions, patches and final mean pooling; ECHO preprocessing and global CLS extraction; and the text and CXR encoders. Remaining details include exact checkpoint versions, some input-processing parameters, boundary policies and model/training settings.

This operational guide complements:

- [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md): architecture, formulas, tensors, objectives and hyperparameters.
- [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md): exact pages, sections and formula locations.
- [ASSUMPTIONS.md](ASSUMPTIONS.md): unspecified details and independent configuration keys.
- [TEST_PLAN.md](TEST_PLAN.md): unit, integration and scientific correctness tests.
- [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md): dependencies and scaling schedule.

Primary evidence: [NOAH.pdf](../paper/NOAH.pdf), arXiv:2609.09140v1, 50 pages, SHA-256 `0f3a7bb049d6f7eb6cdf61c2aa9692d0695037851f95b73711a26ed803576730`. Technical details reference pp.15-21,29,49-50. PDF and printed page numbers coincide. Displayed formulas are unnumbered; E01-E16 in the audit documents are local locators, not the paper's equation numbers.

### Separate decision classification from evidence source

| Classification | Meaning | Implementation requirement |
|---|---|---|
| EXPLICIT | Directly specified by NOAH | Implement as stated; classify changes as DEVIATION |
| DERIVED | Follows unambiguously from the paper | Document the derivation without adding discretionary parameters |
| ASSUMED | A concrete choice for an unspecified detail | Configure independently and record rationale and consequences |
| DEVIATION | Changes a specified mechanism, scale or information condition | Name the experiment separately and explain the change |

Use a separate `evidence_source` for the NOAH paper, version-matched author code, encoder upstream code and this repository. Verified upstream code does not establish the exact settings used by NOAH. Upstream choices absent from NOAH remain ASSUMED until corroborated. Register candidates as independent configuration choices and proceed within the researcher's active task.

## 2. Current implementation evidence

The 2026-10-19 [architecture audit](ARCHITECTURE_AUDIT.md) is the code-to-paper verification matrix. [PROJECT_STATUS.md](PROJECT_STATUS.md) distinguishes earlier pilot reports, current source fixes, CPU tests and pending corrected real-data execution. Existing experiments do not establish frozen-encoder provenance or paper protocol equivalence.

Current code includes DuckDB ETL, windows, projections, temporal transformer, prior/posterior, decoder, dual loss, training and rollout. Main open gates are pretrained ECG assets, pinned text/CXR recipes, ICU vital aggregation, ECHO processing, exact parameter reconciliation and complete evaluation/recovery protocols. Preserve original content dimensions and regenerate affected caches/windows after extraction changes. Architecture changes require corrected-model training; partial loading of old checkpoint tensors is not a migration.

## 3. Complete data path and implementation order

```text
Versioned source tables, text and media
  -> Patient/encounter/study joins and event mappings
  -> Time, units, missingness and dense vital-sign handling
  -> Modality preprocessing and frozen feature extraction
  -> Content embedding caches and source indices
  -> Demographic prefixes, windows, attention masks and loss masks
  -> Trainable NOAH projections, temporal attention, prior/posterior and hierarchical decoding
  -> Dual-path training and KL annealing
  -> Autoregressive rollout using history and prior only
```

At implementation entry, put source fields, encoder choices and time conventions into run configuration. Data work and independent mathematical reference tests can proceed independently. Validate a few real records and synthetic mechanisms, then complete the two-patient path before expanding to the 1,000 candidates. Define preprocessing before full training.

Encoder assets consist of weights, architecture configuration, tokenizer/processor, preprocessing code and versions. Public release pages are identified; an unverified local copy is an inventory dependency, not evidence that an asset cannot be obtained.

## 4. Paper requirements for modality feature extraction

`N` is the number of valid records of that modality in the batch, not the patient count. A patient can have several records. Content space and NOAH hidden space are distinct.

| Modality | Explicit NOAH requirement | Content representation | Evidence |
|---|---|---|---|
| Text/event specifics | Frozen external BioLORD; precompute text and specifics | Text `[N,768]`; 768-D specifics derived from the shared encoder space | pp.15,17; Table 15 p.49; ref.39 p.29 |
| CXR | RAD-DINO image encoding | `[N,768]` | p.15; Table 15 p.49 |
| ECG | 12 leads, 10s, 500Hz; nonoverlapping 1x24 patches; mean of final patches | Patches `[N,2496,192]`; global `[N,192]` | p.15 |
| ECHO | Specified ultrasound preprocessing; EchoPrime global CLS | `[N,512]` | pp.15-16; Table 15 p.49 |
| Numeric | Fourier features with 22 dyadic scales `2^-7 ... 2^14` | Complete sin/cos pairs yield a candidate 44-D representation; full convention unspecified | pp.17,20; Table 16 p.50 |
| Categorical | Event-category embeddings learned during pretraining | Full-paper category representation `[N,768]` | p.17; Table 16 p.50 |

Freeze external parameters. Train NOAH projections jointly with the model to map content dimensions into shared `d=768` for the paper profile. Do not truncate or zero-pad all external outputs to 768 merely to match dimensions.

### 4.1 Text and specifics: BioLORD

**Paper:** text and type specifics use a frozen BioLORD encoder. Long-note segmentation, pooling, output normalization and exact weight revision are not fully specified.

**Verified upstream:** the [BioLORD-2023 model card](https://huggingface.co/FremyCompany/BioLORD-2023), [pooling configuration](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/1_Pooling/config.json) and [sentence configuration](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/sentence_bert_config.json) provide concrete interfaces. The sentence configuration sets `max_seq_length=128`; pooling averages valid word tokens rather than CLS. [modules.json](https://huggingface.co/FremyCompany/BioLORD-2023/raw/main/modules.json) contains Transformer and Pooling. The model card's manual Transformers example additionally applies L2 normalization; these call paths cannot be assumed numerically identical.

Implementation steps:

1. Define text-event granularity and source, separating note content from item/drug-name specifics.
2. Use `FremyCompany/BioLORD-2023` as a public candidate; pin revision, tokenizer and pooling. This does not establish NOAH's exact revision.
3. Set `text_max_length`, `text_long_policy`, `text_pooling` and `text_l2_normalize` explicitly. Truncation or chunk aggregation choices absent from NOAH belong under A-09.
4. Produce one defined 768-D representation per input and retain truncation/chunk statistics. Silent truncation does not establish that the whole note was consumed.
5. Validate short text, long reports, empty text and batching. Match length, pooling and normalization before comparing APIs.

Current `item_<id>` strings are lab-item placeholders. Join item dictionaries and units before treating them as clinically meaningful specifics; producing a vector alone does not establish semantic correctness.

### 4.2 Chest radiographs: RAD-DINO

**Paper:** RAD-DINO produces 768-D content vectors. NOAH does not specify its exact image resize/crop, view aggregation or pooling interface.

**Verified upstream:** the [official model card](https://huggingface.co/microsoft/rad-dino) exposes CLS and patch features; CLS is 768-D. The [released processor](https://huggingface.co/microsoft/rad-dino/raw/main/preprocessor_config.json) uses RGB, shortest edge 518, 518x518 center crop, `1/255` rescaling and per-channel mean=0.5307/std=0.2583. This is a public candidate, not a confirmed NOAH recipe. The card distinguishes public weights from weights in RAD-DINO's original paper; that distinction does not identify NOAH's choice.

Implementation steps:

1. Resolve image IDs/files from metadata. Configure image-level events versus study-level aggregation; NOAH's exact rule remains a decision.
2. Check provenance, resolution and existing transforms of `mimic-cxr-resized`. Upscaling to 518 cannot recover discarded information.
3. Pin weights and the matching processor; check grayscale-to-RGB conversion, pixel range, interpolation, cropping and existing normalization to prevent duplicate rescaling.
4. Public CLS can be an A-09 pilot candidate. Patch grids, classifier outputs and DINO training heads are different representations.
5. Preserve image ID, processor revision, output selection and cache links. Validate multiple views, extreme aspect ratios and decoding failures.

Do not indiscriminately reapply RAD-DINO's own DICOM pretraining transformations to NOAH's existing CXR-JPG inputs.

### 4.3 ECG waveforms: OTIS

**Paper:** each waveform has 12 leads, 10 seconds and 500Hz, giving raw `[N,12,5000]`. Each lead supplies 208 patches of width 24: 2,496 tokens of width 192. Mean-pool final patches to `[N,192]`, excluding CLS and classification heads.

**Verified upstream:** the [official repository](https://github.com/oetu/otis) describes samples as `(channels,variates,time)`, supporting an adapter input `[N,1,12,5000]`. [main_linprobe.py](https://github.com/oetu/otis/blob/main/main_linprobe.py) defaults to `(1,100)` patches and can drop old patch weights on mismatch. NOAH requires `(1,24)`; copying that reinitialization behavior does not reproduce its frozen encoder.

[models_vit.py](https://github.com/oetu/otis/blob/main/models_vit.py) contains different pooling branches, heads and domain/variate position representations. A method named `forward_features` does not guarantee a full patch sequence. Fix the checkpoint, final-norm position and domain settings.

Implementation steps:

1. Read WFDB headers and signals; check sample rate/count, lead names/order, physical units, missingness and anomalous values.
2. Independently configure `ecg_lead_order`, `ecg_normalization`, `ecg_missing_lead_policy` and `ecg_domain_policy`. Record any filtering, clipping, resampling or zero-filled leads.
3. Find compatible 192-D, 1x24 weights and check load keys/shapes. Do not hide width-100 incompatibility with `strict=False`.
4. Extract final-layer patches, verify 2,496 valid tokens and take their mean. Random patch dropping is outside this complete-record recipe; record output normalization.
5. `208*24=4992` leaves eight samples per lead. Configure `ecg_tail_policy`; truncation, padding or other handling is not a confirmed author rule.
6. Validate lead permutations, abnormal lengths, constant/nonfinite signals, patch means against independent references and repeated extraction stability.

Machine-measurement tables or ECG text reports can be separate sources; neither replaces the paper's waveform-content path.

### 4.4 Echocardiography videos: EchoPrime

**NOAH's specified order (pp.15-16):**

1. Extract the ultrasound region using DICOM `SequenceOfUltrasoundRegions`.
2. Remove ECG overlays, burned-in text and other annotations.
3. Resize to 224x224 and save as AVI.
4. Apply an additional 10% zoom crop when loading, then resize to 224x224.
5. Normalize with EchoPrime training-set mean/std.
6. Sample 16 frames with stride=2; zero-pad short videos.
7. Freeze EchoPrime and extract the paper's global CLS, 512-D, for each video.

**Concrete upstream candidates:** [echo_prime/model.py](https://github.com/echonet/EchoPrime/blob/main/echo_prime/model.py) uses `[N,3,16,224,224]`. It normalizes raw 0-255 floats using RGB mean `[29.110628,28.076836,29.096405]` and std `[47.989223,46.456997,47.20083]`, then zero-pads and slices from start 0 across a span of 32 with stride 2, producing 16 frames. `embed_videos` returns 512-D; `encode_study` appends 11-D view information and returns 523-D. The latter is not NOAH's video representation. Verify how the public 512-D output corresponds to NOAH's stated CLS; do not arbitrarily select an internal CLS.

[utils/utils.py](https://github.com/echonet/EchoPrime/blob/main/utils/utils.py) provides crop/interpolation helpers, including proportional cropping on each edge. This does not establish NOAH's zoom convention. Its pixel-mask helper does not use DICOM region metadata; it neither replaces NOAH's metadata extraction nor automatically fulfills all annotation-removal steps.

Independent configuration:

- `echo_checkpoint_revision`, `echo_output_selection`: pin the video encoder and 512-D path; do not invoke report retrieval or append view one-hot features.
- `echo_region_policy`, `echo_missing_region_policy`: define multiple-region selection and missing-tag handling. Replacing specified metadata extraction is a DEVIATION.
- `echo_overlay_removal`: preserve crop/mask locations and visually inspect residual text/ECG and damage to the ultrasound region.
- `echo_zoom_convention`: 10% per edge and 10% total differ; record integer bounds and interpolation.
- `echo_color_policy`, `echo_pixel_range`, `echo_normalization`: match decoded color/grayscale channels to constants. Do not divide by 255 before applying 0-255 constants.
- `echo_frame_start`, `echo_sampling_policy`: upstream start 0 is an ASSUMED pilot candidate. Multiple-clip averaging or uniform full-video sampling is not NOAH's stated method.
- `echo_padding_stage`: raw black-frame padding and zero-padding after normalization differ. NOAH does not specify the order; the upstream choice is a recorded candidate.

Validate video length and actual frame indices. For a long video, the start-0 candidate is `0,2,...,30`, not 32 output frames. Check short-video frame count/padding, 224x224 size, channels/range, annotation removal and finite 512-D outputs. Ultrasound measurement tables alone do not implement the video path.

The researcher supplied `/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iv-echo/1.0.1/`; directory existence was verified on 2026-10-19, while actual video preprocessing/encoding remains pending. A five-modality pilot can proceed under an agreed scope; complete six-modality reproduction requires ECHO. Do not fabricate zero videos or zero embeddings.

### 4.5 Numeric and categorical content

Numeric content needs no downloaded pretrained numeric model. Retain clinical type/units and use Table 16's 22 dyadic scales. Current `numeric.py` uses a sin bank followed by a cos bank, phase=1 and raw float32: a candidate convention. Its `normalize` argument currently does not perform normalization. Independently configure units, transforms, phase, ordering and precision (A-10); validate large phases against high-precision references. Do not substitute the temporal base-10000 geometric frequency bank.

Category lookup is trainable. Current enum integer ordering is a repository convention, not published author IDs. Categorical payloads, missing values and prefix encoding remain under A-08. Missing content is not a pseudo-zero numeric target.

## 5. Frozen extraction, caches and provenance

These are engineering acceptance requirements, not a paper-specified cache format:

1. Freeze external parameters, use evaluation mode and extract without external encoder gradients. Check repeatability, dropout/random crops and seeds.
2. Encode raw content by record ID and preserve `record_id -> embedding`; distinguish study, video and patient-level averaging.
3. Cache keys cover content identity/checksum, weight revision, processor, output selection and preprocessing configuration/version. A change invalidates cached features.
4. Record input/success/filter counts, failure reasons, dimensions/norms/nonfinite counts and patient coverage. Failures do not silently become zero vectors.
5. Run trainable NOAH projections during training. Do not cache and freeze a random projection to 768-D.
6. Explicitly declare fitting scope for data-derived scalers, thresholds, class weights and category-modality support. Train-only fitting is an engineering candidate to prevent cross-split information; the paper does not specify every fitting scope. Use the relevant A-ID.

Each run manifest records data versions/root, cohort/split hashes, encoder model IDs/revisions, weight/configuration hashes, tokenizer/processor, normalization/pooling, sample scope, randomness, cache precision, active modalities, event set, failure policy and evidence/classification for unresolved choices. Public `main` URLs are mutable; pin a concrete revision when running.

## 6. Timelines, windows and loss masks

Declare choices for A-01..07, A-12..14 and A-62 before implementing DATA-01..07. Retain source table/row and record granularity after joins to prevent one-to-many duplication. Missing is distinct from zero. Emit process-end events at actual termination.

Current code uses January 1 of a proxy birth year as origin, hourly times and age in years: ASSUMED conventions. The vital-sign helper's 2h span/1h stride are not paper-specified. Final `delta_next=0` cannot represent both an unknown successor and a simultaneous successor; use an availability flag/target mask for boundaries.

The paper specifies a 70/15/15 patient split, at least 64 scored tokens, training/validation windows of up to 2048 tokens with 256 overlap, six demographic prefixes and no prefix/overlap losses:

- Keep every window of a patient in one split; freeze cohort/split before windowing.
- Configure whether 2048 includes prefixes, first-window handling, tails and test chunks.
- Define loss ownership on shifted target events. Score each eligible clinical event exactly once; do not discard the first window's initial 256 targets without a reason.
- Keep attention and loss masks separate. Prefixes and overlap supply context.
- Apply ownership/padding masks to posterior reconstruction, prior reconstruction and KL. Add component-validity masks for specifics, value and positive gaps.
- Window-end metadata, mean-time aggregation and true forward gaps may introduce look-ahead. Configure/report literal-paper and strict as-of conditions separately; do not silently label a causal modification as the original paper.

**Alignment example:** event times `[0,2,5]`. When predicting the second event, historical content is the first event; the second event's incoming gap is 2 and outgoing gap is 3. The decoder text states outgoing-gap prediction; 3 is not automatically the second event's arrival offset. Keep the A-14 indexing/scheduling choice and initial free-rollout gap consistent across configuration, diagrams and tests.

## 7. Minimum requirements for connecting NOAH

Implement TIME/ALIGN/VAR/DEC/LOSS/OPT/ROLL in the [specification](REPRODUCTION_SPEC.md), rather than substituting standard Transformer/VAE defaults.

| Module | Required mechanism | Settings still to configure |
|---|---|---|
| Input/projection | Category, specifics, value and time form shared tokens; external encoders frozen, projections trainable | Missingness, input-time fusion, linear bias/norm |
| Temporal attention | Q receives position/age/forward features; K position/age/backward; `log1p` gaps; causal content mask | Units, boundaries/indices, interpretation of layers 0-3, backbone details |
| Prior/posterior | `p(z_t|h_(t-1))`, `q(z_t|h_(t-1),e_t)`; diagonal Gaussians; shared latent | MLP depth/activation, h0, posterior embedding stage |
| Randomness | One shared z per event and decoder pass; smooth prior sigma restriction to 0.05-2 | Exact smooth map; no independent noise per head |
| Hierarchical decoding | Category features, specifics features, modality/value branches; time reads z | Routing, head structure, class weights/compatibility support |
| Gates | Hard specifics-presence and positive-gap decisions; positive-gap log regression only on positive targets | Decision thresholds, positive-value numerical protection |
| Numeric rollout | Coefficients in the same Fourier basis as input; reuse directly | Do not silently invert to scalars and re-encode; diagnostics may be separate |

Full-paper model: d=768, 12 layers, 12 heads, latent=384, FFN expansion=8, 150,322,970 trainable parameters (p.50). Reduced models/context are named pilot DEVIATIONs. Retain a full profile for structural/parameter checks; tiny models do not reproduce paper capacity.

### 7.1 Training objective and optimization

The paper combines posterior reconstruction, KL and auxiliary prior reconstruction (pp.20-21):

`L_t = 0.3 * L_rec(posterior sample) + beta * KL(q || p) + 0.1 * L_rec(prior sample)`.

Linearly anneal beta from 0 to 1 over 10 epochs. LR warmup uses a separate six-epoch clock. Peak LR=6e-4, cosine schedule, weight decay=0.01, gradient clip norm=10. Muon applies to hidden 2D matrices; AdamW to embeddings, heads and non-2D parameters. Tensor rank alone must not put head matrices into Muon. Dropout, drop path and attention dropout are each 0.1 (p.50).

Continuous-embedding likelihood/regression losses, variance, dimensional reduction, component weights and batch reduction are not fully published. Do not label a chosen MSE/cosine loss as the author's original objective. Record A-27..30 separately. Total epochs, global batch, optimizer internals and checkpoint selection also remain unspecified.

Validate independent synthetic mathematical references, then tiny-batch overfitting, then the real pilot. Log prior/posterior reconstruction, raw/weighted KL, beta, sigma, invalid samples, time and modality targets. Total loss alone does not verify causal/variational correctness. Record any changes to the 10-epoch KL or six-epoch LR clocks for a shorter pilot.

### 7.2 Inference

Free rollout uses observed/generated history and the conditional prior only. The posterior's true target embedding is unavailable. Decode, reproject content, combine with event/time information and append as input; predicted embeddings are not raw images/text to re-encode through external encoders.

Time-controlled mode receives specified gaps; the paper evaluates with true gaps. Report it separately from free forecasting. Simultaneous events may have true zero gaps; do not force time forward with epsilon. Record token budget, context truncation and stopping conditions.

## 8. Dependency order and completion evidence

See [TEST_PLAN.md](TEST_PLAN.md) for the complete test inventory.

| Stage | Work | Required evidence |
|---|---|---|
| A | Fix run profile, data root, source mapping and encoder candidates | Config key, classification and rationale for every unspecified setting; distinguish public assets from author evidence |
| B | Mathematical/boundary references and a few real records | Temporal/KL/Fourier/ownership references; decoded/encoded shapes and provenance |
| C | Synthetic model, complete objective and rollout | Passing alignment, causal, dual-reconstruction, shared-latent, hard-gate and prior-only tests |
| D | Two-patient complete real path | Events, real embeddings, windows and masks, beyond printed media IDs |
| E | 1,000-candidate small-scale training | Resumable checkpoints, validation, prior rollout, component logs, memory/throughput |
| F | Scale decision and paper-scale work | All modalities/events, profiles, corpus scale and evaluation protocol checked before resource estimates |

With O2 available, measure small-model peak memory, tokens and encoding/training throughput before estimating expansion. Eight A100 80GB GPUs are the paper's pretraining environment (p.50), not a minimum for the pilot.

The proposed five-modality, 1,000-candidate experiment differs from full six-modality reproduction. Configure/log active modalities, whether to retain the global six-modality vocabulary, uncovered categories and enrichment. A reduced scope must not remove prior, posterior or temporal mechanisms while claiming the original model.

## 9. Independent choices to retain

The A-ID register provides full details. Do not collapse the following into `faithful=True`:

| Scope | Independent settings | Consequences |
|---|---|---|
| Text | Checkpoint, length, long-text policy, pooling, L2 | Truncation and representation-space changes |
| CXR | Source/resize, processor, view aggregation, output selection | Visible structures, record granularity and feature changes |
| ECG | Checkpoint/patch, lead order, norm, tail, domain, final norm | Signal semantics and fidelity of frozen patch features |
| ECHO | Region, cleanup, zoom, color/range, norm, frame start, padding stage, output | Motion segment, pixel distribution and 512/523 interfaces |
| Numeric/time | Separate bases, units/transforms, phase, boundaries, gap alignment | Aliasing, causal information and target misalignment |
| Training/rollout | Likelihood/reduction, weights, annealing clocks, budget, time profile | Loss scales, learning behavior and inference information |

Within an authorized pilot, record a justified candidate and continue without waiting for every future evaluation A-ID. Keep unresolved settings visible in paper-scale configuration.

## 10. External sources and limits

Public-source check date: 2026-10-14. Pin weights, processors and code versions at execution and record loading acceptance.

- BioLORD: [model card](https://huggingface.co/FremyCompany/BioLORD-2023), [pooling](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/1_Pooling/config.json), [sentence length](https://huggingface.co/FremyCompany/BioLORD-2023/blob/main/sentence_bert_config.json), [modules](https://huggingface.co/FremyCompany/BioLORD-2023/raw/main/modules.json).
- RAD-DINO: [model card](https://huggingface.co/microsoft/rad-dino), [processor](https://huggingface.co/microsoft/rad-dino/raw/main/preprocessor_config.json).
- OTIS: [official repository](https://github.com/oetu/otis), [models_vit.py](https://github.com/oetu/otis/blob/main/models_vit.py), [main_linprobe.py](https://github.com/oetu/otis/blob/main/main_linprobe.py). Its README points to `util/preprocess.py` and weights; the exact numeric normalization formula remains to be confirmed.
- EchoPrime: [official repository](https://github.com/echonet/EchoPrime), [model.py](https://github.com/echonet/EchoPrime/blob/main/echo_prime/model.py), [utils.py](https://github.com/echonet/EchoPrime/blob/main/utils/utils.py).

Public candidates become a concrete run recipe after revision/hash pinning, shape/interface checks and assumption recording. They do not automatically establish all original NOAH settings.

## Controlled evaluation remediation

The [remediation matrix](ARCHITECTURE_AUDIT.md#controlled-remediation-preserved-evidence-and-revision-separation) defines completed guards versus pending scientific decisions and real-data integration. Historical source/configuration/checkpoint/report preservation precedes changes. Current full-dimension and legacy checkpoints must not be mixed; strict tensor loading alone does not attest semantic configuration or encoder lineage.

New low-level APIs are `evaluation.outcomes.encounter_labels` / `readmission_label` and `evaluation.landmarks.extract_landmark_state`. They require selected encounters, endpoint parameters and a fully evidenced cutoff-bound view. They do not choose a last encounter, repair old shards, infer availability, sanitize a true next gap, build clinical NEWS2 labels, select folds or train a probe. The legacy pooled/subject-max clinical CLI and old reconciliation/reporting paths are held until approved integration exists.

Keep model/training/input-encoding configurations unchanged during this initial guard phase. Source availability is unknown where absent, not replaced by occurrence or aggregate mean time. Before a real corrected evaluation, implement the approved raw-source/provenance/view contract and verify future-suffix invariance at that boundary. Keep literal paper timing and strict prospective timing as separate manifested modes. No alpha-sweep or retraining is authorized by a passing synthetic contract test.


## M1 bounded occurrence-time mortality integration

The researcher's M1 authorization supersedes B0's blanket hold for this named phase. `configs/m1.json` is the effective protocol. Use the isolated launcher `python -I tools/run.py m1-check --output artifacts/reduced_pilot/evidence/<unique>.json`, then `python -I tools/run.py m1 --hosp <MIMIC-IV2.2 hosp> --ed <MIMIC-IV-ED2.2 ed> --cohort <verified original cohort CSV> --run-id <unique> --fixture-evidence <executed evidence>`. Only the supplied original1000-patient manifest hash is accepted. The `m1` command is CPU-only, refuses existing output directories, and caps optimizer updates at16 (the configured demonstration retains four updates). Generic training and alpha launchers remain blocked.

The M1 scope includes hospital/ED entry/exit categories and ED heart rate in bpm. Hospital `admissions`, `patients`, ED `edstays` and `vitalsign` source hashes are recorded before/after extraction. Patient, hospital and ED stay joins are checked explicitly. No raw clinical text is extracted. `vitalsign.charttime` is used as a source occurrence-coordinate proxy, not proof of availability; all four source adapters retain unknown available-at as null. Header/body CSV record ordinals plus source hash identify heart-rate rows without dropping identical records. Units follow the [official ED vital-sign definition](https://mimic.mit.edu/docs/iv/modules/ed/vitalsign.html). Only heart rate is modeled, avoiding a mixture of unlabeled numeric measurement types while external specifics encoders are absent. This is not the paper's complete vital preprocessing.

Index selection is the first hospital admission by `(admittime, hadm_id)`, before outcome/eligibility checks; an excluded short/early-death index is never replaced by a later stay. The24h risk set requires discharge and death after24h, known mortality labels/timing and consistent death markers. The label is the selected `hadm_id`'s hospital-expire flag. Original train/val/test membership is immutable. Unknown/malformed intervals and contradictory outcomes have explicit exclusion reasons.

Views include only that encounter's source events at or before admission+24h, including its linked pre-admission ED records. Dynamic demographics are not inferred from future admissions: birth year is derived from source anchor year/age and encoded in one numeric prefix; the remaining five demographic slots are Unknown/unconditioned. They are not zero substitutes for text targets. Numeric44-D candidate encoding and all model/loss implementations remain unchanged. Context64/overlap16/min-events1 and omitted specifics are explicit engineering deviations.

`m1.data.Window` adds aligned full source/encounter/time/unit/transformation provenance and an independent shifted-target owner ledger around candidate model tensors. Prefix/overlap/padding do not duplicate losses. Retained gaps are recomputed after cutoff; each chunk's last gap is zero with `has_next_gap=False`, distinguishing unknown from an observed zero. One final-window event state is extracted per index, checked against an independent exact `h[last]` readout; no window or trajectory pooling occurs. The decision cutoff and state-event time are both saved.

`m1.training.Trainer` is the canonical bounded training/recovery path. It retains the256/128/4-layer pilot dimensions and candidate weighted component means, Muon/AdamW routing, LR/KL clocks and microbatch-mean accumulation. Category/modality/presence weights use owned train targets and the fixed existing `(count+1)^(-0.5)` candidate; this is not an alpha sweep. Mean-of-microbatch objectives are checked against their reference, not falsely asserted equivalent to arbitrary global-event normalization. Checkpoints include exact effective configuration, source/data/window identities, parameter weights, optimizer buffers, RNGs, update count and data cursor. Loading refreshes optimizer parameter-group aliases before subsequent LR updates; uninterrupted and resumed operations must agree bitwise.

The probe fits StandardScaler and logistic regression only on training patients. Validation log loss selects C from0.01/0.1/1, without a train+validation refit; the test set is used only for final evaluation. Real/fixture perturbations of test inputs and labels must leave preprocessing, selected C and coefficients unchanged. Predictions and all metric computations use a fixed serialized protocol; metric reductions use float64 so saved-artifact replay agrees. Single-class AUROC is null, not invented.

All patient-level events, windows, labels, representations, probes, checkpoints and predictions remain in ignored `artifacts/reduced_pilot/runs/<run-id>/`. Every successful run saves full manifests, source tar, configuration/environment, source/cohort hashes, source parity checks, reference objective, exact recovery, suffix checks, probe invariance, predictions, metrics and artifact inventory. Existing release gates allow only conservative retrospective diagnostic reporting, and reject paper-faithful or prospective promotion. These engineered occurrence-time diagnostics are not clinical validation or converged NOAH results.

## M1.5 bounded learning execution

The researcher-approved [M1.5 protocol](M15_EXECUTION_PROPOSAL.md) uses a separate `noah_baseline.m15` controller and `tools/m15_run.py`, preserving M1's source and sixteen-update guard. It references the immutable M1 windows/manifests in place, trains from seed42 initialization, fits statistics only on train and evaluates only train/validation. The permanent `approved-run-claim.json` prevents a second run under this approval; there is no automatic continuation or tuning launcher. Use an exactly two-CPU SLURM step with16GiB memory, external elapsed timeout and the explicit one-run identifier. Outputs remain under ignored `artifacts/reduced_pilot/m15/`, separate from M1.

Category selection is teacher-forced, time-conditioned prior prediction; posterior reconstruction, time diagnostics and numeric44-D reconstruction remain explicitly distinct. Component means preserve the candidate loss reductions, while predictive scores use event means. Epoch checkpoints contain model/optimizer/RNG/cursor/identities; prediction NPZ files retain lossless numeric outputs and targets for metric replay. `tools/m15_verify.py` is the inexpensive real operation-equivalence check; `tools/m15_report.py` consumes completed saved artifacts and runs no model. Consult the executed M1.5 report for source commit, command, resource usage, selected checkpoint and actual learning results. No GPU, test evaluation, mortality tuning, alpha sweep or scientific-method change is part of this protocol.
