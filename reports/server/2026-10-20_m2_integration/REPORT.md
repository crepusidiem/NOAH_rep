# M2 genuine multimodal encoding and bounded dataset integration

Executed 2026-10-20 (America/New_York), branch `development/clean-baseline`, clean worktree `output/worktrees/clean-baseline`. Generating HEAD was `f40aca5167807752939b13df4f5c81b11503b1e5` with new source captured per execution. The executed result is a **bounded six-modality engineering chain**, not completed cohort-wide M2, paper-exact NOAH or prospective clinical validation. No optimizer update, retraining, alpha sweep, mortality probe or held-out clinical metric evaluation ran.

## Executed evidence and scope

The fixed 15,000-patient cohort and its 10,500/2,250/2,250 patient-disjoint assignments are unchanged. Split SHA-256 remains `5df6b9e96b1921b94867cdacd04440de25a42e84d33c4a6fbd9ad5e26277d869`. Expanded source coverage, inherited protected-pilot enrichment and first-index +24h mortality supports remain in the [M2-A report](../2026-10-20_m2/REPORT.md). Assigned membership is not a count of qualified model-ready patients.

| Evidence | Active private run under `artifacts/reduced_pilot/m2/` | Actual result |
|---|---|---|
| Real ECG/ECHO and source-item encoding | `m2-media-real-20261020-02` | GPU encoder checks and bitwise replay passed; preprocessing remains candidate |
| Six-modality shards, windows and model routing | `m2-integration-real-20261020-05` | 208 events, 7 training patients, 10 occurrence-diagnostic windows per separate OTIS recipe |
| Structured/notes source adapters | `m2-adapters-real-20261020-02` | 46 genuine rows across 23 tables; 40 timed candidate events and 19 deferred candidate events |
| Full-M2 materialization gate | `m2-readiness-20261020-04` | Expected HOLD with nonzero exit2; no promotion of sample results |
| Bounded ICU heart-rate cadence | `m2-cadence-20261020-02` | 5,000 source observations, 19 training patients, positive-gap quartiles/p90 all60min |

Earlier attempts and intermediate namespaces remain private and are excluded from the active results above. The first media attempt saved a successful ECG check and an ECHO metadata `DataElement` TypeError; a source-shaped regression test and corrected metadata accessor resolve that implementation defect. An uncapped cadence aggregation hit its4GiB DuckDB memory cap. The reported replacement is explicitly the first5,000 eligible source rows, not a population cadence estimate. Neither failure was hidden, nor was a resource cap automatically increased.

## Six-modality status

| Modality | Status | Genuine evidence and remaining boundary |
|---|---|---|
| Categorical | PARTIALLY VERIFIED | Source boundaries and category/specifics routing work; the registry declares all31 categories. Real adapter cases cover23 tables, not complete scientific verification of every mapping. |
| Numeric | PARTIALLY VERIFIED | 92 actual heart-rate events in the integration, unchanged44-D candidate; source-like tests preserve numeric zero/units. Full item classification, units and aggregation remain unselected. |
| Text | VERIFIED for the named small candidate | Prior four actual note embeddings and two dictionary labels remain authenticated; three additional real source-item strings encoded. Public pinned BioLORD config specifies128 tokens/mean pooling; L2 is a declared candidate. All four note examples truncate (436–1,802 original tokens). |
| CXR | PARTIALLY VERIFIED; encoder check VERIFIED | Two original JPEG embeddings authenticated by source content/checkpoint/recipe/file hashes and attached to correct events. Cohort-wide linkage/preprocessing remains pending. |
| ECG | PARTIALLY VERIFIED; encoder check VERIFIED | Two actual12-lead recordings, two temporal-position candidates, final2,496×192 patches and192-D patch means; strict trained-key accounting and exact replay. Production position/normalization policy remains unselected. |
| ECHO | PARTIALLY VERIFIED; encoder check VERIFIED | Two genuine81-frame DICOMs: metadata-region crop, candidate annotation mask, lossless224AVI, explicit zoom/normalization/frame indices,512-D video embeddings and exact replay. Broader clips and annotation policy remain unverified. |

ECG and ECHO each use two records from one subject. These are functional real-source checks, not representative population validation. The cohort does not require all modalities per patient.

BioLORD/RAD-DINO files were absent at their configured locations at execution start. Their exact previously verified revisions were restored, and original weight hashes matched. `tools/download_encoder_weights.py` now manages all four pinned assets; `configs/encoder_assets.json` binds their weights and evidence. No alternate weights or old unverified embeddings were substituted. Existing clean M2 caches were reused only after hash/provenance verification.

OTIS retains the trained ECG domain (offset4, embedding indices5–16), all learned encoder weights and1×24 patch kernel. No new lead/patch weights are initialized. Decoder/projector/predictor-only checkpoint keys are explicitly accounted for. The adapter reproduces both trained attention projections, independently checked against explicit attention mathematics. It extracts final normalized patches and excludes CLS from the mean. Eight trailing samples per lead are explicitly discarded by the patch boundary.

The saved43-entry time bank must cover209 positions for full records. Both `upstream_linear` ([official linear interpolation](https://github.com/oetu/otis/blob/30282e9e2fd98511f9e33d90c2b9d86d5ae874e0/util/pos_embed.py)) and `analytic_extension` (validated analytic continuation, preserving saved entries exactly) were executed into separate caches/shards. Embedding cosine similarities are0.959835 and0.969257; mean absolute differences0.080900 and0.072259. The difference is scientifically material enough that no production winner is silently selected. These checks do not establish NOAH's exact OTIS inference settings.

ECHO uses observed `SequenceOfUltrasoundRegions` before any image mask. Missing/invalid metadata fails; there is no image-only fallback. Pydicom converts YBR to RGB once. A motion-hull/static-chromatic mask removes the observed header/ECG/depth annotations in the two privately reviewed examples. This is an ASSUMED B-mode candidate, not a general annotation-removal proof; color-flow regions explicitly fail pending a separate validated policy. FFV1 AVI round trips preserve pixels/frame order exactly. The loading crop removes5% per edge (10% total candidate), normalizes raw0–255 RGB, takes indices0,2,…,30 and pads short inputs with zeros after normalization. Released EchoPrime video output is512-D; study/view features are not appended.

## Provenance and conformance checks

Active code paths are relative to `packages/noah_baseline/`:

| Requirement | Implementation | Concrete executed verification |
|---|---|---|
| Source clocks/IDs/units and explicit unresolved rows | `m2/adapters.py:adapt_row`, `m2/adapter_smoke.py:adapter_smoke` | Source-shaped and46 actual rows; cross-patient joins rejected, date-only/missing clocks retained without midnight/discharge proxies, not-given eMAR is not administration. |
| Frozen ECG features | `m2/otis.py:OTISEncoder`, `normalize_ecg`, `temporal_positions` | Both genuine position candidates; exact patch mean/replay, lead-order/rate/nonfinite/constant checks and independent double-projection reference. |
| Metadata-first ECHO | `m2/echo.py:preprocess_echo`, `ultrasound_region`, `clip_tensor` | Actual ROI and512-D outputs, source-shaped DICOM regression, lossless AVI, color/frame/padding checks, private visual inspection. |
| Content/cache identity | `m2/events.py:EmbeddingCache`, `m2/dataset.py:validate_entry` | Independent source hash must match the reference; wrong recipe/path/dataset/modality/item/units, absent known specifics and file tampering fail. No zero/missing-value fallback. |
| Split/shard/window identity | `m2/dataset.py:write_shards`, `read_shards`, `model_windows` | Deterministic hash-checked shards, bitwise serialized tensors/exact sidecars, independent shifted owner ledger, rejected namespace escapes and duplicate targets. |
| Information boundary | `m2/dataset.py:model_windows` | Two actual overlap histories preserve every retained tensor and hidden state after future suffix perturbation. Strict mode rejects unknown/future demographic availability. |
| Full-cohort release | `m2/readiness.py:require_full_m2`, `check_materialized_run` | Actual HOLD/exit2 and forged-count fixtures;10,500 assigned members cannot replace verified ≥64-target patient counts. |

Each OTIS candidate has the same208 actual events:106 categorical,92 numeric,4 text and2 each CXR/ECG/ECHO. There are96 bound specifics targets and no missing value embeddings. Two authenticated clean M1 histories provide real overlap stress cases; they were selected for test coverage, not to alter the cohort or estimate performance. All7 patients are training patients. Alternate recipe copies are never counted as additional subjects or targets.

The unchanged4,760,468-parameter M1 candidate completed10 finite untrained forward passes per recipe. No class statistics were fitted and no parameters were optimized. The losses are numerical routing diagnostics, not learning curves or clinical metrics. Model dimensions/losses and all frozen M1/M1.5 outputs remain unchanged.

The strict event filter alone is insufficient for a prospective model: source-derived birth-year/age availability is unknown. The current guard rejects that metadata rather than inheriting the occurrence-diagnostic prefix. Consequently the final sample produces **zero strict prospective model windows**; source-known note availability does not imply a valid prospective representation. Clinical evaluation must still use the existing manifest/landmark/release contracts in `noah_baseline.evaluation` (migrated from `src/evaluation`). No retrospective, posterior, oracle-time or pooled-record output is promoted to prospective prediction.

The isolated suite passes107 tests; see [verification.json](verification.json). Existing patient-cluster95% bootstrap, exact paired alignment and metric-replay tests remain passing. No real clinical comparison was run. Clinical support is still train7,693/193 deaths, validation1,675/47 and test1,678/39 at the approved first-encounter +24h risk set; these counts do not guarantee power.

## Production choices awaiting freeze

`configs/m2_integration.json:production_choices` remains null. Candidates below are implementation proposals, not revised paper facts. A-01/02/04/09 and the existing decision register remain authoritative.

| Choice | Alternatives and recommendation | Scientific impact / required sensitivity |
|---|---|---|
| ECG temporal table | Recommend official upstream linear interpolation as the initial candidate; analytic extension is the explicit alternative. | Observed cosine0.960–0.969; retain the comparison and freeze before bulk caching. Neither proves author settings. |
| Vital span/stride | Propose2h span/1h stride for an engineering candidate, versus1h/30min and4h/1h. The bounded HR sample is hourly. | Window choice changes extrema, timestamp positions, target density and category weights. Must freeze min/max token identity and contributor availability; the sample is not a cohort cadence estimate. |
| Long notes | Recommend native128-token mean pooling with current declared L2 and explicit truncation statistics for the first candidate; chunked all-content aggregation is a separate alternative. | Four/four note examples truncate. Native upstream settings are not NOAH's exact long-text recipe; chunking changes compute and representation. |
| Missing clocks | Recommend retain untimed/date-only records separately; do not invent occurrence times. A distinct retrospective discharge/ED-boundary ordering proxy is an alternative requiring explicit approval and separate raw-occurrence/proxy fields. | Conservative handling leaves some Diagnosis/Triage/Procedure records outside model timelines; it does not complete a31-category paper stream. Prospective views must never treat a retrospective proxy as observed availability. |
| Prescription duplicates | Recommend a prescriptions-primary candidate with pharmacy retained for provenance/reconciliation; pharmacy-primary and joined-order representations are alternatives. | Dose/order/dispensing/administration differ. Do not emit both sources as independent identical prescriptions or infer observed stop events from planned stop times. |
| ECHO annotation/coverage | Keep the executed B-mode mask as a calibration candidate; validate additional region/length/color-flow cases before a production freeze. | Two reviewed clips cannot establish full preprocessing coverage. Unsupported videos fail explicitly; no raw-frame fallback, invented zero vector or silent modality loss. |

For strict prospective aggregates, construct the raw occurrence/availability cutoff view before aggregation (or use independently versioned cutoff-safe closed-window outputs); maximum contributor availability alone does not prove future-suffix invariance. These choices block a complete model-ready corpus, not independent source indexing. Strict prospective demographic availability is an additional source-evidence dependency; no alternative demographic imputation was selected.

## Measured resources and next execution proposal

See [resources.json](resources.json) for measured values and conditional estimates. GPU job56998960 used one L40S,2 CPUs and16GiB allocation: queue176s, Slurm elapsed77s, program55.96s, peak allocated GPU484,416,512 bytes and peak RSS2,721,064KiB. Initial job56998205 elapsed43s and preserved its partial/failed checks. The account's allowed `gpuquad_qos` enabled a compatible public queue; no additional GPU was allocated to change queue position.

Per-record ECHO preprocessing was5.00–6.70s; encoder forward plus exact repeat was0.037–0.193s. ECG preprocessing was0.018–0.070s; warm forward plus repeat was about0.013s, with a0.241s initial warmup-inclusive example. ECHO preprocessing/I/O dominates these examples. Applying the two clip times to48,130 files gives67–90 serial worker-hours, **conditional only**, before failures/length variation. Persisting similarly sized AVIs would consume about83.6GiB, so bounded temporary staging is preferable to retaining a second large media collection. These are not approved bulk budgets or measured population throughput.

Media/note float32 vectors alone are0.83GiB. Treating every lab/ICU-chart source row as one44-D numeric vector would add an illustrative6.33GiB before aggregation (not a canonical count or complete upper bound after expansion). Provenance, item vectors, cache metadata and windows add storage. The earlier0.83GiB figure must not be used as a full dataset quota. Next training resources cannot be honestly fixed until actual canonical targets/windows, encoder failures and batch throughput exist; the small untrained routing timing is not training throughput.

**Prepared, not submitted: M2-C1 provenance index.** `m2/source_index.py` and `sbatch/m2_source_index.sh` are concrete code; fictional CSV/Parquet fixtures pass, including multiline CSV row ordinals, patient split isolation, numeric zero, date precision, text hashes and overwrite rejection. The complete source controller is not claimed to have run at scale.

Requested next budget: **one CPU job,2 cores,16GiB,60min,8GiB new output**. It indexes the unchanged15,000-patient cohort into compressed Parquet, preserves source fields/unknown clocks, hashes source files before/after, retains item/encounter/row identity and hashes large text bodies without copying notes or media. It produces source-row counts and full source identities, not model-ready event counts or final ≥64-target eligibility. No new chronology proxies, vital aggregation, media encoding, training or alpha run is included. It can proceed independently of the production choices above.

Execution is disabled by `configs/m2_c1.json:execution_authorized=false` and a missing approval reference. A fixture confirms rejection before any output/source scan. Slurm caps CPU/RAM/wall time; the controller reserves a1GiB spill budget, uses per-file OS size limits plus metadata reservation for the8GiB output budget, interrupts at its deadline, preserves completed/partial artifacts and reports incomplete status without extending resources. The source-index runner is available only after the separately required preprocessing resource approval. No C1 job has been launched.

After C1: freeze the applicable production choices, complete canonical adapters/counts and representative preprocessing checks, then propose a separate bulk encoder budget from measured source composition. Full cohort materialization, complete31-category/rare supports, verified ≥10,000 qualified training patients, production six-modality coverage and the next training protocol remain pending. The full-M2 gate is intentionally HOLD; the alpha gate remains closed.
