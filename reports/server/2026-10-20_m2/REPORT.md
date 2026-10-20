# M2-A and bounded M2-B execution

Executed 2026-10-20 in `output/worktrees/clean-baseline`, branch `development/clean-baseline`. Generating HEAD: `0d30e7c95b0de571bc286cbfa86add078b1ed945`, with new M2 source captured privately by file/bytecode hash. This report covers expanded source inventory and small actual-source checks. M2-C integrated shards/windows and M2-D final event statistics are pending. No NOAH training, alpha sweep, clinical performance study or bulk embedding extraction ran.

## Cohort and immutable evidence

The hospital-admitted source universe contains 180,733 distinct subjects. Preserve all 1,000 M1 patients and their original 700/150/150 split assignments, then rank remaining subjects by SHA-256 of `m2:20261020:{subject_id}` and take 9,800/2,100/2,100 for train/validation/test. Inputs to this sampling function are IDs and protected split memberships only; outcomes, event density and modality availability do not enter it. All admissions for a selected subject inherit the one patient split. No all-modality intersection is required.

The protected pilot was originally enriched using ECG/CXR metadata. Thus the combined cohort inherits that sampling component and must not be described as an unbiased MIMIC population sample. Newly sampled patients are outcome/modality blind. The cohort is frozen before source/outcome counts; no replacement based on subsequently observed coverage is permitted. Minimum 64-token eligibility is not yet applied: **10,500 assigned training patients is not a claim of 10,500 qualified training sequences**.

| Split | Patients | Hospital admissions | First-index +24h eligible | In-hospital deaths among eligible |
|---|---:|---:|---:|---:|
| Train | 10,500 | 25,733 | 7,693 | 193 |
| Validation | 2,250 | 5,316 | 1,675 | 47 |
| Test | 2,250 | 5,520 | 1,678 | 39 |

Index selection and eligibility use the unchanged M1 `select_indexes`: first admission by timestamp/ID, selected before eligibility, alive and still admitted at +24h, encounter-specific outcome. Exclusions (train/validation/test): not in encounter at +24h 2,788/573/568; death after encounter end 7/1/3; contradictory mortality markers 5/1/1; invalid interval 7/0/0. These are support counts, not evaluated predictions or guarantees of statistical power. No held-out metric/model tuning was performed.

Private manifests: `artifacts/reduced_pilot/m2/m2-a-20261020-01/`. Patient split SHA-256 `5df6b9e96b1921b94867cdacd04440de25a42e84d33c4a6fbd9ad5e26277d869`; protected original split `0beadca1a4580c37abff93252c57b73ca5ea7dadabb26865f1e43b1d5f0cac7e`; index manifest `8c969db659332b9339849efb26f9ec868e42a29aac2501b220747f22aefe7208`. Full admissions/patients hashes and source identities are in `cohort_manifest.json`. Large source inventory fingerprints are explicitly path/size/mtime/header identities, not full content hashes; full extraction must strengthen these before cache release.

M1's 32 and M1.5's 105 inventoried artifacts were rehashed and remain read-only, unchanged. See [preservation receipt](preservation.json). No raw source, historical model or old embedding cache was overwritten or copied.

## Actual source coverage

Hospital/ICU, ED and Notes are MIMIC-IV 2.2; CXR 2.0.0; ECG 1.0; ECHO 1.0.1. ECHO differs from the paper's 1.0. Roots are explicit in `configs/m2.json`. Twenty-six candidate source tables were counted: 25 in the initial 136.73s inventory and the 13.73GB lab table in a separate 90s-capped query, completed in 35.40s. [Source counts](source_counts.csv) include contributing patients, linked admissions and distinct item counts where available. Candidate category mappings are hypotheses for adapters, not final category counts. A source table can produce multiple end/start events, include invalid/unusable rows or overlap another source.

| Source | Train records / patients | Validation records / patients | Test records / patients |
|---|---:|---:|---:|
| Lab rows | 6,847,469 / 10,331 | 1,400,976 / 2,219 | 1,533,476 / 2,213 |
| Discharge notes | 19,973 / 8,488 | 4,168 / 1,836 | 4,387 / 1,854 |
| Radiology notes | 127,596 / 9,502 | 26,764 / 2,013 | 26,885 / 2,039 |
| CXR images | 22,992 / 3,438 | 4,546 / 733 | 4,595 / 722 |
| ECG records | 45,855 / 7,549 | 9,554 / 1,647 | 9,728 / 1,628 |
| ECHO DICOM files | 33,341 / 253 | 8,139 / 60 | 6,650 / 45 |

Distinct lab items: 850/767/769. Patient counts across tables are not additive. Rare-category support and actual post-aggregation event frequencies remain unmeasured until canonical adapters are completed; zeros must not replace unknown counts. File counts are not unique clinical studies or eligible landmark events.

All cohort-referenced ECG `.hea/.dat` pairs and ECHO files physically exist. The initially inspected CXR resized PNG derivative contains only 67/0/14 cohort images. The complete original JPEG tree at `/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/2.0.0` contains all 32,133 referenced cohort images. Its record list and decompressed metadata are content-identical to the derivative's indexes. The compressed metadata bytes differ; both container/content hashes are retained in `cxr-source-resolution.json`. JPEG resolution uses the index path with `.dcm` replaced by `.jpg`. No images were regenerated or overwritten. CXR physical census took 73.93s; combined earlier derivative/ECG/ECHO census took 216.34s. Existence is not bulk decoding or encoder verification.

## Genuine-source modality checks

| Modality | Status | Executed evidence | Remaining dependency |
|---|---|---|---|
| Categorical | PARTIALLY VERIFIED | Two real ED admission records; two real lab item labels encoded with frozen BioLORD, 768-D | Complete 31-category/specific-item adapters and duplicate-source ownership |
| Numeric | PARTIALLY VERIFIED | Two actual heart-rate measurements, 44-D dyadic Fourier candidate, units, exact cache readback | Valid broader units/ranges and explicit aggregation span/stride; no cohort aggregation selected |
| Text | VERIFIED for the small candidate recipe | Four actual notes and two item labels; frozen BioLORD, strict checkpoint loading, 768-D, bitwise repeated inference | Full dataset integration; long-note truncation/pooling remains ASSUMED, not NOAH-exact |
| CXR | PARTIALLY VERIFIED; encoder check VERIFIED | Two actual JPEGs decoded and encoded by frozen RAD-DINO, 768-D CLS, bitwise repeated inference | Candidate preprocessing/output selection; complete encounter/time joins and event integration |
| ECG | BLOCKED for encoding | Two genuine waveforms decoded using WFDB; source lead/rate/unit/shape metadata saved | User reports weights under a model folder; exact path still pending. Need pretrained OTIS 192-D, 1×24 verification; no random fallback |
| ECHO | BLOCKED for decoding/encoding | Two actual DICOM identities and file presence checked | Locate reported EchoPrime weights; isolated DICOM/video decoder dependencies; metadata-region crop, frames and 512-D video output verification |

BioLORD revision `167aab527b238a50ca65224e6319215d2ff4fc9f`, weights SHA-256 `f8c969586cc6b0fd393faa7d879de93d6cc532123956041fefd3474194322050`. Candidate text recipe is tokenizer right truncation to 128, attention-mask mean pooling, L2 normalization, unchanged text content. Original token counts and truncation flags are saved. This inexpensive smoke recipe is not a selection of the paper's long-document policy.

RAD-DINO revision `110cbc18d5133582e320b43d53bf5c44e410c936`, weights SHA-256 `dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae`. Original JPEG is converted to RGB and passed once through the pinned processor; final hidden CLS is retained. Processor/config/tokenizer and source content hashes bind new caches. Encoders were frozen and strict loading rejected missing/unexpected/mismatched keys. No legacy embeddings or initialized substitute weights were used.

GPU job 56938773: one NVIDIA L40S, 2 allocated CPUs, 16GiB allocation, 48s Slurm elapsed; encoder-check timer 5.98s, observed GPU allocated peak 482,716,672 bytes, Slurm peak RSS 2,903,236KiB. The first job 56937293 received a V100 unsupported by this PyTorch CUDA build and recorded BLOCKED results; it performed no successful encoding. The next request selected compatible L40S hardware without changing weights or recipes. Both attempts and raw logs remain private. Application exit zero is not modality PASS evidence.

Fourteen small event records retain source/encounter/time provenance: 2 each categorical/numeric/CXR/ECG/ECHO and 4 text. Four have source-observed encounter links; ten are explicitly unlinked rather than guessed. Four note availability times are observed `storetime`; ten are unknown. Linkage validation forbids cross-subject encounter IDs, and unique interval inference is explicitly labeled. This sample is not a completed multimodal event/window dataset. Source-event and encoder-cache association have not yet been materialized across the cohort.

## Executable contracts and statistical readiness

`m2/events.py:Event` requires source/version/row/file/encounter/linkage/time/unit/transformation identity and content/encoder/recipe identity when an embedding reference is attached. `cutoff_view` separates occurrence diagnostics from strict available-at filtering. Unknown availability is never converted into chart time. `aggregate_minmax` requires explicit positive span/stride, preserves contributors/mean occurrence/units, and derives availability no earlier than its latest contributor; any unknown contributor keeps aggregate availability unknown. It is not a default aggregation recipe.

`EmbeddingCache` uses the clean namespace and content/dataset/checkpoint/recipe hashes, rejects zero fallbacks and nondeterministic collisions, and verifies file hashes on read. No missing-modality vectors are fabricated. M2-C must persist and check the one-to-one event/reference associations and ordered window sidecars before training.

`evaluation/bootstrap.py:patient_cluster_bootstrap` implements fixed-seed PCG64 patient resampling, all encounters/landmarks retained together, percentile 95% intervals for AUROC/AP/Brier, and paired first-minus-other differences on exactly aligned patient/encounter/cutoff/label keys. No silent row intersection. Undefined class-degenerate replicates are counted; CI is withheld below max(100, 80% of requested replicates). AP uses sklearn average precision, not trapezoidal PR area. Training-seed variability remains a separate estimand. The API requires an explicit resampling budget/seed; a final clinical protocol must freeze these before use.

`python -I tools/run.py test -q`: **86 passed in 16.92s**, including the prior 79 tests plus seven M2 tests. Sampling shuffle/seed/protected-membership checks, future/unknown/delayed availability, encounter identity, cache tampering/collisions, aggregation contributors/units, bootstrap paired alignment/degenerate cases/serialized replay and an independent patient-level Brier percentile oracle passed. The reusable source-smoke launcher was executed again in fresh namespace `m2-source-smoke-20261020-04`; its 14-event JSON is byte-identical to run03. Eighty M2 evidence/source files were frozen read-only under private inventory SHA-256 `aa609c8d77fcb6ab8aec86512246bd3db7a6da9d528d1badf9f6ff9e0de586c6`. See [verification receipt](verification.json). No actual clinical model predictions were bootstrapped or compared. No claim of CI coverage calibration at sparse real outcome counts follows from these fixtures.

## Reproduction commands and limits

Executed commands used `python -I tools/m2_run.py inventory`, `lab-census`, `media-census`, `cxr-census`, `source-smoke`, and `encoder-smoke` inside Slurm allocations. Per-execution source copies/hashes identify the earlier runner without output-name arguments. The reusable runner now requires a fresh name and rejects overwriting any run. For example, from the clean worktree:

```bash
PY=/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python
srun --overlap --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=threads --mem=16G --time=00:20:00 "$PY" -I tools/m2_run.py inventory --output-name m2-a-new
srun --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:03:00 "$PY" -I tools/m2_run.py source-smoke --inventory-name m2-a-new --output-name m2-source-new
sbatch --export=ALL,M2_RUN_ID=m2-encoder-new sbatch/m2_encoder_smoke.sh
```

The encoder script currently reads the executed `m2-a-20261020-01` inventory by default; use the CLI's explicit `--inventory-name` for a different inventory. Scripts request Slurm allocations before encoder computation; no heavy work runs on the login node. Inventory has 90s/query and 20min allocation bounds. The initial per-file 4GiB cap deferred labs; the separate read-only lab count was explicitly time bounded. Large-file fingerprints remain provisional and are not a checksum of their data contents.

The source census SQL is parameterized over each recorded table path and cohort membership:

```sql
SELECT c.split, count(*) AS source_rows,
       count(DISTINCT t.subject_id) AS contributing_patients,
       count(DISTINCT t.hadm_id) AS linked_hospital_admissions,
       count(DISTINCT t.itemid) AS distinct_specific_items
FROM read_csv_auto(?, ignore_errors=false) t
JOIN read_csv_auto(?) c USING (subject_id)
GROUP BY c.split ORDER BY c.split;
```

`inventory.py:run` substitutes the actual existing item column (`itemid`, `drug`, `medication`, `icd_code`, `label`) and NULL for absent item/encounter columns. Actual lab SQL is retained privately in `query.sql`. Table indexes, headers and executed source locate all queries without publishing clinical rows. Sampling implementation is `cohort.py:select_cohort`; unchanged landmark selection is `m1/data.py:select_indexes`.

## Readiness and next bounded work

**GO:** continue independent small ECG/ECHO checks once reported assets are located; complete canonical adapters and schema-to-cache association tests. **HOLD:** bulk six-modality encoding, full multimodal training, any prospective/paper-exact claim and alpha experiments.

Raw float32 vectors for the observed 209,773 note rows, 32,133 CXR images, 65,137 ECG records and 48,130 ECHO clips would require about 0.83GiB before deduplication (768/768/192/512 dimensions). This excludes specific-item vectors, numeric events, provenance, per-file metadata, intermediate video decoding, failed records, indexes, windows and model/checkpoint outputs. It is a vector-payload estimate, not a disk quota or measured full-cohort throughput. No raw media duplication is proposed.

Before requesting bulk resources: fix aggregation span/stride and item/unit mappings as declared candidates, verify OTIS/ECHO weights/preprocessing on real records, create pilot canonical shards, count post-transformation tokens/patients per split/category/modality, then measure bounded batch throughput/peak memory and temporary bytes. Estimate GPU-hours as record counts divided by measured per-modality throughput, including I/O and failure handling. The two-example encoder timings include loading and cannot justify full-scale throughput or a training budget. The next training proposal must use measured windows/targets, candidate model dimensions and exact per-update throughput; no training resource estimate is yet executable. Preserve the fixed cohort even if some patients fail token eligibility; report the count rather than resampling for event density.

Independent unresolved choices are long-note policy, broad numeric normalization/aggregation, ambiguous media encounters, and ECHO metadata/frame preprocessing. The smoke candidates are explicitly ASSUMED, not silently made production scientific defaults. Reuse the existing specification, assumptions and test plan; do not restart an architecture audit. The immediate asset blocker is a precise local path for the user's reported model folder.
