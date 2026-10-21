# M2-C1 recovery execution and independent post-validation

Executed 2026-10-20, America/New_York, in `/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/NOAH_rep/output/worktrees/clean-baseline`, branch `development/clean-baseline`. The frozen 15,000-patient split remains 10,500/2,250/2,250. M1/M1.5 and historical artifacts remain unchanged.

## C1 result

The single authorized recovery job **57044273** finished **FAILED/exit 1:0**, with controller status **INCOMPLETE**. All 18 remaining inputs were indexed or dictionary-hashed, but the independent auxiliary CXR metadata census failed. ICU indexing itself completed and passed the independent prior census. This original job outcome and its failure receipt remain immutable; there was no resubmission, retry, requeue or resource increase.

The failure was DuckDB's auto-detected empty quote character on a source containing a late quoted comma, outside the earlier bounded sample. It was an independent-validator parser failure, not a streaming-index failure. The source record was not dropped or corrected. `m2/csv_reference.py` specifies comma delimiter and double-quote quote/escape characters without disabling strict parsing. A25,002-record fixture includes the quoted comma after the initial 20,480-record sniff region.

A separate inexpensive post-check in existing allocation 57035083, step 3, passed for **all 32,133 selected CXR metadata rows**. It checks every raw field, full parsed-row hash, global CSV ordinal, exact split, all existing chunk digests, whole-source selected counts/patient coverage, and source SHA-256 before/after. It neither rewrites indexes nor replaces the failed job receipt. The consolidated source collection is now **ALL_PLANNED_SOURCE_INDEXES_ACCEPTED_NOT_MODEL_READY**:28 patient-associated tables and 4 dictionaries, combining 11 accepted hospital indexes/3 dictionaries with 17 new table indexes/1 dictionary.

[Executed summary](summary.json), [all table/split row counts and patient coverage](source_counts.csv), [regression output](regression.txt), [verification source hashes](verification_source.json), and [preservation checks](preservation.json) are safe aggregates. Original raw failure details stay private because parser errors can include source records.

| Source family | Selected table rows |
|---|---:|
| Hospital, reused | 15,559,829 |
| ICU | 30,091,132 |
| ED | 587,682 |
| Notes | 209,773 |
| CXR record list plus metadata | 64,266 |
| ECG record list | 65,137 |
| ECHO record list | 48,130 |
| Total across tables | **46,625,949** |

CXR's two tables describe the same 32,133 images; they are not 64,266 image events. ICU chart events contribute 28,806,356 selected rows from 313,645,063 original CSV records in 1,649 chunks. Train/validation/test chart rows are 20,916,205/4,097,919/3,792,232; contributing patients are 3,069/665/658. These counts are source records, not validated numeric events or unique clinical observations.

| Media/source | Train rows / patients | Validation rows / patients | Test rows / patients |
|---|---:|---:|---:|
| CXR images, either source table | 22,992 / 3,438 | 4,546 / 733 | 4,595 / 722 |
| ECG records | 45,855 / 7,549 | 9,554 / 1,647 | 9,728 / 1,628 |
| ECHO DICOM records | 33,341 / 253 | 8,139 / 60 | 6,650 / 45 |
| Discharge notes | 19,973 / 8,488 | 4,168 / 1,836 | 4,387 / 1,854 |
| Radiology notes | 127,596 / 9,502 | 26,764 / 2,013 | 26,885 / 2,039 |

Cross-table patient unions and new per-field timestamp-missingness distributions were not computed by this execution. Do not sum patient supports across overlapping tables. Raw clocks, precision/unknown-availability fields, missing encounter IDs, values/units and distinct medication source identities remain preserved. Full canonical category/modality supports, valid scored targets, ≥64-target patients and trainable windows remain **NOT_COMPUTED**. Availability is not certified by an observed storage clock.

## Identity, resources and preservation

The executed clean source is `d9679d949c2d0c2b58e76e0f64b78c57e4c7064f`, dirty=false. Config SHA-256 is `86df660ca8fb1ef32f976b8f12fc8fd9f3d8b15a7369748d1cd315a3c028752b`; patient-split SHA-256 is `5df6b9e96b1921b94867cdacd04440de25a42e84d33c4a6fbd9ad5e26277d869`. Captured source/configs and full source hashes accompany private receipts. Post-validation used the later dirty source with exact file captures/hashes; it does not replace the job's executed code identity.

Submission 22:19:41, start 22:23:24, end 22:54:42. Allocation 2 CPUs/8GiB/no GPU,60min cap,256MB DuckDB validation,8GiB new-output cap. Slurm elapsed 31m 18s; controller 1873.78s; total CPU29m 40.502s. Recovery output 1,451,270,258 bytes, about 1.352GiB. Process `ru_maxrss`364,368KiB, about 355.8MiB. Slurm step MaxRSS8,388,132KiB is a separate scheduler accounting measurement, close to the allocation cap; do not substitute the process measurement for scheduler accounting or claim an 8GiB safety margin. No OOM was recorded. CXR post-validation took 3.41s, process peak 290,176KiB, approximately 10KiB output. Its existing development step requested 2 CPUs/4GiB; it was not another indexing batch job.

The final isolated suite passes **124 tests in 36.92s**, including the five new manifest/parser tests. A fixture caught and resolved a new accounting-helper SQL insertion error before real artifacts used that helper. All 32 M1,105 M1.5 and 37 accepted original-C1 artifact hashes were rechecked and match. Existing hospital indexes were referenced, not rebuilt. Recovery authorization is consumed and disabled in current configuration.

## Progress toward the unified dataset

The private accepted source manifest is `artifacts/reduced_pilot/m2/m2-c1-verified-source-manifest-20261020-01/manifest.json`; it references original index/completion/source hashes and the separate post-validation attestation. It is a source manifest, not a completed event/embedding manifest.

`m2/cohort_manifest.py:CohortManifest` adds bounded SQLite accounting on top of existing event/cache validators. Tests cover all assigned patient retention, missing/failed encoding outcomes, real cache identity, source hashes, independently derived shifted targets, duplicate/cross-patient ownership rejection and rollback safety. It records actual ingested canonical/category/modality support and ≥64-owned-target membership separately. Source-processing completeness needs separate evidence; unregistered patients are not declared clinically event-free. This helper has fixture evidence only; it has not constructed a cohort-scale canonical dataset.

Reuse existing six-modality adapters/encoders and recipe-bound caches. No cohort-scale embeddings were generated here; the previously executed 208-event/seven-patient integration remains the bounded real content evidence. The [scientific exception register](../../../docs/OPEN_SCIENTIFIC_ISSUES.md) contains seven PI summary questions and 18 consolidated unresolved issues. Specifications/assumptions/traceability remain authoritative. Unknown prospective availability does not block a declared occurrence-time engineering dataset. Unresolved identity/time mappings stop only affected outputs; no chronology, encounter ID or failed content vector is invented.

## Next concrete bulk action: separate approval required

Propose **one CPU canonical-preparation execution**,4 CPUs/16GiB/120min/**32GiB new output**, no GPU/retry/requeue/training, with 256MB validation and atomic output/resume receipts. These are proposed hard caps, not authorized resources or a guaranteed completion estimate. Source rows cannot yet predict canonical target counts or deduplicated embedding storage reliably.

Read accepted indexes, apply existing source-backed adapters under one versioned conservative candidate recipe, materialize numeric representations and candidate-event shards, and emit deduplicated text/specifics/media encoding queues with source/content/recipe/asset identities. Preserve longitudinal missing-encounter labs, unknown availability, untimed/date-only records and unresolved medication actions explicitly. Record each deferred/unsupported mapping. Freeze the applicable candidate aggregation/unit and source-mapping recipe in the reviewable run configuration before submission; do not silently promote the existing smoke settings. Until required specifics/content encodings exist, candidate counts are not valid model targets. Full media inference needs a subsequent GPU resource proposal based on actual valid/deduplicated task queues, not raw source totals. No further source-indexing job is needed for the planned C1 inputs.

M2 readiness stays **HOLD**. No bulk preprocessing/encoding, clinical evaluation, training or alpha sweep was launched.

## Reproduction entry points and private evidence

```bash
sbatch sbatch/m2_c1_recovery.sh
# The command above already consumed the sole approval; do not execute it again.

srun --jobid=57035083 --overlap --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=threads --mem=4G --time=00:04:00 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q
srun --jobid=57035083 --overlap --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=threads --mem=4G --time=00:04:00 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m2_postvalidate.py --output-name m2-c1-postvalidation-20261020-01
```

Existing output names fail rather than overwrite evidence. Allocation 57035083 was the already running development allocation, not a newly submitted batch job. Private namespaces under `artifacts/reduced_pilot/m2/` are `m2-c1-recovery-20261020-01`, `m2-c1-postvalidation-20261020-01`, `m2-cohort-preparation-20261020-01` and `m2-c1-verified-source-manifest-20261020-01`. The preparation namespace preserves exact aggregate-consolidation Python logic, executed source captures and logs. `summary.json` binds original execution, post-validation and consolidated manifests by SHA-256. No patient rows, raw logs, media, embeddings or weights are included in this report bundle.
