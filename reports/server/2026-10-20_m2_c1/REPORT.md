# M2-C1 read-only source index: executed evidence

Executed2026-10-20 (America/New_York), branch `development/clean-baseline`. This is source indexing, not canonical event construction, model readiness or prospective clinical validation.

Source commit: `d404af557944961980a45bce43f636d871465cb6`; dirty: `False`. Job: `57006471`; private run: `artifacts/reduced_pilot/m2/m2-c1-source-index-20261020-01/`. Configuration SHA-256: `70a140a403e8fb8f29ce971b46aa4a76cdc4680d65450d3d89e4c9ba5cfee180`. Source hashes and captured source hashes are attached.

`sbatch --parsable --export=ALL,M2_RUN_ID=m2-c1-source-index-20261020-01 sbatch/m2_source_index.sh` was submitted exactly once. No requeue, retry, extra budget, C2, media encoding, model optimization, clinical evaluation or alpha run is authorized by this execution.

Status: **INCOMPLETE**. Indexed source tables: 11; verified source files including dictionaries: 14/32. Error type: `OutOfMemoryException`. Source-row counts below do not count distinct clinical events or valid training targets.

The job stopped at ICU chartevents when DuckDB could not allocate another30.5MiB at its8,000,000,000-byte engine limit (reported7.4GiB/7.4GiB used). Slurm reports FAILED/exit1, not a successful completed index. The exact failing operator inside schema discovery/COPY/diagnostics was not separately recorded. No data-invalidity conclusion follows. There is no remaining partial Parquet or spill file for the failed source; its rows/full verified hash are NOT_MEASURED. Completed indexes and failure receipts are retained. Eighteen intended inputs remain unverified; see `failure.json` for every source status.

## Actual source counts and patient coverage

Patient denominators are train10,500, validation2,250, test2,250, total15,000. Splits are unchanged. Each cell below is **source rows / contributing patients**. Patients can appear in many tables; patient counts must not be summed across tables.

| Dataset/table | Train | Validation | Test | Total rows | Total patients /15,000 |
|---|---:|---:|---:|---:|---:|
| hosp/admissions | 25,733 / 10,500 | 5,316 / 2,250 | 5,520 / 2,250 | 36,569 | 15,000 (100.00%) |
| hosp/diagnoses_icd | 288,661 / 10,498 | 60,058 / 2,246 | 61,957 / 2,248 | 410,676 | 14,992 (99.95%) |
| hosp/emar | 1,557,577 / 5,977 | 356,321 / 1,332 | 379,768 / 1,317 | 2,293,666 | 8,626 (57.51%) |
| hosp/labevents | 6,847,469 / 10,331 | 1,400,976 / 2,219 | 1,533,476 / 2,213 | 9,781,921 | 14,763 (98.42%) |
| hosp/microbiologyevents | 183,873 / 8,317 | 37,434 / 1,755 | 39,471 / 1,779 | 260,778 | 11,851 (79.01%) |
| hosp/patients | 10,500 / 10,500 | 2,250 / 2,250 | 2,250 / 2,250 | 15,000 | 15,000 (100.00%) |
| hosp/pharmacy | 827,514 / 9,194 | 167,731 / 1,984 | 183,542 / 1,994 | 1,178,787 | 13,172 (87.81%) |
| hosp/prescriptions | 940,067 / 9,194 | 190,231 / 1,984 | 208,062 / 1,994 | 1,338,360 | 13,172 (87.81%) |
| hosp/procedures_icd | 40,418 / 7,158 | 8,285 / 1,500 | 8,678 / 1,536 | 57,381 | 10,194 (67.96%) |
| hosp/services | 27,987 / 10,500 | 5,760 / 2,250 | 5,995 / 2,250 | 39,742 | 15,000 (100.00%) |
| hosp/transfers | 103,313 / 10,500 | 21,621 / 2,250 | 22,015 / 2,250 | 146,949 | 15,000 (100.00%) |

The CXR record list and CXR acquisition metadata describe overlapping media and must not be summed as separate images. Patient demographics are source rows, not clinical events. Hospital/ICU/ED tables mix categorical and numeric candidates; splitting them into canonical modalities requires the unresolved item/mapping/aggregation policies. ECG/ECHO rows are media references, not validated videos/waveforms, completed embeddings or trainable events. The writer supports hash-only note payloads, but this job did not reach note tables; no new note-document count or text-encoding evidence was produced.

| Intended modality/source family | Evidence from this C1 job |
|---|---|
| Categorical/numeric candidate sources | Eleven hospital indexes; mixed values, unknown mappings and demographic rows prevent canonical modality counting. |
| ICU/ED sources | ICU chartevents failed; remaining ICU and all ED inputs were not reached. |
| Text | Both note tables NOT_REACHED; C1 coverage NOT_MEASURED. |
| CXR | Record list and acquisition metadata NOT_REACHED; C1 coverage NOT_MEASURED. |
| ECG | Record list NOT_REACHED; C1 coverage NOT_MEASURED. |
| ECHO | Record list NOT_REACHED; C1 coverage NOT_MEASURED. |

The pre-existing M2-A census remains separate dated evidence and is not substituted for this incomplete execution. No new cross-table union was computed because the final coverage phase was not reached.

## Raw time and information completeness

The attached `clock_completeness.csv` reports each retained clock independently by table and split: denominator, empty/null, date-pattern-only, parseable timestamp and other nonempty strings. Date-pattern classification is not calendar validation. CXR StudyDate/StudyTime are separate components; a nonempty time component is not an invalid timestamp, and date/time assembly remains unverified. A present storetime or verifiedtime is not proof of cutoff-safe clinical availability. In particular, raw microbiology `chartdate`/`storedate` strings parse as timestamps, but those DATE-named source fields still do not establish subday clinical precision or observed availability. The table classifies string syntax only; date-field semantics must be retained independently, including any midnight-formatted dates. No clock assembly or precision upgrade was performed.

| Dataset/table.field | Denominator | Missing | Date-only strings | Parseable timestamp strings | Other nonempty |
|---|---:|---:|---:|---:|---:|
| hosp/admissions.admittime | 36,569 | 0 | 0 | 36,569 | 0 |
| hosp/admissions.deathtime | 36,569 | 35,782 | 0 | 787 | 0 |
| hosp/admissions.dischtime | 36,569 | 0 | 0 | 36,569 | 0 |
| hosp/emar.charttime | 2,293,666 | 0 | 0 | 2,293,666 | 0 |
| hosp/emar.storetime | 2,293,666 | 0 | 0 | 2,293,666 | 0 |
| hosp/labevents.charttime | 9,781,921 | 0 | 0 | 9,781,921 | 0 |
| hosp/labevents.storetime | 9,781,921 | 148,894 | 0 | 9,633,027 | 0 |
| hosp/microbiologyevents.chartdate | 260,778 | 0 | 0 | 260,778 | 0 |
| hosp/microbiologyevents.charttime | 260,778 | 20,263 | 0 | 240,515 | 0 |
| hosp/microbiologyevents.storedate | 260,778 | 958 | 0 | 259,820 | 0 |
| hosp/microbiologyevents.storetime | 260,778 | 1,998 | 0 | 258,780 | 0 |
| hosp/pharmacy.entertime | 1,178,787 | 0 | 0 | 1,178,787 | 0 |
| hosp/pharmacy.starttime | 1,178,787 | 1,424 | 0 | 1,177,363 | 0 |
| hosp/pharmacy.stoptime | 1,178,787 | 6,046 | 0 | 1,172,741 | 0 |
| hosp/pharmacy.verifiedtime | 1,178,787 | 346 | 0 | 1,178,441 | 0 |
| hosp/prescriptions.starttime | 1,338,360 | 1,424 | 0 | 1,336,936 | 0 |
| hosp/prescriptions.stoptime | 1,338,360 | 1,971 | 0 | 1,336,389 | 0 |
| hosp/procedures_icd.chartdate | 57,381 | 0 | 57,381 | 0 | 0 |
| hosp/services.transfertime | 39,742 | 0 | 0 | 39,742 | 0 |
| hosp/transfers.intime | 146,949 | 0 | 0 | 146,949 | 0 |
| hosp/transfers.outtime | 146,949 | 36,569 | 0 | 110,380 | 0 |

Indexed tables without retained raw clock fields: `hosp/diagnoses_icd`, `hosp/patients`. No occurrence or availability proxies were invented. Missing death time in admissions normally includes survivors and is not itself a data defect.


## Measured mapping and quality observations

- Lab source coverage is14,763/15,000 patients (98.42%);9,781,921 raw lab rows are not9,781,921 validated numeric targets.
- Lab `storetime` is absent in148,894/9,781,921 rows (1.522%); `hadm_id` is absent in4,460,269/9,781,921 (45.597%). No cross-encounter assignment was guessed.
- Lab `valuenum` is absent in1,339,225/9,781,921 rows (13.691%); `valueuom` is absent in1,443,946/9,781,921 (14.761%). Nonnumeric labs and unitless results require item-specific decisions rather than blanket exclusion or numeric substitution.
- Microbiology `charttime` is absent in20,263/260,778 rows (7.770%); `storetime` is absent in1,998/260,778 (0.766%). Date-named companion fields were preserved without conversion into observed precise clocks.
- Hospital diagnoses have no retained source event clock;57,381/57,381 procedure rows (100%) have date-only `chartdate` syntax. They were not assigned midnight/discharge proxies.
- Pharmacy and prescriptions each report13,172 contributing patients but contain1,178,787 and1,338,360 rows respectively. Equal patient counts do not establish identical patient sets, record-level equivalence or safe deduplication.
- Transfer `outtime` is absent in36,569/146,949 rows; this is an observed source pattern, not a corrected encounter interval. Its semantics remain pending.

## Outstanding mapping and quality decisions

- Canonical category/modality counts, rare-category supports and ≥64-target-qualified patient counts: **NOT_COMPUTED**. Source-index success does not pass these requirements.
- Encounter matching, ED/ICU/hospital hierarchy reconciliation, duplicate records/repeated studies and cross-source medication reconciliation: **PENDING**. Missing hadm_id can reflect unlinked/outpatient records; it is not automatically invalid source data.
- Numeric item classification, units/harmonization, vital aggregation span/stride and contributor timing: **PENDING**. Unit missingness is measured in `field_completeness.csv`; clinical validity is not established by a nonempty unit string.
- Date-only/absent diagnosis, procedure and triage clocks remain unresolved, with their source rows retained. No discharge-time, midnight or ED-boundary clock was inserted.
- ECG/ECHO encounter/timing/content checks, CXR metadata linkage and image validity, note pooling/truncation, OTIS position policy and production ECHO preprocessing remain separate dependencies. Unknown prospective availability remains unknown.
- M2 readiness remains **HOLD**. No model-ready corpus, next-stage budget or scientific-method selection is implied.

## Reproducibility, resources and preservation

The fixed patient split hash is `5df6b9e96b1921b94867cdacd04440de25a42e84d33c4a6fbd9ad5e26277d869`. Every completed Parquet index retains global CSV record ordinal, patient split, selected raw source coordinates/structured fields and text hashes. These are selected-field provenance indexes, not complete source-table copies; discarded columns remain recoverable from unchanged source records. Source files are hashed before and after indexing; only matching, fully written and hashed files lose their `.partial` suffix. Dictionaries are hash-verified, not converted into clinical events.

Slurm allocation was2 CPUs/16GiB, queue146s, elapsed348s (5min48s), total CPU273.173s. Scheduler batch MaxRSS was16,776,692KiB (15.9995GiB); process `ru_maxrss` was8,593,448KiB (8.1954GiB). These distinct accounting measures are both retained; the process value must not replace scheduler usage. Index namespace size after termination was190,017,758 bytes (181.21MiB), far below8GiB. The database failed at its own8GB limit; no memory setting, job allocation or budget was raised.

Controller elapsed: 324.13s. Process peak RSS: 8,593,448KiB. Slurm accounting, queue time, allocation and final output sizes are in `resources.json`; process RSS and scheduler memory accounting must be reported separately. Approved caps were2 CPUs,16GiB,60min,8GiB new output, with no GPU. Spill has a1GiB sublimit and output files use an OS size limit with metadata reservation. Partial artifacts are retained on failure; no automatic extension is permitted.

Prior-census replay checks: 30 comparable split/table rows; all source counts and patient coverages match: `True`. This is an independent aggregate consistency check, not proof of event or availability fidelity. Preflight verification passed108 tests. Preservation verification is attached separately; historical source, configurations, checkpoints, predictions and results remain in their original namespaces.

Clinical records, note text, Parquet rows, media, embeddings and model weights stay private. The public bundle contains only source hashes, schema names, aggregate counts, implementation identity and measured execution status. The next action is researcher review of these observations before choosing any preprocessing strategy or budget.

The one-shot approval is consumed: the current configuration has `execution_authorized=false` and records job57006471, while the immutable executed configuration retains its original approved value/hash. A bounded streaming/batched source-index writer preserving the same row/field identity is a possible future engineering option; it has not been implemented, budgeted or run. Do not infer that more RAM or another submission is authorized.
