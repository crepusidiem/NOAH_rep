# C1 recovery: bounded source indexing implementation and verification

Executed2026-10-20 in `output/worktrees/clean-baseline`, branch `development/clean-baseline`. Generating HEAD was `1d5b408` with new source captured privately; source/core hashes accompany the verification artifacts. Only inexpensive fixtures and bounded source-prefix checks ran in existing Slurm allocation56795452 (each step2 CPUs/4GiB). **No new full indexing job was submitted.** The proposed recovery configuration is disabled. Accepted C1 indexes/failure evidence, M1/M1.5, source data and the15,000-patient split remain separate and unchanged.

## Targeted memory investigation

The accepted failure was DuckDB's8GB engine allocation failure during the monolithic ICU indexing path. That execution did not record a native operator stack or separate its schema/COPY/diagnostic stages. The allocation site cannot be reconstructed conclusively from its exception message alone.

The new real-source schema check succeeds at a512MB DuckDB limit in0.146s; EXPLAIN completes in0.334s total. The [physical plan](legacy-plan.txt) shows `STREAMING_WINDOW` row numbering and a hash join with the15,000-row cohort, not a full-table sort. The instrumented legacy schema/COPY/post-write path also passes on the500,000-record real prefix at256MB in4.79s. These observations do **not** reproduce or identify the exact historical full-scale allocator failure. They rule out claiming that a proven global sort or wrong join build side caused it.

The replaced operation is the whole-source DuckDB CSV/window/join-to-Parquet COPY path. `m2/stream_index.py:stream_source` instead uses Python's streaming CSV reader, exact dictionary membership and explicitly bounded Arrow/Parquet chunks. No complete source table is materialized, and the streaming path does not allocate DuckDB buffers. It also removes heavyweight model imports from indexing. Legacy-process RSS includes those imports and is not an isolated measurement of a COPY algorithm's memory.

## Implemented source and resume contract

| Requirement | Implementation and verification |
|---|---|
| Exact source identity/filter | Full source SHA-256 before/after, dataset/version/path/header, global1-based **CSV record** ordinal (multiline-aware), full parsed-row hash and split-map digest. Raw subject/encounter fields remain strings; cohort filtering uses exact integer IDs. Source duplicates retain different ordinals. |
| Bounded memory | Commit after250,000 source records,25,000 selected records or16MiB selected serialized payload, whichever comes first. Arrow row groups4096. Record/field size bounds fail explicitly rather than truncate/drop. Verified output is read back in1024-row batches. |
| Atomic chunks | Write a temporary directory, close/fsync Parquet, hash and independently read/validate contents, write/fsync receipt, then publish the directory by atomic rename. Data and receipt become read-only. Incomplete temporary directories are quarantined, never counted as committed data. |
| Exact resume | Single-writer lock; source/split/code/recipe identities must match. Check every chunk's file/content digest, order, range, counts and coverage. Reparse gzip prefixes and verify source-prefix digests; do not rewrite committed chunks. Reject changed sources/splits/recipes, missing ranges, tampering or completion mismatches. Committed directories are the recovery ledger, so a crash after publication cannot lose ownership. |
| Failure/caps | Preserve verified/provisional chunks and failure receipts; complete-source status requires EOF and matching source hashes. Resource failures stop without automatic extension/retry. Ordinary source failures retain a distinct failure receipt; expensive ICU runs last. Complete source indexing still means readiness HOLD. |
| Independent references | Raw-field/ordinal/split/hash comparison with a separate DuckDB parser on real prefixes. Proposed full execution checks exact split counts **and** patient coverage against the accepted prior whole-source census. The small auxiliary CXR metadata table gets an independent raw SQL census; bounded first-chunk SQL checks use256MB. A mismatch blocks acceptance. |
| Reuse | Reference the11 accepted hospital indexes and3 dictionary receipts in place, rechecking their existing artifact hashes and source stat fingerprints. No source rows are rebuilt or large files copied. Reused raw-source hashes inherit the accepted C1 evidence; they are not falsely described as freshly rehashed source tables. |

All raw source columns are preserved as strings except the existing explicitly declared hash-only text payloads. Text remains recoverable through source identity and record ordinal; no embedding is computed. This recipe preserves **parsed CSV values**, not original quoting/line-ending bytes; the complete file hash retains byte identity. Source identity belongs to each chunk receipt and must accompany its rows.

Missing `hadm_id` remains empty/unknown encounter attribution. No encounter join is guessed. Each raw clock remains unchanged and gets a precision column; DATE-named fields stay date precision even when formatted with midnight. Raw store/verification/entry clocks are kept separately from an explicit unknown observation state. `raw_clock_present` is not certification of clinical availability; no generic prospective available-at clock is fabricated. Pharmacy/prescriptions/eMAR identities remain separate. The accepted C1 indexes receive precision/provenance overlays through references, without rewriting their bytes. These are source-index policies, not approved canonical mappings.

## Executed verification

The complete isolated suite passes **119 tests in24.62s**, including11 targeted streaming/controller tests. [Executed log](regression.txt) and [verification summary](verification.json) are attached. Fixtures exercise multiline ordinals, zero/units/raw values, absent encounters, date precision, unknown/delayed clocks, all splits, empty/chunk boundaries, clinical duplicates, injected failures before write/before publication/after publication, exact resume and reference mismatch rejection. A fictional end-to-end controller test completes both CXR inputs and preserves them when ICU fails.

| Genuine source-prefix check | Parsed source records | Selected cohort rows | Source-patient support | Result |
|---|---:|---:|---|---|
| ICU chartevents small | 50,000 | 458 | 1 train | Independent field/ordinal/split/count/coverage and resume PASS |
| ICU chartevents larger | 500,000 | 127,725 | 5 train/3 test | Same checks PASS; no validation patient represented in this prefix |
| CXR record list | 5,000 | 360 | 36 train/8 val/9 test | Independent raw-field reference PASS |
| CXR metadata | 5,000 | 360 | 36 train/8 val/9 test | Independent raw-field reference PASS; not additional images |
| ECG record list | 5,000 | 402 | 44 train/6 val/14 test | Independent raw-field reference PASS |
| ECHO record list | 5,000 | 100 | 1 train | Independent raw-field reference PASS; limited patient coverage |
| Discharge notes | 500 | 46 | 13 train/3 val/4 test | Raw coordinates and independent text hashes PASS |
| Radiology notes | 500 | 63 | 3 train/4 val/1 test | Raw coordinates and independent text hashes PASS |

All prefixes start at the first original CSV record; no outcome, density or all-modality filter was used. Original files were read only and retain matching stat fingerprints. The temporary prefix copies have their own full hashes, **not** purported full-original-source hashes. Their EOF completion statuses apply only to those explicitly named copies. No population counts, valid events or ≥64-target eligibility are inferred from these checks.

The50k/500k streaming subprocesses peak at134,404/208,916KiB RSS (about131/204MiB) while input grows10-fold and selected rows grow279-fold. With the actual proposed production chunk limits, the same500k prefix produces6 chunks, exactly matches the independently verified30-chunk layout, preserves chunks across resume, and peaks at204,936KiB (about200MiB), elapsed7.10s including recovery. Six remaining-source prefix checks take17.58s, peak223,436KiB. These are bounded memory observations, not a guarantee for every possible source record or full-corpus throughput.

## Remaining indexing plan

`m2/c1_recovery.py:recovery_plan`, `configs/m2_c1_recovery.json`, `tools/m2_recover.py` and `sbatch/m2_c1_recovery.sh` implement the prepared route. [Exact source plan](source_plan.json):18 remaining inputs, including the ICU item dictionary. Each source has its own chunks/completion/failure state.

1. CXR record/metadata indexes, ECG and ECHO metadata.
2. Six ED tables.
3. ICU item dictionary and small ICU tables: stays, input, output, procedures.
4. Discharge/radiology notes, with hash-only bodies.
5. ICU chartevents last.

No pixel/waveform/video decoding or encoding occurs. Accepted prior references total31,033,987 selected source rows across the remaining original census tables, plus the separate CXR metadata reference; these are **expected raw-row references**, not newly executed indexes, unique events or eligible patients. An ICU failure cannot block smaller sources already finalized. Storage/deadline failures stop the execution rather than proceeding beyond budget.

## Proposed one-execution budget: not authorized

| Resource | Proposal |
|---|---|
| Slurm jobs | One CPU job, no GPU, no requeue or automatic retry |
| CPU / RAM | 2 CPUs / **8GiB** RAM; no assumption that all allocated memory belongs to an engine |
| DuckDB | **256MB** for bounded independent reference checks; absent from the streaming writer |
| Wall time | Hard60min; controller deadline3530s, external timeout3550s, kill grace10s |
| New output | Hard8GiB, including partial/quarantined chunks;64MiB metadata reserve. Existing181MiB C1 data referenced, not duplicated. |
| Planning range | Conditional35–55min,1–4GiB output; completion within the cap is not guaranteed |

The production-prefix chunks use5,699,629 bytes for127,725 rows (44.62 bytes/row); applying that composition to31.03M reference rows suggests1.29GiB, **conditional only**. Patient ordering, note widths, total unselected rows, gzip decompression and filesystem latency differ. Runtime includes source hashing, independent checks and durable publication. The full source-record count is not inferred from a compressed-file planner estimate. On cap/failure, retain the explicit incomplete status and verified ledger; resuming requires a new approval/reference and an explicit resume directory. No extra submission is automatic.

`execution_authorized=false` and a missing approval reference block the real controller before source scans/output creation. The accepted C1 authorization remains consumed and untouched. Current source hashes, immutable evidence inventory and proposed limits are attached. M2 readiness stays **HOLD** even after successful source indexing.

After C1 completion, the next task is canonical event construction: approved category/specifics/modality mappings, longitudinal versus encounter-attributed views, exact occurrence/availability/precision, valid scored targets and ≥64-target patient counts, with category/modality contributing-patient supports. Those counts and six-modality content integration remain pending. No canonical strategy, clinical evaluation, model training, alpha experiment or bulk encoding was selected or started here. Existing GPU Slurm scripts remain the required route for later separately approved heavy encoder work.
