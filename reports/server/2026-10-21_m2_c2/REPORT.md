# M2-C2 executed canonical-preparation report

Executed 2026-10-21, America/New_York. **FAILED / INCOMPLETE; M2 readiness HOLD.** The single authorized CPU job was consumed. No retry, GPU inference, model training, clinical evaluation or alpha experiment ran.

## Execution identity and verification

Slurm job `57053580` executed clean source `c381ca4296432c8a204968114cb346582cdb832e` using frozen configuration SHA-256 `1ba6f6dffb3c0b953975cd9aff3d266db42d04508e84e57eda0969d675f750d0`. The scientific recipe digest remains `0490bec6a1999989890f218f68b398ba396881ad63338e2d8c988300f196be28`. Source C1 manifest SHA-256 is `474b8f31758f22c12af99ed6bd92fcb36e40d45782cb6b7a62abf753c51d89dd`; patient-split SHA-256 is `5df6b9e96b1921b94867cdacd04440de25a42e84d33c4a6fbd9ad5e26277d869`.

The [preflight](preflight.json) passed 132 tests in 29.58s, bounded genuine-source checks across all 28 table paths, source/dictionary/split verification and all four pretrained weight hashes. Its dirty pre-commit source hashes bind the subsequently committed execution source; the actual job manifest records clean `c381ca4`. The preserved run namespace is `artifacts/reduced_pilot/m2/m2-c2-canonical-20261020-01` relative to this clean worktree. Its name is an identifier, not its execution date.

Submission command: `sbatch sbatch/m2_c2.sh`. The script used isolated Python, 4 CPUs, 32GiB, 120 minutes, no GPU/requeue, and an external 7,180s timeout. Final accounting was read using `sacct -j 57053580 --format=JobID,State,ExitCode,Start,End,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS -P`.

The original terminal receipt, private exception evidence, captured source/configuration and all partial outputs remain unchanged. [Preservation verification](preservation.json) confirms all 32 M1, 105 M1.5 and 37 original-C1 artifact hashes. A separate private inventory freezes 970 failed-C2 files, inventory SHA-256 `186252021626fe87fecebc4473e831784d6921192da3e6eea258537343a87b09`. Original failed C1 and C1-recovery job receipts remain failed; accepted independent C1 validation is not relabeled as job success.

## Failure and targeted fix

At `SOURCE:icu.outputevents`, `c2_io.Shards.flush` successfully published an event shard and receipt, then enumerated the directory to count bytes. A transient NFS `.nfs` unlink tombstone disappeared before its `stat`, raising `FileNotFoundError`. This is a confirmed post-publication bookkeeping defect, not a missing clinical source or invalid measurement. No OOM or time/storage-limit termination was recorded.

The fix counts only the two explicitly named files before publication and clears the committed buffer before invoking accounting callbacks. Missing required files still fail. Queue export now publishes atomic files and checks/accounts each file independently, rather than postponing aggregate output accounting until all exports finish. These are engineering changes; no mapping, architecture, loss, unit, timestamp or scientific recipe changed. Production authorization is now explicitly disabled and identifies the consumed job.

The original receipt reports an uncommitted 20,000-event buffer because buffer clearing followed the failed callback. Those events are already in the published shard total; they must not be counted again or subtracted as lost.

[Post-failure verification](postfailure-verification.json) binds the fixed source hashes to [136 passing tests in 30.94s](regression.txt), including NFS disappearance, interrupted post-publication accounting, queue budget interruption, and incomplete-C2 GPU-proposal rejection. Fixtures ran in the existing development allocation, not another bulk job. The isolated command was `srun --jobid=57035083 --overlap --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=threads --mem=8G --time=00:05:00 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/run.py test -q`.

A read-only private verifier (`m2-c2-failure-verification-20261021-01/finalize_partial_report.py`, source hash in the [summary](summary.json)) checked all 459 published shard hashes/counts, SQLite integrity, eight independently computed numeric vectors and their original C1 source ordinals/values/units/times/patient/encounter/split identities. It passed in 34.19s. No additional canonical processing occurred. Earlier private verification-script errors remain preserved; this PASS concerns preserved partial artifacts only, not full C2 or all source-row values.

## Actual retained outputs

| Measure | Actual result |
|---|---:|
| Completely processed patient-associated source tables | 22 / 28 |
| Rows from complete sources, including auxiliary descriptions | 7,370,315 |
| Published candidate canonical events | 4,910,497 |
| Candidates from the 22 complete sources | 4,810,497 |
| Additional published partial ICU output candidates | 100,000 |
| Explicit deferred inventory entries | 4,039,138 |
| Published ICU vital aggregation inputs | 0; chartevents not reached |
| Final window plans and owned-target eligibility | NOT_COMPUTED |

[Completed-source counts](completed-source-counts.csv), [category counts](partial-category-counts.csv) and [modality counts](partial-modality-counts.csv) give actual partial frequencies and contributing-patient counts. Specific-item frequencies, including native units and patient support, remain in private `analysis-v2/partial-support-specific_item.parquet`; they are not uploaded as clinical item strings. Deferred entries are not necessarily distinct source rows: a row can produce several candidates or deferrals. Full raw-source totals are not model-target totals.

| Split | Assigned | With published candidates | With at least 64 partial candidates | Feature-complete events | Patients with at least 64 valid owned targets |
|---|---:|---:|---:|---:|---|
| Train | 10,500 | 10,500 | 6,542 | 99,108 | NOT_COMPUTED |
| Validation | 2,250 | 2,250 | 1,438 | 21,024 | NOT_COMPUTED |
| Test | 2,250 | 2,250 | 1,438 | 21,121 | NOT_COMPUTED |

Feature completeness of an individual event does not establish complete context or unique shifted loss ownership. In particular, **6,542 is not the number of qualified training patients**. No full-cohort ≥64-valid-target count is available. No patient is removed for missing modalities. No future-density eligibility is used for a landmark cohort.

CXR has 27,156 timed candidates and 4,977 explicit mapping/parser deferrals among 32,133 indexed images. The latter expose a fractional StudyTime parsing limitation; they are not evidence of invalid source records and remain deferred under the frozen recipe. Retrospective/untimed diagnoses, medication source reconciliation and observed-vs-unknown availability remain explicit. See sanitized reason groups in the summary; raw reasons and source identities remain private.

## Encoding readiness and conditional GPU budget

| Preserved deduplicated task kind | Tasks | Readiness |
|---|---:|---|
| CXR | 27,145 | SQLite task inventory and event bindings retained; no final queue export |
| ECG | 65,137 | Same; actual waveform identities pinned |
| ECHO | 48,130 | Same; DICOM records do not imply valid clips or broad patient coverage |
| Specific-item text | 8,001 | Partial; remaining numeric/note sources can add items |
| Other text | 259 | Partial microbiology/inline content; **not** the clinical-note queue |

Clinical discharge/radiology note preparation was not reached. No complete six-modality queue manifest or model-ready dataset exists. The four already exercised encoder implementations/weight recipes remain available; this failure does not require another encoder-feasibility investigation.

The [conditional GPU ceiling](gpu-budget-conditional.json) is **one L40S 48GiB GPU, 16 CPUs, 64GiB host RAM, 24h wall time, 48GiB new output including at most 8GiB temporary clips**, no retry/requeue/training/alpha. Proposed batches are text/specifics 64, CXR 8, ECG/ECHO 4, with eight one-thread CPU preprocessing workers. Phase caps are text/specifics 3h, CXR 3h, ECG 3h, ECHO 14h and verification 1h. Existing two-clip measurements imply approximately 67–90 serial CPU hours for 48,130 ECHO tasks, or an idealized 8.4–11.2h with eight workers. This is conditional on two B-mode clips from one patient, not a cohort throughput guarantee. Other bulk throughput is unmeasured.

This is an explicit resource ceiling, **not a submission-ready full-queue proposal or completion prediction**: note tasks and final hashed queues are missing. `m2/gpu_plan.py` refuses incomplete C2, derives actual counts/hashes from completed materialized queues, and never submits or performs inference. `tools/m2_gpu_plan.py` and the successful-run-only `tools/m2_c2_verify.py` were not executed on the failed run. Any future encoding must recheck exact content/checkpoint/preprocessing identities, reuse only verified clean caches, preserve failed task bindings without zero substitutes, and save partial state at hard caps. GPU authorization remains required after CPU recovery.

## Resources and next dependency

[Actual accounting](resources.json): Slurm FAILED/exit 1:0, 00:22:02–00:56:47, 34m45s wall, 16m10.591s total CPU, 4 CPUs/32GiB/no GPU. Controller peak RSS is 1,090,608KiB; Slurm batch MaxRSS separately reports 33,553,872KiB, near the allocation. These scopes must not be conflated into a claim of memory headroom. Retained logical files total 723,185,583 bytes (approximately 0.674GiB); the terminal controller's pre-receipt accounting is 723,182,054 bytes. No budget was extended.

A targeted CPU recovery needs **new authorization**. Reuse verified complete-source shards, deferrals and content queues without repeating C1 indexing or hashing/encoding media unnecessarily. Retain the immutable failed run as a parent, exclude its 100,000 partial ICU-output events from a new final dataset, and replay that one source in full to make ownership unambiguous. The six remaining inputs are ICU outputevents (394,737 rows), procedureevents (62,847), discharge notes (28,528), radiology notes (181,245), labs (9,781,921) and chartevents (28,806,356): 39,255,634 source rows when outputevents is replayed in full. They are not predicted canonical-target counts.

A recovery resource proposal is capped at 4 CPUs/32GiB/120min/48GiB new output, no GPU/requeue/retry, with a separate namespace and integrity status at any cap. It is not submitted or authorized. A source-bound reuse/resume implementation and inexpensive interruption/ownership checks must precede any new submission; this report does not claim those recovery operations already exist.

After completed canonical preparation and genuine encoding, regenerate context-valid target eligibility before proposing training. High-impact scientific candidates remain native-unit/Fourier conventions, incomplete medication reconciliation, retrospective/unknown availability, equal-time serialization and preprocessing/coverage sensitivity. None is silently promoted to author fidelity or strict prospective evidence. M1/M1.5/C1/legacy remain frozen; M2 readiness remains HOLD.
