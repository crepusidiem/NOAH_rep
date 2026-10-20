# Readmission interval audit: MIMIC-IV 2.2 source, ETL and label impact

Executed 2026-10-19 (America/New_York), in existing O2 allocation56795452 on `compute-b-16-177.o2.rc.hms.harvard.edu`. Baseline commit: `1d21fd6f85d4fabbf6cdf0022e865db82cc52dc7`, with the existing dirty tree preserved. This was a read-only investigation of source records and existing artifacts. New outputs are audit artifacts only; no production labels, model/training code, checkpoints or historical metrics were changed. No training, model inference, encoder extraction or new Slurm jobs ran.

The negative gaps are present in the source hospital admissions table. They are rare and do **not** change the historical test cohort's subject-level labels under the interval repair examined here. The more consequential unresolved problem is the absence of an index encounter in those historical labels. Adding a lower bound to the immediate-successor expression alone is insufficient and creates a missed positive in the pilot validation cohort.

## Evidence and reproducibility

Primary artifacts:

- [summary.json](summary.json): source paths/versions/bytes/SHA-256, runtime, full frequency and comparison tables, ETL checks and read-only verification.
- [final_sensitivity_summary.json](final_sensitivity_summary.json): explicit discharge-eligibility, admission-type and encounter-attribution sensitivities. These are **unapproved diagnostic alternatives**, not replacement clinical labels.
- [queries.sql](queries.sql), [source_overlap_checks.sql](source_overlap_checks.sql), [source_overlap_checks.json](source_overlap_checks.json), [type_filter_sensitivity.sql](type_filter_sensitivity.sql): exact SQL and additional source diagnostics.
- [source_lineage.json](source_lineage.json): function AST/file identities at HEAD, pre-remediation snapshot and current tree.
- [overlap_context_summary.json](overlap_context_summary.json), [overlap_context_checks.sql](overlap_context_checks.sql) and [overlap_context_checks.py](overlap_context_checks.py): nonadjacent-overlap checks, exact test-population equality and attribution of subject-label changes.
- [audit_readmission_intervals.py](../../../scripts/audit_readmission_intervals.py) and [audit_readmission_sensitivities.py](../../../scripts/audit_readmission_sensitivities.py): executable extraction, comparisons and assertions. The primary NumPy successor search agreed with an independent SQL self-join for every source admission.

Private derived records remain under ignored `output/readmission_audit/2026-10-19/`: all negative pairs; all encounter-keyed diagnostic labels; linked ED/ICU/transfer records for involved admissions; represented test IDs; 41 representative cases in CSV and human-readable timelines; protected-file hashes. They must not be committed or uploaded. The private [representative timelines](../../../output/readmission_audit/2026-10-19/representative_timelines.md) contain subject IDs, both hospital encounter IDs, both admission/discharge timestamps, both admission types, gaps and the exact pairing transformation for each case. The [case CSV](../../../output/readmission_audit/2026-10-19/representative_cases.csv) also records the later non-overlapping candidate when present.

Executed commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python scripts/audit_readmission_intervals.py --inspect-test-shards
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python scripts/audit_readmission_sensitivities.py
PYTHONDONTWRITEBYTECODE=1 /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python reports/server/2026-10-19_readmission_intervals/overlap_context_checks.py
```

The primary run completed in166.81s. Outputs refuse to overwrite an executed summary. For a repeat, pass fresh `--private-output`/`--public-output` to the first script and matching `--private-input`/`--public-output` to the second. Only aggregate outputs belong in the public report directory. The additional source SQL operates on the same in-memory `admissions` relation created by the first script. Function lineage was checked by hashing `ast.dump(function_node, include_attributes=False)` from `git show HEAD:<path>`, the preserved pre-remediation source and current source.

Source: `/home/wey371/rajpurkarlab/datasets/mimic-iv/physionet.org/files/mimiciv/2.2/hosp/admissions.csv`,73,099,499 bytes, SHA-256 `59a99ad5a5549f1ffe2e4995caafb8f727a1b726e595737f3aeb43eee6bb32ac`. Companion tables are hospital `transfers.csv.gz`, ICU `icustays.csv.gz`, and MIMIC-IV-ED2.2 `edstays.csv`; their full identities are in `summary.json`.

The source contains431,231 rows,431,231 unique hospital admissions and180,733 subjects. There are zero duplicate hospital IDs, zero exact duplicate rows, zero missing admission times, zero missing discharge times,139 admissions with discharge before their **own** admission and3 zero-duration admissions. These139 records are distinct from the59 adjacent negative gaps; they are excluded from valid-index diagnostic comparisons, not repaired in source.

## Frequency and interval distribution

An adjacent pair means the next row for the **same subject** ordered by hospital `admittime`, with `hadm_id` added only as a deterministic tie-break. A potentially misclassified index stay is the earlier row of a negative pair; it is not necessarily a changed final label.

| Quantity | Entire MIMIC-IV2.2 hospital source | Pilot manifest | Represented test patients |
|---|---:|---:|---:|
| Subjects with a negative pair / all subjects |59/180,733 (0.03264%)|1/1,000 (0.10000%)|0/146 (0%)|
| Distinct admissions participating in negative pairs / admissions |118/431,231 (0.02736%)|2/4,129 (0.04844%)|0/517 (0%)|
| Negative pairs / adjacent pairs |59/250,498 (0.02355%)|1/3,129 (0.03196%)|0/371 (0%)|
| Potentially affected index stays / valid timestamp index stays |59/431,092 (0.01369%)|1/4,127 (0.02423%)|0/517 (0%)|

The entire pilot's one negative pair is in **validation**:1/150 subjects,2/613 admissions and1/463 adjacent pairs. Training has0/700 affected subjects; manifest test has0/150. All three existing test-shard variants contain the same146 represented subjects and reproduce46 positive subject-level labels. Exact subject-list equality was verified in `overlap_context_summary.json`; the lists remain private.

These are **adjacent-pair** counts. A full interval self-join additionally finds one nonadjacent overlap:60 overlapping hospital-account pairs involving119 admissions in the same59 subjects. Do not use118 as the denominator for every account participating in any overlap. The negative-gap frequency table intentionally retains the59 adjacent pairs relevant to the historical `LEAD` expression.

Signed gaps range from **-713.55h to -0.03333h**:29.73 days of overlap down to2 minutes. Median=-11.56667h; signed quartiles=-23.24167h and-1.8h. Equivalently, median overlap magnitude is11h34m and its interquartile range is1.8–23.24167h. The pilot overlap is10h59m.

| Absolute overlap magnitude | Pairs /59 | Percentage |
|---|---:|---:|
| >0 to1h |13|22.03%|
| >1 to6h |10|16.95%|
| >6 to24h |22|37.29%|
| >24 to72h |8|13.56%|
| >72h |6|10.17%|

## Root causes and source-to-transformation trace

| Candidate cause | Evidence and conclusion |
|---|---|
| Distinct overlapping hospital accounts | **Confirmed source structure:** all59 pairs use distinct `hadm_id`s already present in `hosp.admissions`. There are36 crossing overlaps,22 nested successors and1 identical-start pair. “Overlap” describes the ADT timestamps; it does not establish simultaneous physical hospitalization or an invalid source record. |
| ED/ICU/hospital hierarchy confusion | **Not the source of these label pairings.** The historical outcome query uses only hospital admissions and a unique subject filter. Linked records include35 ED rows for34 involved hospital IDs,19 ICU rows for18 IDs and360 transfer rows for all118 IDs. All matched subject/hospital relationships agree. ED/ICU rows are separate children, not additional rows in the readmission successor sequence. |
| Duplicate rows or join multiplication | **Not observed:** zero duplicate `hadm_id`s, zero exact duplicate admission rows, unique pilot manifest subject IDs; no child-table join in the historical readmission CTE. Multiple valid stays for one subject are not duplicates. |
| Incorrect chronological sorting | The historical SQL orders by timestamp, not random hospital ID. One subject has tied admission times; the deterministic tie-break and literal historical query produced zero successor differences in this run and zero subject-label differences. Tie behavior remains unspecified and must not be interpreted clinically. There are125,006 descending-ID chronological pairs, demonstrating why ID sorting would be inappropriate, but that is not what this code does. |
| Erroneous cross-encounter joins | **No such join introduces the59 gaps.** The readmission CTE partitions by subject and reads hospital timestamps directly. A different confirmed error occurs later: subject-level `MAX` loses index identity and can assign another stay's positive outcome to the representation being evaluated. |
| Transfers, donation or other administrative workflows | **Clinical explanation unresolved.** Of the59 pairs,18 index rows have discharge disposition `DIED`;9 have hospital-expire flag1 and9 have flag0. There are21 missing index discharge dispositions. No pair has matching nonmissing death timestamps on both accounts. Disposition/admission-location combinations include psychiatric and acute-facility transitions, but they do not prove episode continuity or transfer status. No encounter was manually reclassified. |

Four of the59 successor records have discharge before their own admission. None of the59 index records has that property. Their administrative meaning is unresolved; a negative-duration successor is not evidence of a clinically completed new hospital stay. The remaining interval classifications describe the recorded timestamp geometry without declaring source data invalid. The optional exclusion table in `final_sensitivity_summary.json` excludes accounts at endpoints of adjacent negative pairs; it is not a complete all-overlap eligibility protocol.

The [official admissions documentation](https://mimic.mit.edu/docs/iv/modules/hosp/admissions.html) identifies `hadm_id` as the hospital-account key, distinguishes urgency/observation admission types, and cautions about organ-donor accounts with unusual durations. That contextual possibility does not prove which workflow produced any audited pair. The [MIMIC-IV2.2 release description](https://physionet.org/content/mimiciv/2.2/) states that one date offset is used per subject, preserving within-subject intervals; uniform date shifting cannot itself produce these overlaps. ED linkage is documented separately in [MIMIC-IV-ED2.2](https://physionet.org/content/mimic-iv-ed/2.2/).

The exact code path is:

1. Historical `scripts/evaluate_clinical_probes.py:extract_patient_labels` (HEAD line178; preserved pre-remediation line181) reads `hosp/admissions.csv` directly. `target_sids` filters patients, retaining **all** hospital admissions for each selected subject. `manifest_path` is unused in that function. The CTE computes `LEAD(admittime) OVER(PARTITION BY subject_id ORDER BY admittime)` and the outer query takes `MAX(gap_seconds<=30*86400)` by subject. Negative gaps satisfy that expression. No hospital event ETL, ED linkage, ICU linkage, pooled tensor or clinical representation determines the pairing.
2. The current function is blocked by the earlier remediation; it was not unblocked or executed for this audit. The historical query was replayed independently. Its function AST matches HEAD and the preserved pre-remediation copy; the current guard intentionally differs.
3. Separately, `src/data/duckdb_extractor.py:208`, `_extract_admissions_and_demographics`, retains hospital IDs and emits admission/discharge events with source times expressed as hours from a patient origin. Its function AST is identical at HEAD, pre-remediation and current source. Replaying only this method for all1,000 cohort subjects produced4,129 admission and4,129 discharge events, with **zero** identity/time discrepancies and **zero** adjacent-gap discrepancies relative to source. The one pilot negative gap is preserved, not introduced. This is not a full ETL re-execution.
4. `extract_all` interleaves all event categories in physical time. `ModelWindow` later drops `hadm_id` and source-row identity. Thus exact hospital-event lineage in old shards cannot be recovered from the tensors alone. Reading their subject IDs confirms the represented test population; it does not restore missing encounter provenance. The existing curation report's4,104 hospital events concerns975 eligible patients, whereas the replay's4,129 concerns all1,000 candidates.

## Encounter-specific diagnostic labels and label impact

The observed-record reference is keyed by `(subject_id, hadm_id)`. For each valid index stay, search **all subsequent distinct hospital admissions**, skip candidates before its discharge, and select the earliest remaining candidate. The diagnostic is positive when its gap is between0 and30 days inclusive. No admission-type or transfer exclusions are silently applied. Independent NumPy search and SQL self-join implementations agreed for every source row.

This is an encounter-specific **file-observation diagnostic**, not a fully identified clinical readmission endpoint: a zero means no qualifying return recorded in this file. It cannot establish complete30-day follow-up, survival, out-of-network admission absence or unplanned status. “FP/FN” below means disagreement relative to this declared diagnostic, not chart-adjudicated clinical truth.

| Fixed population and comparison | Old positives -> reference positives | FP / FN | Changed subjects | Prevalence |
|---|---:|---:|---:|---:|
| All431,092 valid timestamp stays: historical per-stay expression -> forward search |87,228 ->87,183|45 /0|45|20.23420% ->20.22376%|
| All422,554 flag0 valid discharge stays: same comparison |87,126 ->87,090|36 /0|36|20.61890% ->20.61038%|
| All422,392 conservative live candidates: same comparison |87,084 ->87,057|27 /0|27|20.61687% ->20.61048%|
| Same422,392: immediate `gap>=0` patch -> forward search |87,043 ->87,057|0 /14|14|20.60716% ->20.61048%|
| Pilot4,065 conservative live candidates: historical expression -> forward search |1,051 ->1,051|0 /0|0|25.85486% ->25.85486%|
| Same4,065: immediate lower-bound patch -> forward search |1,050 ->1,051|0 /1|1|25.83026% ->25.85486%|
| Test509 conservative live candidates: historical expression -> forward search |108 ->108|0 /0|0|21.21807% ->21.21807%|

Conservative live candidates additionally exclude `discharge_location='DIED'` and nonmissing in-hospital death time. This eligibility rule is a sensitivity requiring approval. Across source there are163 flag0/DIED-disposition rows, including3 in the pilot and0 in represented test. Missing/contradictory eligibility must not be resolved by a silent default.

The interval bug yields59 positive pair expressions, but only45 changed valid-stay labels:14 affected index stays have a genuine later recorded candidate within30 days and remain positive. The pilot's affected stay is one of those14: its immediate candidate overlaps by10h59m, while a later distinct admission starts17h14m after the index discharge. Simply rejecting the immediate candidate without searching farther misses that return.

At the **historical subject-any** aggregation level, the immediate lower-bound patch removes32/40,196 source positives. The forward reference instead removes35, from40,196 to40,161 across180,733 subjects (22.24054% ->22.22118%). These are not interchangeable counts:3 changes come from excluding invalid index durations and32 from repairing negative candidate timing on valid indexes. Ten of the139 negative-own-duration admissions have a positive legacy expression. Pilot subject-any labels remain347/1,000; represented test remains46/146 (31.50685%), with0 changed test patients. Therefore no change in the historical test readmission AUROC can be attributed to this interval repair alone. Predictions were not replayed and no corrected AUROC was estimated.

### Errors and sensitivities that a nonnegative-gap patch misses

**Encounter attribution is the confirmed broader error.** As a counterfactual diagnostic, assigning the existing subject-any label to every conservative live index stay yields2,766 positives among4,065 pilot stays, versus1,051 encounter-specific positives:1,715 mismatches across343 patients (68.04428% versus25.85486%). Among the represented test patients'509 eligible stays, the corresponding values are309 versus108, with201 mismatches across45 patients (60.70727% versus21.21807%). These are all-stay attribution comparisons, **not counts of mislabeled rows in a historical encounter dataset**, because that dataset did not exist: the old evaluator had one pooled vector and one label per subject.

Selecting the last recorded admission is not a valid generic remedy for readmission. It mechanically lacks a later recorded admission in this file; the resulting diagnostic is zero for every selected stay. That is selection on future record structure, not evidence that all patients were clinically readmission-free.

Other nonnegative pairs require explicit policies:

- Source contains1,736 exact-zero pairs and6,861 strictly positive same-calendar-day pairs; pilot has18 and51, respectively; represented test has2 and4. Treating all same-day events as transfers would discard actual positive-time returns without evidence. No adjacent pair is exactly30 days apart in this source.
- Among conservative live indexes, one **strictly positive** immediate gap still starts inside an ongoing overlap-connected episode;0 such cases occur in the pilot/test. The private [three-account timeline](../../../output/readmission_audit/2026-10-19/positive_gap_context_timelines.md) shows the earlier open account, the discharged index and its positive-gap successor. The broader episode sensitivity merges touching intervals too and therefore affects many zero-gap links. The primary summary's `positive_gap_index_in_active_overlap_episode` field includes zero gaps; it must not be read as a strictly-positive count. The sensitivity summary supplies the strict count.
- Under a sensitivity excluding `ELECTIVE` and `SURGICAL SAME DAY ADMISSION` successors, source positives decrease by5,430/87,057; pilot by56/1,051; test by8/108. Excluding all admission types containing `OBSERVATION` changes27,992,373 and41 positives respectively. Applying both changes33,977,431 and50. Each sensitivity searches all remaining candidates within the horizon, rather than deleting the first and stopping. These are **definition-sensitive differences**, not confirmed false positives: type alone is not a validated planned-admission classifier.
- The historical expression labels102 expired index stays positive across source, including1 pilot stay and0 represented test stays. A positive gap does not make an expired index an eligible live discharge. Donation/account handling and terminal-event policy need explicit review.

Full confusion matrices and affected-subject counts for each alternative are in the JSON artifacts. Denominators are fixed within each comparison; rates across different eligibility sets are not presented as one treatment effect.

## Representative cases

There are41 private cases, selected deterministically to include the pilot overlap, different geometry/type combinations, small and large overlaps, later-candidate recovery, planned/observation sensitivities, expired indexes, zero-gap boundaries and subject-max misattribution. The selection excludes already chosen records, so quantile-targeted examples are nearest unused cases rather than exact quantile observations.

For review, cases1 and35 demonstrate the lower-bound patch's false negative; cases19–28 show nested intervals; case29 shows tied starts; cases3 and14 contrast a2-hour and2-minute crossing overlap; case22 shows the largest overlap; cases37–38 show positive-gap type-policy sensitivities; case39 shows a positive gap after an expired index; case40 shows an exact-zero boundary; case41 shows a positive gap greater than30 days receiving a subject-level positive from another stay. Case36 is a zero-gap/touching-episode example, despite its original selection-reason identifier containing `positive_gap`. No duplicate-row or cross-subject-join examples are invented, because those causes were not observed.

## Minimal recommended fix and policy decisions

The implementation repair should preserve raw accounts and add an explicit index/candidate/label relation. It should **not** rewrite source timestamps, collapse accounts automatically or implement only `LEAD` plus a lower bound. The core candidate relation is:

```sql
-- Conceptual candidate relation; index eligibility and qualifying admission types
-- must come from an approved protocol. This is not a production label migration.
SELECT i.subject_id, i.hadm_id AS index_hadm_id,
       r.hadm_id AS candidate_hadm_id, i.dischtime, r.admittime,
       datediff('second', i.dischtime, r.admittime) AS gap_seconds
FROM approved_index_encounters i
JOIN approved_return_admissions r
  ON r.subject_id = i.subject_id
 AND r.hadm_id <> i.hadm_id
 AND r.admittime >= i.dischtime
 AND r.admittime <= i.dischtime + INTERVAL '30 days';
```

Retain one output row per eligible index even when this relation is empty. Record candidate identity/time, overlap flags, eligibility, observation status and label status. Do not substitute zero for unknown observation or an ineligible index. Independent clinical follow-up is needed before an empty candidate relation is interpreted as a confirmed negative. Simultaneous starts, zero-duration index stays and other ongoing accounts need explicit handling before using this relation clinically.

| Issue | Recommended policy for approval | Alternative / scientific consequence |
|---|---|---|
| Overlapping hospital accounts | Preserve source; flag all involved accounts; for an initial clinical primary analysis quarantine ambiguous indexes rather than labeling the overlap a readmission. Search later candidates in a separate diagnostic. | Merge continuous-care episodes using a running maximum discharge only after approving continuity/touching/transfer rules. This changes the index unit and horizon origin. |
| Transfers | Distinguish within-hospital child transfers from new hospital accounts; require a reviewed transfer/continuity definition for cross-account merging. Do not infer transfer from a negative gap alone. | Keep all accounts as paper event tokens while reporting a distinct clinical episode endpoint. A facility disposition alone may not identify the linked destination encounter. |
| Same-day returns | Use elapsed seconds; preserve positive-time same-day returns unless an approved continuity rule excludes them. Adjudicate exact-zero boundaries separately. | Zero-inclusive versus strictly-positive gaps, or an episode rule, change1,736 source pairs; arbitrary24h merging is a methodological change. |
| Missing or invalid discharge | Unknown/ineligible label with an explicit reason; never fabricate a discharge or force a negative. | The present source has0 missing discharges but139 negative own durations; their administrative meaning remains unresolved. |
| Mortality / conflicting disposition | Exclude confirmed expired indexes from a live-discharge task; quarantine contradictory eligibility markers; preregister competing-death handling during follow-up. | Using only hospital-expire flag0 retains163 DIED-disposition records; censoring and a composite death/readmission endpoint are different estimands. |
| Multiple admissions | Index-specific labels and a prospectively defined eligible index population; patient-grouped splits and uncertainty. Search all qualifying future candidates, then retain the earliest. | A single operationally defined index can be approved, but “ever positive” or “last recorded stay” cannot silently replace it. |
| Return admission type | Keep paper-event and clinically restricted analyses separate; approve exact observation, elective and planned-admission rules before clinical claims. | Excluding type strings is an informative sensitivity, not a validated unplanned-readmission algorithm. |
| Incomplete follow-up | Report observed-hospital return as such; use unknown/censored status when complete follow-up cannot be demonstrated. Never use the cross-patient maximum shifted date as a common observation cutoff. | Broader clinical readmission prediction requires an observation/ascertainment source and outside-system policy. |

Paper facts: NOAH p.22 specifies a30-day discharge reference for zero-shot Monte Carlo readmission, a prompt ending one timestep before discharge and25 rollouts. It does not operationalize the hospital-account overlap, transfer, eligibility, follow-up or admission-type rules investigated here. The last-stay linear-probe protocol on pp.23–24 concerns its listed probe outcomes; it is not authority for choosing the last recorded discharge for readmission. All policy recommendations above are assumptions requiring approval, not newly discovered paper requirements. The existing trained linear readmission probe is itself distinct from the paper's zero-shot task.

## Verification and limits

The primary audit asserted equality of NumPy versus independent SQL successor searches; literal historical versus deterministic subject labels; unique source IDs and unique manifest subjects; source versus admission-ETL identity/time/gaps. The private cohort lists and aggregate positives tie the current test shards to the reported146/46 population, but do not establish exact historical generating-source lineage or metric replay.

All42 protected source/data/configuration/historical-report files had identical SHA-256 before and after the primary audit. All131 checkpoint/shard files retained their sizes and nanosecond modification times; none was opened for writing. No clinical record was transmitted to an external service. Individual administrative explanations remain unadjudicated; no source record was declared corrupt solely because of an overlap. Production remediation and methodology approval remain pending.
