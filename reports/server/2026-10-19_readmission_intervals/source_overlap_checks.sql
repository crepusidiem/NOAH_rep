-- Run each named query against the in-memory admissions table initialized by
-- scripts/audit_readmission_intervals.py. Results contain aggregate counts only.

-- death_markers
WITH p AS (
 SELECT *, lead(deathtime) OVER w next_deathtime,
 lead(hospital_expire_flag) OVER w next_expire_flag,
 lead(admittime) OVER w next_admittime
 FROM admissions WINDOW w AS (PARTITION BY subject_id ORDER BY admittime,hadm_id)
)
SELECT hospital_expire_flag,next_expire_flag,count(*) pairs,
 count(*) FILTER(WHERE deathtime=next_deathtime) identical_nonmissing_deathtime
FROM p WHERE next_admittime<dischtime GROUP BY 1,2 ORDER BY 1,2;

-- tie_order_sensitivity
WITH p AS (
 SELECT *,lead(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime) literal_next_hadm,
 lead(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime,hadm_id) canonical_next_hadm
 FROM admissions
)
SELECT count(*) FILTER(WHERE literal_next_hadm IS DISTINCT FROM canonical_next_hadm) differing_successors
FROM p;

-- negative_location_pairs
WITH p AS (
 SELECT *,lead(admittime) OVER w next_admittime,
 lead(admission_location) OVER w next_admission_location
 FROM admissions WINDOW w AS (PARTITION BY subject_id ORDER BY admittime,hadm_id)
)
SELECT discharge_location,next_admission_location,count(*) pairs
FROM p WHERE next_admittime<dischtime GROUP BY 1,2 ORDER BY pairs DESC,1,2;

-- affected_discharge_order
WITH p AS (
 SELECT *,lead(admittime) OVER w next_admittime,
 lead(dischtime) OVER w next_dischtime
 FROM admissions WINDOW w AS (PARTITION BY subject_id ORDER BY admittime,hadm_id)
)
SELECT count(*) FILTER(WHERE next_admittime<dischtime AND next_dischtime<dischtime) nested_earlier_end,
 count(*) FILTER(WHERE next_admittime<dischtime AND next_dischtime=dischtime) shared_discharge,
 count(*) FILTER(WHERE next_admittime<dischtime AND next_dischtime>dischtime) later_discharge
FROM p;
