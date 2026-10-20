-- In-memory DuckDB tables are initialized by scripts/audit_readmission_intervals.py.
-- source_quality
SELECT count(*) raw_rows, count(DISTINCT subject_id) subjects,
        count(DISTINCT hadm_id) admissions, count(*)-count(DISTINCT hadm_id) duplicate_hadm_rows,
        count(*) FILTER(WHERE admittime IS NULL) missing_admit,
        count(*) FILTER(WHERE dischtime IS NULL) missing_discharge,
        count(*) FILTER(WHERE dischtime<admittime) negative_own_stay,
        count(*) FILTER(WHERE dischtime=admittime) zero_own_stay
        FROM admissions;

-- exact_duplicates
SELECT coalesce(sum(n-1),0) duplicated_rows FROM
        (SELECT *,count(*) n FROM admissions GROUP BY ALL HAVING count(*)>1);

-- ties
SELECT subject_id,admittime,count(*) n
        FROM admissions GROUP BY subject_id,admittime HAVING count(*)>1;

-- canonical_pairs
SELECT *,
        lead(hadm_id) OVER w next_hadm_id, lead(admittime) OVER w next_admittime,
        lead(dischtime) OVER w next_dischtime, lead(admission_type) OVER w next_admission_type,
        lead(admission_location) OVER w next_admission_location,
        lead(discharge_location) OVER w next_discharge_location
        FROM admissions WINDOW w AS(PARTITION BY subject_id ORDER BY admittime,hadm_id)
        ORDER BY subject_id,admittime,hadm_id;

-- literal_historical_query
WITH ranked_adm AS (
        SELECT subject_id, hadm_id, admittime, dischtime,
        lead(admittime) OVER(PARTITION BY subject_id ORDER BY admittime) next_admittime
        FROM admissions)
        SELECT subject_id,max(CASE WHEN datediff('second',dischtime,next_admittime)<=30*86400
        THEN 1 ELSE 0 END) old_subject_label FROM ranked_adm GROUP BY subject_id;

-- independent_forward_reference
SELECT a.hadm_id,min(b.admittime) first_after_discharge
        FROM admissions a LEFT JOIN admissions b
        ON a.subject_id=b.subject_id AND b.hadm_id<>a.hadm_id
        AND (b.admittime>a.admittime OR (b.admittime=a.admittime AND b.hadm_id>a.hadm_id))
        AND b.admittime>=a.dischtime AND a.dischtime>=a.admittime
        GROUP BY a.hadm_id;

-- edstays_linked
SELECT c.* FROM edstays c JOIN involved i USING(hadm_id);

-- icustays_linked
SELECT c.* FROM icustays c JOIN involved i USING(hadm_id);

-- transfers_linked
SELECT c.* FROM transfers c JOIN involved i USING(hadm_id);
