-- a is the private diagnostic admissions relation.
SELECT older.subject_id,older.hadm_id ongoing_hadm_id,
        older.admittime ongoing_admittime,older.dischtime ongoing_dischtime,older.admission_type ongoing_type,
        i.hadm_id index_hadm_id,i.admittime index_admittime,i.dischtime index_dischtime,i.admission_type index_type,
        n.hadm_id next_hadm_id,n.admittime next_admittime,n.dischtime next_dischtime,n.admission_type next_type,
        i.gap_hours,datediff('second',older.dischtime,n.admittime)/3600.0 remaining_overlap_hours
        FROM a i JOIN a n ON i.subject_id=n.subject_id AND i.next_hadm_id=n.hadm_id
        JOIN a older ON older.subject_id=i.subject_id AND older.hadm_id<>i.hadm_id
        AND older.admittime<=i.admittime AND older.dischtime>n.admittime
        WHERE i.valid_index AND i.hospital_expire_flag=0
        AND coalesce(i.discharge_location,'')<>'DIED' AND i.deathtime IS NULL
        AND i.gap_hours>0 AND i.gap_hours<=720;

-- all_overlap_pairs
SELECT count(*) pairs,count(DISTINCT subject_id) subjects,
          count(*) FILTER(WHERE adjacent) adjacent_pairs,count(*) FILTER(WHERE NOT adjacent) nonadjacent_pairs
          FROM overlap_pairs;

-- all_overlap_involved_admissions
SELECT count(DISTINCT hadm_id) admissions FROM(
          SELECT index_hadm_id hadm_id FROM overlap_pairs UNION ALL SELECT other_hadm_id FROM overlap_pairs);

-- negative_pair_duration_markers
SELECT count(*) FILTER(WHERE next_dischtime<next_admittime) next_negative_own_duration,
          count(*) FILTER(WHERE next_dischtime=next_admittime) next_zero_own_duration,
          count(*) FILTER(WHERE dischtime<admittime) index_negative_own_duration
          FROM a WHERE negative;

-- legacy_subject_validity_decomposition
SELECT subject_id,
          bool_or(legacy_index) legacy_any,bool_or(legacy_index AND valid_index) valid_legacy_any,
          bool_or(lower_bound_only AND valid_index) valid_lower_bound_any,
          bool_or(forward_observed) forward_any FROM a GROUP BY subject_id;