-- a is the diagnostic admissions relation saved privately by the interval audit.
SELECT x.hadm_id,
       count(*) FILTER(WHERE y.admission_type NOT IN ('ELECTIVE','SURGICAL SAME DAY ADMISSION'))>0 no_planned_type,
       count(*) FILTER(WHERE NOT contains(y.admission_type,'OBSERVATION'))>0 no_observation_type,
       count(*) FILTER(WHERE y.admission_type NOT IN ('ELECTIVE','SURGICAL SAME DAY ADMISSION')
                            AND NOT contains(y.admission_type,'OBSERVATION'))>0 neither_type
       FROM a x JOIN a y ON x.subject_id=y.subject_id AND x.hadm_id<>y.hadm_id
       AND (y.admittime>x.admittime OR(y.admittime=x.admittime AND y.hadm_id>x.hadm_id))
       AND y.admittime>=x.dischtime AND y.admittime<=x.dischtime+INTERVAL '30 days'
       GROUP BY x.hadm_id;
