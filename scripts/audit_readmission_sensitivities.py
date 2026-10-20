"""Aggregate diagnostic sensitivities from the private, read-only interval audit.

All alternatives are explicitly unapproved. No existing labels or metrics are written.
"""
import argparse
from pathlib import Path
import json

import duckdb
import pandas as pd

from scripts.audit_readmission_intervals import compare, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-input',type=Path,default=Path('output/readmission_audit/2026-10-19'))
    parser.add_argument('--public-output',type=Path,default=Path('reports/server/2026-10-19_readmission_intervals'))
    args = parser.parse_args()
    private, public = args.private_input, args.public_output
    public.mkdir(parents=True,exist_ok=True)
    output = public/'final_sensitivity_summary.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite an executed sensitivity audit')
    path = private/'all_admission_diagnostic_labels.csv'
    columns = ['subject_id','hadm_id','admittime','dischtime','deathtime','hospital_expire_flag',
               'discharge_location','admission_type','next_hadm_id','gap_hours','negative',
               'valid_index','legacy_index','lower_bound_only','forward_observed',
               'candidate_before_active_episode_end','episode_observed','index_episode_end','next_admittime']
    frame = pd.read_csv(path,usecols=columns,dtype={'next_hadm_id':'Int64'},
                        parse_dates=['admittime','dischtime','deathtime','index_episode_end','next_admittime'])
    conservative_live = (frame.valid_index & frame.hospital_expire_flag.eq(0)
                         & frame.discharge_location.ne('DIED') & frame.deathtime.isna())
    overlap = frame[frame.negative]
    involved = set(overlap.hadm_id.astype(int)) | set(overlap.next_hadm_id.astype(int))
    cohort = pd.read_csv('data/pilot_cohort_manifest.csv')
    c = duckdb.connect(':memory:')
    c.execute('set threads=2')
    c.register('a',frame)
    sql = '''SELECT x.hadm_id,
       count(*) FILTER(WHERE y.admission_type NOT IN ('ELECTIVE','SURGICAL SAME DAY ADMISSION'))>0 no_planned_type,
       count(*) FILTER(WHERE NOT contains(y.admission_type,'OBSERVATION'))>0 no_observation_type,
       count(*) FILTER(WHERE y.admission_type NOT IN ('ELECTIVE','SURGICAL SAME DAY ADMISSION')
                            AND NOT contains(y.admission_type,'OBSERVATION'))>0 neither_type
       FROM a x JOIN a y ON x.subject_id=y.subject_id AND x.hadm_id<>y.hadm_id
       AND (y.admittime>x.admittime OR(y.admittime=x.admittime AND y.hadm_id>x.hadm_id))
       AND y.admittime>=x.dischtime AND y.admittime<=x.dischtime+INTERVAL '30 days'
       GROUP BY x.hadm_id'''
    types = c.execute(sql).fetchdf().set_index('hadm_id')
    for key in ('no_planned_type','no_observation_type','neither_type'):
        frame[key] = frame.hadm_id.map(types[key]).fillna(False).astype(bool)
    populations = {'all_hospital':frame.index,'pilot_manifest':frame.index[frame.subject_id.isin(cohort.subject_id)]}
    subjects_path = private/'processed_test_subjects.json'
    if subjects_path.exists():
        populations['represented_test'] = frame.index[frame.subject_id.isin(json.loads(subjects_path.read_text()))]
    result = {}
    for name,idx in populations.items():
        data = frame.loc[idx]
        live = data[conservative_live.loc[idx]]
        unambiguous = live[~live.hadm_id.isin(involved)]
        result[name] = {
            'index_stays':len(data),'conservative_live_candidates':len(live),
            'negative_own_stay_rows':int((data.dischtime<data.admittime).sum()),
            'negative_own_stay_legacy_positive':int((~data.valid_index&data.legacy_index).sum()),
            'strictly_positive_gap_inside_active_episode':int((live.gap_hours.gt(0)&live.lower_bound_only&live.next_admittime.lt(live.index_episode_end)).sum()),
            'strictly_positive_gap_at_active_episode_end':int((live.gap_hours.gt(0)&live.lower_bound_only&live.next_admittime.eq(live.index_episode_end)).sum()),
            'zero_gap_adjacent_pairs':int(data.gap_hours.eq(0).sum()),
            'strictly_positive_gap_legacy_live_positives':int((live.gap_hours.gt(0)&live.legacy_index).sum()),
            'broadcast_subject_any_vs_conservative_live_index':compare(live.subject_id.map(frame.groupby('subject_id').legacy_index.max()),live.forward_observed,live.subject_id),
            'flag_zero_but_died_disposition':int((data.hospital_expire_flag.eq(0)&data.discharge_location.eq('DIED')).sum()),
            'flag_zero_but_nonmissing_deathtime':int((data.hospital_expire_flag.eq(0)&data.deathtime.notna()).sum()),
            'negative_pair_index_with_died_disposition':int((data.negative&data.discharge_location.eq('DIED')).sum()),
            'legacy_vs_forward_conservative_live':compare(live.legacy_index,live.forward_observed,live.subject_id),
            'lower_bound_vs_forward_conservative_live':compare(live.lower_bound_only,live.forward_observed,live.subject_id),
            'legacy_vs_forward_excluding_overlapping_index_stays':compare(unambiguous.legacy_index,unambiguous.forward_observed,unambiguous.subject_id),
            'overlap_index_stays_excluded_from_conservative_live':len(live)-len(unambiguous),
            'forward_vs_no_planned_type':compare(live.forward_observed,live.no_planned_type,live.subject_id),
            'forward_vs_no_observation_type':compare(live.forward_observed,live.no_observation_type,live.subject_id),
            'forward_vs_neither_type':compare(live.forward_observed,live.neither_type,live.subject_id)}
    result = {'policies':'UNAPPROVED diagnostic sensitivities only; observed-file absence is not confirmed clinical absence',
              'conservative_live_definition':'valid timestamps, hospital_expire_flag=0, discharge_location!=DIED, missing deathtime; no transfer or follow-up inference',
              'type_exclusion_definition':'search all future distinct admissions in the30d window after applying the named type filter; not a validated unplanned/inpatient classification',
              'inputs':{'private_labels_sha256':sha(path),'cohort_sha256':sha('data/pilot_cohort_manifest.csv')},
              'audit_script_sha256':sha(__file__),'populations':result}
    output.write_text(json.dumps(result,indent=2)+'\n')
    (public/'type_filter_sensitivity.sql').write_text('-- a is the diagnostic admissions relation saved privately by the interval audit.\n'+sql+';\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
