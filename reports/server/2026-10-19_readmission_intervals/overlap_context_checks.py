"""Inspect nonadjacent overlap contexts without changing any input artifact."""
from pathlib import Path
import hashlib
import json

import duckdb


def main():
    private = Path('output/readmission_audit/2026-10-19')
    public = Path('reports/server/2026-10-19_readmission_intervals')
    output = public/'overlap_context_summary.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite executed overlap checks')
    c = duckdb.connect(':memory:')
    c.execute('SET threads=2')
    source = private/'all_admission_diagnostic_labels.csv'
    c.execute('CREATE TABLE a AS SELECT * FROM read_csv_auto(?)',[str(source)])
    c.execute("CREATE TABLE cohort AS SELECT * FROM read_csv_auto('data/pilot_cohort_manifest.csv')")
    c.execute('CREATE TABLE overlap_pairs AS SELECT x.subject_id,x.hadm_id index_hadm_id,y.hadm_id other_hadm_id,'
              'x.next_hadm_id=y.hadm_id adjacent FROM a x JOIN a y ON x.subject_id=y.subject_id '
              'AND(y.admittime>x.admittime OR(y.admittime=x.admittime AND y.hadm_id>x.hadm_id)) '
              'AND y.admittime<x.dischtime')
    sql = '''SELECT older.subject_id,older.hadm_id ongoing_hadm_id,
        older.admittime ongoing_admittime,older.dischtime ongoing_dischtime,older.admission_type ongoing_type,
        i.hadm_id index_hadm_id,i.admittime index_admittime,i.dischtime index_dischtime,i.admission_type index_type,
        n.hadm_id next_hadm_id,n.admittime next_admittime,n.dischtime next_dischtime,n.admission_type next_type,
        i.gap_hours,datediff('second',older.dischtime,n.admittime)/3600.0 remaining_overlap_hours
        FROM a i JOIN a n ON i.subject_id=n.subject_id AND i.next_hadm_id=n.hadm_id
        JOIN a older ON older.subject_id=i.subject_id AND older.hadm_id<>i.hadm_id
        AND older.admittime<=i.admittime AND older.dischtime>n.admittime
        WHERE i.valid_index AND i.hospital_expire_flag=0
        AND coalesce(i.discharge_location,'')<>'DIED' AND i.deathtime IS NULL
        AND i.gap_hours>0 AND i.gap_hours<=720'''
    cases = c.execute(sql).fetchdf()
    cases.to_csv(private/'strict_positive_gap_inside_ongoing_hospital_account.csv',index=False)
    markdown = ['# Private positive-gap context: an earlier hospital account remains open','',
                'The chronological index-to-next pair has a positive gap, but the next admission begins before an earlier account ends. This is an episode-policy ambiguity, not a confirmed clinical readmission error.','']
    for number,row in enumerate(cases.to_dict('records'),1):
        markdown += [f'## Context case {number}','',*[f'- {key}: {value}' for key,value in row.items()],
                     '- transformation: ORDER BY admission time -> adjacent index/next hospital account; earlier-account self-join demonstrates ongoing overlap.', '']
    (private/'positive_gap_context_timelines.md').write_text('\n'.join(markdown)+'\n')
    checks = {
        'all_overlap_pairs':'''SELECT count(*) pairs,count(DISTINCT subject_id) subjects,
          count(*) FILTER(WHERE adjacent) adjacent_pairs,count(*) FILTER(WHERE NOT adjacent) nonadjacent_pairs
          FROM overlap_pairs''',
        'all_overlap_involved_admissions':'''SELECT count(DISTINCT hadm_id) admissions FROM(
          SELECT index_hadm_id hadm_id FROM overlap_pairs UNION ALL SELECT other_hadm_id FROM overlap_pairs)''',
        'negative_pair_duration_markers':'''SELECT count(*) FILTER(WHERE next_dischtime<next_admittime) next_negative_own_duration,
          count(*) FILTER(WHERE next_dischtime=next_admittime) next_zero_own_duration,
          count(*) FILTER(WHERE dischtime<admittime) index_negative_own_duration
          FROM a WHERE negative''',
        'legacy_subject_validity_decomposition':'''SELECT subject_id,
          bool_or(legacy_index) legacy_any,bool_or(legacy_index AND valid_index) valid_legacy_any,
          bool_or(lower_bound_only AND valid_index) valid_lower_bound_any,
          bool_or(forward_observed) forward_any FROM a GROUP BY subject_id'''}
    result = {}
    for name,query in checks.items():
        cursor = c.execute(query)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols,row)) for row in cursor.fetchall()]
        if name == 'legacy_subject_validity_decomposition':
            result[name] = {'removed_by_invalid_index_exclusion':sum(r['legacy_any'] and not r['valid_legacy_any'] for r in rows),
                'removed_by_negative_candidate_repair_on_valid_indexes':sum(r['valid_legacy_any'] and not r['forward_any'] for r in rows),
                'lower_bound_only_misses_forward_positive_subject':sum(not r['valid_lower_bound_any'] and r['forward_any'] for r in rows)}
        else:
            result[name] = rows
    lists = [json.loads((private/(folder+'_test_subjects.json')).read_text())
             for folder in ('processed','processed_with_cxr','processed_dynamic_demo')]
    result['test_subject_lists_identical'] = lists[0]==lists[1]==lists[2]
    result['strictly_positive_context_cases'] = len(cases)
    result['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['recipe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result,indent=2)+'\n')
    (public/'overlap_context_checks.sql').write_text('-- a is the private diagnostic admissions relation.\n'+sql+';\n\n'+
        '\n\n'.join('-- '+name+'\n'+query+';' for name,query in checks.items()))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
