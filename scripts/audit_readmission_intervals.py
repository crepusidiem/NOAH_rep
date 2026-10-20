"""Read-only source-to-label audit; derived patient records stay in an ignored folder.

No production label function is called or changed. Alternative labels below are
diagnostic sensitivities, not approved replacements or complete-follow-up outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import duckdb
import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fraction(n, d):
    return {'n': int(n), 'denominator': int(d), 'percent': 100 * int(n) / int(d) if d else None}


def compare(old, new, subjects):
    old, new = np.asarray(old, bool), np.asarray(new, bool)
    fp, fn = old & ~new, ~old & new
    return {'n': len(old), 'old_positive': int(old.sum()), 'new_positive': int(new.sum()),
            'old_prevalence_percent': float(old.mean() * 100) if len(old) else None,
            'new_prevalence_percent': float(new.mean() * 100) if len(new) else None,
            'false_positive_relative_to_diagnostic': int(fp.sum()),
            'false_negative_relative_to_diagnostic': int(fn.sum()),
            'true_positive': int((old & new).sum()), 'true_negative': int((~old & ~new).sum()),
            'affected_subjects': int(pd.Series(np.asarray(subjects)[fp | fn]).nunique())}


def self_check():
    # An overlapping immediate successor can hide a valid later readmission.
    starts = np.array([0, 4, 12], dtype=np.int64)
    ends = np.array([10, 6, 13], dtype=np.int64)
    assert starts[np.searchsorted(starts, ends[0], side='left')] == 12
    assert starts[1] - ends[0] < 0
    # Exact-zero and exact-30d are explicit sensitivity boundaries, not defaults.
    assert np.searchsorted(np.array([10, 40]), 10, side='left') == 0
    assert (40 - 10) <= 30
    # A true later-stay label cannot be broadcast back to an unrelated index stay.
    assert max([False, True, False]) and not [False, True, False][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimic-root', type=Path, default=Path('/home/wey371/rajpurkarlab/datasets'))
    parser.add_argument('--private-output', type=Path, default=Path('output/readmission_audit/2026-10-19'))
    parser.add_argument('--public-output', type=Path, default=Path('reports/server/2026-10-19_readmission_intervals'))
    parser.add_argument('--inspect-test-shards', action='store_true')
    args = parser.parse_args()
    started = time.time()
    self_check()
    private, public = args.private_output, args.public_output
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    if (public / 'summary.json').exists():
        raise RuntimeError('Refusing to overwrite an executed audit; choose new output directories')
    hosp = args.mimic_root / 'mimic-iv/physionet.org/files/mimiciv/2.2/hosp'
    paths = {'admissions': hosp/'admissions.csv', 'transfers': hosp/'transfers.csv.gz',
             'icustays': hosp.parent/'icu/icustays.csv.gz',
             'edstays': args.mimic_root/'mimic-iv-ed/physionet.org/files/mimic-iv-ed/2.2/ed/edstays.csv'}
    protected = [*paths.values(), Path('data/pilot_cohort_manifest.csv'),
                 *Path('src').rglob('*.py'), *Path('configs').glob('*.yaml'),
                 *Path('reports/server').glob('*.json'),
                 Path('scripts/evaluate_clinical_probes.py'), Path('scripts/train_pilot.py')]
    before = {str(p): sha(p) for p in protected}
    # Stats establish that no checkpoint or shard was rewritten; no training is run.
    immutable_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                       for folder in ('checkpoints', 'data/processed', 'data/processed_with_cxr', 'data/processed_dynamic_demo')
                       for p in Path(folder).rglob('*.pt')}
    con = duckdb.connect(':memory:')
    con.execute('SET threads=2')
    con.execute("SET memory_limit='4GB'")
    con.execute('CREATE TABLE admissions AS SELECT * FROM read_csv_auto(?)', [str(paths['admissions'])])
    queries = {}
    def query(name, sql):
        queries[name] = sql
        return con.execute(sql).fetchdf()
    quality = query('source_quality', '''SELECT count(*) raw_rows, count(DISTINCT subject_id) subjects,
        count(DISTINCT hadm_id) admissions, count(*)-count(DISTINCT hadm_id) duplicate_hadm_rows,
        count(*) FILTER(WHERE admittime IS NULL) missing_admit,
        count(*) FILTER(WHERE dischtime IS NULL) missing_discharge,
        count(*) FILTER(WHERE dischtime<admittime) negative_own_stay,
        count(*) FILTER(WHERE dischtime=admittime) zero_own_stay
        FROM admissions''').iloc[0].to_dict()
    duplicate_rows = query('exact_duplicates', '''SELECT coalesce(sum(n-1),0) duplicated_rows FROM
        (SELECT *,count(*) n FROM admissions GROUP BY ALL HAVING count(*)>1)''')
    quality['exact_duplicate_rows'] = int(duplicate_rows.iloc[0, 0])
    quality['admission_time_tie_groups'] = len(query('ties', '''SELECT subject_id,admittime,count(*) n
        FROM admissions GROUP BY subject_id,admittime HAVING count(*)>1'''))
    frame = query('canonical_pairs', '''SELECT *,
        lead(hadm_id) OVER w next_hadm_id, lead(admittime) OVER w next_admittime,
        lead(dischtime) OVER w next_dischtime, lead(admission_type) OVER w next_admission_type,
        lead(admission_location) OVER w next_admission_location,
        lead(discharge_location) OVER w next_discharge_location
        FROM admissions WINDOW w AS(PARTITION BY subject_id ORDER BY admittime,hadm_id)
        ORDER BY subject_id,admittime,hadm_id''')
    con.register('canonical', frame)
    historical = query('literal_historical_query', '''WITH ranked_adm AS (
        SELECT subject_id, hadm_id, admittime, dischtime,
        lead(admittime) OVER(PARTITION BY subject_id ORDER BY admittime) next_admittime
        FROM admissions)
        SELECT subject_id,max(CASE WHEN datediff('second',dischtime,next_admittime)<=30*86400
        THEN 1 ELSE 0 END) old_subject_label FROM ranked_adm GROUP BY subject_id''')
    frame['gap_hours'] = (frame.next_admittime-frame.dischtime).dt.total_seconds()/3600
    frame['negative'] = frame.gap_hours.lt(0)
    frame['legacy_index'] = frame.gap_hours.le(720)
    frame['lower_bound_only'] = frame.gap_hours.between(0,720,inclusive='both')
    frame['valid_index'] = frame.admittime.notna() & frame.dischtime.notna() & frame.dischtime.ge(frame.admittime)
    frame['live_index'] = frame.valid_index & frame.hospital_expire_flag.eq(0)
    # Work in NumPy buffers: no repeated mutation/copying of large pandas columns.
    count = len(frame)
    all_starts = frame.admittime.to_numpy(dtype='datetime64[ns]')
    all_ends = frame.dischtime.to_numpy(dtype='datetime64[ns]')
    all_ids = frame.hadm_id.to_numpy()
    all_valid = frame.valid_index.to_numpy()
    forward_ids = np.full(count, -1, dtype=np.int64)
    positive_ids = np.full(count, -1, dtype=np.int64)
    forward_times = np.full(count, np.datetime64('NaT'), dtype='datetime64[ns]')
    episode_ends = np.full(count, np.datetime64('NaT'), dtype='datetime64[ns]')
    next_episode_times = np.full(count, np.datetime64('NaT'), dtype='datetime64[ns]')
    # Sorted hospital admissions only. Do not mix ED stay_id or ICU stay_id.
    for idx in frame.groupby('subject_id', sort=False).indices.values():
        starts, ends, ids = all_starts[idx], all_ends[idx], all_ids[idx]
        valid = all_valid[idx]
        offsets = np.arange(len(idx))
        for side, destination in (('left',forward_ids),('right',positive_ids)):
            successor = np.maximum(offsets+1, np.searchsorted(starts, ends, side=side))
            usable = valid & (successor<len(idx))
            destination[idx[usable]] = ids[successor[usable]]
            if side == 'left':
                forward_times[idx[usable]] = starts[successor[usable]]
        # An overlap-connected episode is an explicit sensitivity, not a transfer assertion.
        active = offsets[valid]
        if not len(active):
            continue
        valid_starts, valid_ends = starts[active], ends[active]
        running_end = np.maximum.accumulate(valid_ends)
        new_episode = np.r_[True, valid_starts[1:] > running_end[:-1]]
        boundaries = np.flatnonzero(new_episode)
        episode_numbers = np.cumsum(new_episode)-1
        final_ends = np.maximum.reduceat(valid_ends,boundaries)
        episode_ends[idx[active]] = final_ends[episode_numbers]
        following = np.r_[valid_starts[boundaries[1:]], np.datetime64('NaT')]
        next_episode_times[idx[active]] = following[episode_numbers]
    frame['forward_hadm_id'] = pd.array(np.where(forward_ids<0,None,forward_ids),dtype='Int64')
    frame['strict_positive_hadm_id'] = pd.array(np.where(positive_ids<0,None,positive_ids),dtype='Int64')
    frame['forward_admittime'] = forward_times
    frame['index_episode_end'] = episode_ends
    frame['next_episode_admittime'] = next_episode_times
    frame['candidate_before_active_episode_end'] = frame.forward_admittime.le(frame.index_episode_end)
    reference = query('independent_forward_reference', '''SELECT a.hadm_id,min(b.admittime) first_after_discharge
        FROM admissions a LEFT JOIN admissions b
        ON a.subject_id=b.subject_id AND b.hadm_id<>a.hadm_id
        AND (b.admittime>a.admittime OR (b.admittime=a.admittime AND b.hadm_id>a.hadm_id))
        AND b.admittime>=a.dischtime AND a.dischtime>=a.admittime
        GROUP BY a.hadm_id''').set_index('hadm_id').first_after_discharge
    first_reference = frame.hadm_id.map(reference)
    disagreement = ~(frame.forward_admittime.eq(first_reference) | (frame.forward_admittime.isna() & first_reference.isna()))
    assert not disagreement.any(), 'Forward successor differs from independent self-join reference'
    quality['independent_forward_reference_disagreements'] = int(disagreement.sum())
    frame['forward_gap_hours'] = (frame.forward_admittime-frame.dischtime).dt.total_seconds()/3600
    frame['forward_observed'] = frame.forward_gap_hours.between(0,720,inclusive='both') & frame.valid_index
    frame['episode_gap_hours'] = (frame.next_episode_admittime-frame.index_episode_end).dt.total_seconds()/3600
    frame['episode_observed'] = frame.episode_gap_hours.between(0,720,inclusive='both') & frame.valid_index
    frame['negative_geometry'] = np.where(frame.next_admittime.eq(frame.admittime),'same_admission_timestamp',
        np.where(frame.next_dischtime.le(frame.dischtime),'nested_successor','crossing_overlap'))
    types = frame.set_index('hadm_id').admission_type.to_dict()
    frame['forward_admission_type'] = frame.forward_hadm_id.map(types)
    # Only a sensitivity: these type exclusions are not a complete planned-admission classifier.
    planned = {'ELECTIVE','SURGICAL SAME DAY ADMISSION'}
    frame['forward_planned_type'] = frame.forward_admission_type.isin(planned)
    cohort = pd.read_csv('data/pilot_cohort_manifest.csv')
    assert not cohort.subject_id.duplicated().any(), 'Cohort duplicates multiply SQL joins'
    assert not frame.hadm_id.duplicated().any(), 'Duplicate hospital IDs require separate investigation'
    assert frame.admittime.notna().all(), 'Missing admission ordering needs separate policy'
    expected_subject = frame.groupby('subject_id').legacy_index.max().astype(int)
    actual_subject = historical.set_index('subject_id').old_subject_label
    quality['literal_historical_vs_deterministic_subject_disagreements'] = int((expected_subject.sort_index()!=actual_subject.sort_index()).sum())
    # Directly replay only the admission ETL method: no full ETL or encoder/model execution.
    from src.data.duckdb_extractor import DuckDBEventExtractor
    extractor = DuckDBEventExtractor()
    events, prefixes, history = {}, {}, {}
    events = {int(sid): [] for sid in cohort.subject_id}
    extractor._extract_admissions_and_demographics('JOIN cohort_manifest fs ON t.subject_id=fs.subject_id',events,prefixes,history)
    from src.data.schema import EventCategory
    source = frame.set_index('hadm_id')
    enter, leave, discrepancies = {}, {}, []
    for sid, items in events.items():
        for event in items:
            if event.category_id not in {EventCategory.ENTER_HOSPITALIZATION,EventCategory.LEAVE_HOSPITALIZATION}:
                continue
            raw = source.loc[event.hadm_id]
            raw_time = raw.admittime if event.category_id==EventCategory.ENTER_HOSPITALIZATION else raw.dischtime
            expected = extractor.dt_to_hours(sid, raw_time.to_pydatetime())
            if event.subject_id != raw.subject_id or abs(event.charttime-expected)>1e-8:
                discrepancies.append(event.source_row_id)
            (enter if event.category_id==EventCategory.ENTER_HOSPITALIZATION else leave)[event.hadm_id] = event.charttime
    cohort_rows = frame[frame.subject_id.isin(cohort.subject_id)]
    gap_disagreements = 0
    for row in cohort_rows[cohort_rows.next_hadm_id.notna() & cohort_rows.dischtime.notna()].itertuples():
        transformed_gap = enter[int(row.next_hadm_id)]-leave[row.hadm_id]
        if abs(transformed_gap-row.gap_hours)>1e-7:
            gap_disagreements += 1
    etl = {'scope':'current admission-specific method, all 1000 cohort subjects; not full ETL',
           'admission_events':len(enter),'discharge_events':len(leave),
           'identity_timestamp_discrepancies':len(discrepancies),'adjacent_gap_discrepancies':gap_disagreements,
           'negative_pairs_raw':int(cohort_rows.negative.sum()),
           'historical_shard_hadm_provenance':'ModelWindow does not retain hospital encounter/source-row identity; exact historical event lineage cannot be reconstructed from shards alone'}
    extractor.close()
    # Distinct hierarchy tables are inspected by both subject and hadm, avoiding fan-out joins.
    negative = frame[frame.negative].copy()
    involved_ids = set(negative.hadm_id.astype(int)) | set(negative.next_hadm_id.astype(int))
    con.execute('CREATE TABLE involved(hadm_id BIGINT PRIMARY KEY)')
    con.executemany('INSERT INTO involved VALUES(?)', [(x,) for x in sorted(involved_ids)])
    hierarchy = {}
    child_tables = {}
    for name in ('edstays','icustays','transfers'):
        con.execute(f'CREATE TABLE {name} AS SELECT * FROM read_csv_auto(?)', [str(paths[name])])
        child = query(name+'_linked',f'''SELECT c.* FROM {name} c JOIN involved i USING(hadm_id)''')
        child_tables[name] = child
        hierarchy[name] = {'rows_linked_to_involved_hadm':len(child),
            'distinct_hadm_linked':int(child.hadm_id.nunique()),
            'subject_hadm_disagreements':int(sum(int(row.subject_id)!=int(source.loc[int(row.hadm_id)].subject_id) for row in child.itertuples()))}
    for row in negative.itertuples():
        for name, child in child_tables.items():
            for prefix, hadm in (('index',row.hadm_id),('next',int(row.next_hadm_id))):
                selected = child[child.hadm_id.eq(hadm)]
                negative.loc[row.Index,prefix+'_'+name+'_rows'] = len(selected)
    negative.to_csv(private/'all_negative_pairs.csv',index=False)
    frame.to_csv(private/'all_admission_diagnostic_labels.csv',index=False)
    for name, child in child_tables.items():
        child.to_csv(private/(name+'_involved_encounters.csv'),index=False)
    populations = {'all_mimic_iv_2_2_hospital': frame.index,
                   'pilot_manifest_1000':cohort_rows.index}
    for split in ('train','val','test'):
        populations['pilot_manifest_'+split] = frame.index[frame.subject_id.isin(cohort.loc[cohort.split.eq(split),'subject_id'])]
    represented = {}
    if args.inspect_test_shards:
        import torch
        for folder in ('processed','processed_with_cxr','processed_dynamic_demo'):
            sids = set()
            for path in sorted(Path('data',folder,'test').glob('*.pt')):
                windows = torch.load(path,map_location='cpu',weights_only=False)
                sids.update(int(w.subject_id) for w in windows)
                del windows
            represented[folder] = {'subjects':len(sids)}
            populations[folder+'_represented_test'] = frame.index[frame.subject_id.isin(sids)]
            (private/(folder+'_test_subjects.json')).write_text(json.dumps(sorted(sids))+'\n')
    stats = {}
    for name, idx in populations.items():
        data = frame.loc[idx]
        neg = data[data.negative]
        pair_count = int(data.next_hadm_id.notna().sum())
        involving = set(neg.hadm_id.astype(int)) | set(neg.next_hadm_id.astype(int))
        live = data[data.live_index]
        valid = data[data.valid_index]
        grouped = data.groupby('subject_id')
        patients = grouped[['legacy_index','lower_bound_only','forward_observed']].max()
        last = data.sort_values(['subject_id','admittime','hadm_id']).groupby('subject_id').tail(1)
        live_last = last[last.live_index]
        assigned_any = live.subject_id.map(patients.legacy_index).to_numpy(bool)
        negatives_only_subjects = patients.legacy_index & ~patients.lower_bound_only
        raw_gaps = neg.gap_hours.to_numpy()
        stats[name] = {
            'subjects':int(data.subject_id.nunique()),'distinct_hospital_admissions':int(data.hadm_id.nunique()),
            'adjacent_pairs':pair_count,'valid_discharge_index_stays':int(data.valid_index.sum()),
            'affected_subjects':fraction(neg.subject_id.nunique(),data.subject_id.nunique()),
            'admissions_involved_in_negative_pairs':fraction(len(involving),data.hadm_id.nunique()),
            'negative_adjacent_pairs':fraction(len(neg),pair_count),
            'potentially_misclassified_index_stays':fraction(len(neg),int(data.valid_index.sum())),
            'negative_hours_quantiles':{str(q):float(np.quantile(raw_gaps,q)) for q in (0,.05,.25,.5,.75,.95,1)} if len(neg) else {},
            'negative_magnitude_bins_hours':{label:int(mask.sum()) for label,mask in {
                '(0,1]':neg.gap_hours.ge(-1),'(1,6]':neg.gap_hours.lt(-1)&neg.gap_hours.ge(-6),
                '(6,24]':neg.gap_hours.lt(-6)&neg.gap_hours.ge(-24),
                '(24,72]':neg.gap_hours.lt(-24)&neg.gap_hours.ge(-72),
                '(72,infinity)':neg.gap_hours.lt(-72)}.items()},
            'negative_geometry':neg.negative_geometry.value_counts().to_dict(),
            'negative_type_pairs':neg.groupby(['admission_type','next_admission_type']).size().to_dict(),
            'legacy_vs_forward_all_valid_index':compare(valid.legacy_index,valid.forward_observed,valid.subject_id),
            'legacy_vs_forward_live_index':compare(live.legacy_index,live.forward_observed,live.subject_id),
            'lower_bound_only_vs_forward_live_index':compare(live.lower_bound_only,live.forward_observed,live.subject_id),
            'subject_any_legacy_vs_subject_any_forward':compare(patients.legacy_index,patients.forward_observed,patients.index),
            'subject_any_legacy_vs_subject_any_lower_bound':compare(patients.legacy_index,patients.lower_bound_only,patients.index),
            'broadcast_subject_any_vs_encounter_live':compare(assigned_any,live.forward_observed,live.subject_id),
            'subject_any_legacy_vs_last_live_index':compare(live_last.subject_id.map(patients.legacy_index),live_last.forward_observed,live_last.subject_id),
            'subjects_positive_only_due_to_negative_pairs':int(negatives_only_subjects.sum()),
            'zero_gap_pairs':int(data.gap_hours.eq(0).sum()),
            'same_calendar_day_positive_pairs':int((data.gap_hours.gt(0)&data.next_admittime.dt.normalize().eq(data.dischtime.dt.normalize())).sum()),
            'exact_30_day_adjacent_pairs':int(data.gap_hours.eq(720).sum()),
            'index_expired_legacy_positive':int((data.legacy_index & data.hospital_expire_flag.eq(1)).sum()),
            'simple_fix_missed_later_positive':int((live.negative & live.forward_observed).sum()),
            'positive_gap_index_in_active_overlap_episode':int((live.lower_bound_only & live.candidate_before_active_episode_end).sum()),
            'positive_gap_successor_with_planned_type':int((live.lower_bound_only & live.next_admission_type.isin(planned)).sum()),
            'positive_gap_successor_observation_type':int((live.lower_bound_only & live.next_admission_type.str.contains('OBSERVATION',na=False)).sum()),
            'forward_vs_overlap_episode_sensitivity':compare(live.forward_observed,live.episode_observed,live.subject_id),
            'next_id_order_vs_admit_order_disagreement':int((data.next_hadm_id.notna() & data.next_hadm_id.lt(data.hadm_id)).sum())}
        stats[name]['negative_type_pairs'] = {' -> '.join(k):int(v) for k,v in stats[name]['negative_type_pairs'].items()}
    # Deterministic representative selection: cohort overlap, each geometry/type, quantiles,
    # then nonnegative error classes. Never claim absent causes have examples.
    chosen = []
    def pick(subset, reason):
        candidates = subset[~subset.index.isin([i for i,_ in chosen])]
        if len(candidates):
            chosen.append((int(candidates.index[0]),reason))
    pick(cohort_rows[cohort_rows.negative],'pilot_source_overlap')
    for key, subset in frame[frame.negative].groupby(['negative_geometry','admission_type','next_admission_type'],sort=True):
        pick(subset.sort_values('gap_hours'), 'source_overlap:'+str(key))
    for q in (0,.25,.5,.75,1):
        if len(negative):
            goal=negative.gap_hours.quantile(q)
            pick(frame.loc[(negative.gap_hours-goal).abs().sort_values().index],f'negative_gap_quantile_{q}')
    for subset,reason in [
        (frame[frame.live_index & frame.negative & frame.forward_observed],'lower_bound_fix_misses_later_admission'),
        (frame[frame.live_index & frame.lower_bound_only & frame.candidate_before_active_episode_end],'positive_gap_inside_active_overlap_episode'),
        (frame[frame.live_index & frame.lower_bound_only & frame.next_admission_type.isin(planned)],'positive_gap_planned_type_sensitivity'),
        (frame[frame.live_index & frame.lower_bound_only & frame.next_admission_type.str.contains('OBSERVATION',na=False)],'positive_gap_observation_type_sensitivity'),
        (frame[frame.legacy_index & frame.hospital_expire_flag.eq(1)],'expired_index_not_live_discharge'),
        (frame[frame.gap_hours.eq(0)],'zero_gap_boundary'),
        (frame[~frame.forward_observed & frame.subject_id.map(expected_subject).eq(1)],'subject_max_broadcast_wrong_index')]:
        pick(subset,reason)
    cases = frame.loc[[i for i,_ in chosen]].copy()
    cases.insert(0,'selection_reason',[reason for _,reason in chosen])
    cases.insert(0,'case_number',range(1,len(cases)+1))
    cases['transformation'] = 'raw admissions -> partition subject_id -> ORDER BY admittime,hadm_id -> LEAD(next hospital admission); legacy subject MAX; no ED/ICU joins'
    cases.to_csv(private/'representative_cases.csv',index=False)
    columns=['case_number','selection_reason','subject_id','hadm_id','admittime','dischtime','admission_type',
             'next_hadm_id','next_admittime','next_dischtime','next_admission_type','gap_hours',
             'forward_hadm_id','forward_admittime','forward_gap_hours','negative_geometry','transformation']
    text=['# Private representative hospital-admission timelines','',
          'Derived MIMIC records: keep in the ignored private audit directory. Timestamps are shifted source times.', '',
          'Each case uses hospital hadm_id. ICU/ED child records do not define the legacy successor.','']
    for row in cases[columns].to_dict('records'):
        text += [f"## Case {row['case_number']}: {row['selection_reason']}",'',
                 *[f'- {key}: {value}' for key,value in row.items() if key not in {'case_number','selection_reason'}], '']
    (private/'representative_timelines.md').write_text('\n'.join(text)+'\n')
    (private/'queries.json').write_text(json.dumps(queries,indent=2)+'\n')
    shutil_query = '\n\n'.join('-- '+name+'\n'+sql+';' for name,sql in queries.items())
    (public/'queries.sql').write_text('-- In-memory DuckDB tables are initialized by scripts/audit_readmission_intervals.py.\n'+shutil_query+'\n')
    after = {str(p):sha(p) for p in protected}
    assert before == after, 'Protected source, code, configurations or reports changed during audit'
    assert immutable_stats == {p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in immutable_stats}, 'A checkpoint/shard changed during audit'
    summary = {'date':'2026-10-19','baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'audit_script_sha256':sha(Path(__file__)), 'host':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'python':platform.python_version(),'duckdb':duckdb.__version__,'pandas':pd.__version__,
        'source_tables':{name:{'path':str(path),'bytes':path.stat().st_size,'sha256':before[str(path)]} for name,path in paths.items()},
        'quality':{k:int(v) for k,v in quality.items()},'populations':stats,'etl':etl,'hierarchy':hierarchy,
        'represented_test_sets':represented,'private_cases':len(cases),
        'definitions':{'legacy':'immediate chronological next hospital admission; gap<=30d, including negative; then MAX over subject',
            'lower_bound_only':'same immediate successor, 0<=gap<=30d',
            'forward_observed':'for each valid index hadm_id, earliest subsequent distinct hospital admission with admittime>=index discharge, 0<=gap<=30d; all source admission types; no inferred transfer/planned exclusions',
            'live_index':'nonmissing admit/discharge, discharge>=admit and hospital_expire_flag=0',
            'negative_label_limitation':'zero means no qualifying admission recorded in this file; complete follow-up and outside-hospital outcomes are NOT established; not a clinical gold standard',
            'episode_sensitivity':'merge overlap/touch-connected intervals using running maximum discharge; first new episode within30d of episode end, broadcast to its member indexes for comparison only; different estimand, NOT approved methodology'},
        'read_only_verification':{'protected_sha256_before_equals_after':True,'checkpoint_shard_size_mtime_unchanged':True,'protected_files':len(before),'checkpoint_shard_files':len(immutable_stats)},
        'private_artifacts':str(private),'elapsed_seconds':round(time.time()-started,2),'training_runs':0}
    (private/'protected_hashes.json').write_text(json.dumps(before,indent=2)+'\n')
    (public/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps({'summary':str(public/'summary.json'),'source_negative_pairs':int(frame.negative.sum()),
                      'cohort_negative_pairs':int(cohort_rows.negative.sum()),'cases':len(cases),'elapsed_seconds':summary['elapsed_seconds']}))


if __name__ == '__main__':
    main()
