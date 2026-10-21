"""Bounded SQL aggregation and independently checked candidate window ownership."""
from collections import Counter
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from noah_baseline.m2.c2_io import EVENT_SCHEMA,Shards,hash_json
from noah_baseline.m2.stream_index import atomic_json,digest_file

WINDOW_SCHEMA=pa.schema([('subject_id',pa.int64()),('split',pa.string()),('window_id',pa.int64()),
    ('event_ids',pa.list_(pa.string())),('owned_target_ids',pa.list_(pa.string())),
    ('context_features_complete',pa.bool_()),('owned_targets',pa.int64()),('last_gap_unknown',pa.bool_())])


def connect(out,config):
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='"+config['limits']['duckdb_memory']+"'")
    scratch=Path(out)/'scratch';scratch.mkdir(exist_ok=True)
    c.execute("SET temp_directory='"+str(scratch).replace("'","''")+"'")
    c.execute("SET max_temp_directory_size='"+str(config['limits']['spill_GiB'])+"GiB'")
    return c


def copy_atomic(c,query,path,check,published):
    check();path=Path(path);temp=path.with_suffix('.pending')
    c.execute('COPY ('+query+") TO '"+str(temp).replace("'","''")+"' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 8192)")
    rows=pq.ParquetFile(temp).metadata.num_rows;sha=digest_file(temp);temp.rename(path)
    published(path.stat().st_size);check()
    return dict(path=str(path),rows=rows,sha256=sha)


def aggregate_vitals(c,inputs,events,out,*,check,published):
    if not inputs['rows']:return dict(status='NO_SUPPORTED_VITALS',groups=0)
    c.read_parquet([r['path'] for r in inputs['shards']]).create_view('vital_inputs')
    c.execute('''CREATE VIEW expanded_vitals AS WITH origins AS
        (SELECT aggregate_group,min(floor(occurred_at)) origin FROM vital_inputs GROUP BY aggregate_group)
        SELECT v.*,CAST(floor(v.occurred_at)-r.i AS BIGINT) bucket FROM vital_inputs v
        JOIN origins USING(aggregate_group),range(2) r(i) WHERE floor(v.occurred_at)-r.i>=origin''')
    edges=copy_atomic(c,'''SELECT aggregate_group,bucket,event_id AS contributor_event_id,subject_id,encounter_id,
        source_sha256,source_ordinal,index_path,occurred_at,available_at,raw_numeric,units
        FROM expanded_vitals ORDER BY aggregate_group,bucket,source_ordinal''',Path(out)/'vital-contributors.parquet',check,published)
    query='''SELECT aggregate_group,bucket,count(*) contributors,CAST(sum(epoch_us(CAST(raw_occurrence AS TIMESTAMP))) AS DOUBLE)/count(*)/3600000000.0 mean_time,
        min(raw_numeric) minimum,max(raw_numeric) maximum,max(occurred_at) last_contributor,
        CASE WHEN count(available_at)=count(*) THEN greatest(max(available_at),max(occurred_at)) ELSE NULL END available,
        first(subject_id) subject_id,first(split) split,first(encounter_id) encounter_id,first(stay_id) stay_id,
        first(source_sha256) source_sha256,first(source_path) source_path,first(index_path) index_path,
        first(dataset_version) dataset_version,first(specific_item) specific_item,first(specifics_key) specifics_key,first(units) units
        FROM expanded_vitals GROUP BY aggregate_group,bucket ORDER BY aggregate_group,bucket'''
    groups=copy_atomic(c,query,Path(out)/'vital-groups.parquet',check,published)
    for batch in pq.ParquetFile(groups['path']).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            for extremum in ['minimum','maximum']:
                event_id='icu.vital-extrema:'+row['aggregate_group']+':'+str(row['bucket'])+':'+extremum
                events.add(dict(event_id=event_id,subject_id=row['subject_id'],split=row['split'],encounter_id=row['encounter_id'],
                    stay_id=row['stay_id'],linkage_status='source' if row['encounter_id'] is not None else 'unlinked',
                    dataset='icu',table='chartevents',dataset_version=row['dataset_version'],source_path=row['source_path'],
                    source_sha256=row['source_sha256'],source_row_id=event_id,source_ordinal=None,source_row_sha256=None,index_path=row['index_path'],
                    occurred_at=row['mean_time'],raw_occurrence=None,precision='derived_contributor_mean',available_at=row['available'],
                    observed_available_at=None,availability_source='unknown' if row['available'] is None else 'derived_contributor_max',
                    raw_storetime=None,category=28,modality='numeric',specific_item=row['specific_item'],units=row['units'],
                    raw_numeric=row[extremum],numeric_vector=None,content_key=None,specifics_key=row['specifics_key'],model_ready=False,
                    semantics=extremum+' of declared [bucket,bucket+2h); all source contributors in edge shards',
                    transform_revision='m2-c2-candidate-v1',aggregate_group=row['aggregate_group'],auxiliary_provenance=json.dumps(dict(
                        bucket=row['bucket'],contributors=row['contributors'],last_contributor=row['last_contributor'],edges=edges['path']))))
        check()
    return dict(status='PASS',groups=groups['rows'],edges=edges,group_summary=groups)


def patient_windows(rows,*,length=64,prefix=6,overlap=16):
    capacity=length-prefix;stride=capacity-overlap
    if stride<=0:raise ValueError('Invalid frozen window geometry')
    ids=[r['event_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate canonical event identities')
    if [(r['occurred_at'],r['event_id']) for r in rows]!=sorted((r['occurred_at'],r['event_id']) for r in rows):
        raise ValueError('Canonical serialization order violated')
    result=[];owners={}
    for j,start in enumerate(range(0,len(rows),stride)):
        group=rows[start:start+capacity];owned=group[0 if j==0 else overlap:]
        if not owned:raise ValueError('Window without shifted targets')
        for offset,r in enumerate(owned,start if j==0 else start+overlap):
            reference=0 if offset<capacity else 1+(offset-capacity)//stride
            if reference!=j or r['event_id'] in owners:raise ValueError('Independent shifted ownership violated')
            owners[r['event_id']]=j
        result.append(dict(subject_id=group[0]['subject_id'],split=group[0]['split'],window_id=j,
            event_ids=[r['event_id'] for r in group],owned_target_ids=[r['event_id'] for r in owned],
            context_features_complete=all(r['model_ready'] for r in group),owned_targets=len(owned),last_gap_unknown=True))
        if start+capacity>=len(rows):break
    if set(owners)!=set(ids):raise ValueError('Canonical targets lost across boundaries')
    return result


def finalize(c,events,out,splits,config,*,check,published):
    out=Path(out);c.read_parquet([r['path'] for r in events['shards']]).create_view('canonical_events')
    if c.execute('SELECT count(*)-count(DISTINCT event_id) FROM canonical_events').fetchone()[0]:raise ValueError('Duplicate global canonical identity')
    windows=Shards(out,'candidate-windows',WINDOW_SCHEMA,check=check,published=published)
    patient_counts={sid:dict(candidate_events=0,feature_complete_events=0,valid_owned_targets=0,candidate_windows=0,complete_windows=0) for sid in splits}
    cursor=c.execute('SELECT subject_id,split,event_id,occurred_at,model_ready FROM canonical_events ORDER BY subject_id,occurred_at,event_id')
    patient=[];previous=None
    def flush():
        if not patient:return
        sid=patient[0]['subject_id'];counts=patient_counts[sid];counts['candidate_events']=len(patient)
        counts['feature_complete_events']=sum(r['model_ready'] for r in patient)
        for w in patient_windows(patient,**{k:config['windows'][k] for k in ['length','prefix','overlap']}):
            windows.add(w);counts['candidate_windows']+=1
            if w['context_features_complete']:
                counts['complete_windows']+=1;counts['valid_owned_targets']+=w['owned_targets']
    names=[x[0] for x in cursor.description]
    while batch:=cursor.fetchmany(8192):
        for values in batch:
            row=dict(zip(names,values))
            if previous is not None and row['subject_id']!=previous:flush();patient=[]
            patient.append(row);previous=row['subject_id']
        check()
    flush();window_manifest=windows.close()
    patient_rows=[dict(subject_id=sid,split=splits[sid],**count) for sid,count in sorted(patient_counts.items())]
    patient_path=out/'patient-eligibility.parquet';pq.write_table(pa.Table.from_pylist(patient_rows),patient_path,compression='zstd');published(patient_path.stat().st_size)
    supports={}
    for dimension in ['category','modality','specific_item']:
        supports[dimension]=copy_atomic(c,f'''SELECT split,{dimension},count(*) events,count(DISTINCT subject_id) patients,
            sum(CAST(model_ready AS BIGINT)) feature_complete_events FROM canonical_events GROUP BY split,{dimension}''',
            out/('support-'+dimension+'.parquet'),check,published)
    cohort={}
    for split in ['train','val','test']:
        rows=[r for r in patient_rows if r['split']==split];minimum=config['minimum_scored_targets']
        eligible=[r for r in rows if r['valid_owned_targets']>=minimum]
        cohort[split]=dict(assigned=len(rows),with_canonical_events=sum(r['candidate_events']>0 for r in rows),
            with_64_candidate_events_NOT_MODEL_READY=sum(r['candidate_events']>=minimum for r in rows),
            with_64_feature_complete_events=sum(r['feature_complete_events']>=minimum for r in rows),
            eligible_64_valid_owned_targets=len(eligible),valid_owned_targets=sum(r['valid_owned_targets'] for r in rows),
            trainable_windows=sum(r['complete_windows'] for r in eligible))
    return dict(cohort=cohort,candidate_windows=window_manifest,supports=supports,
        patient_eligibility=dict(path=str(patient_path),sha256=digest_file(patient_path)),
        encounter_attribution=dict(c.execute("SELECT CASE WHEN encounter_id IS NULL THEN 'longitudinal_unknown' ELSE 'source_attributed' END,count(*) FROM canonical_events GROUP BY 1").fetchall()),
        semantics='Candidates with pending required content/specifics are not completed model targets; window contexts must be feature-complete; eligibility is pretraining-only, not a landmark cohort filter')
