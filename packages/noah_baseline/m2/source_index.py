"""Streaming, columnar source index. This does not produce model-ready events.

Untimed records and date precision survive unchanged. Large note bodies are
represented by hashes and row identities. Execution requires a separate budget
approval; the bounded helper is independently testable on fictional CSV input.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time

import duckdb
import pandas as pd

from noah_baseline.m2.adapters import SOURCE_CATEGORIES
from noah_baseline.m2.inventory import dump,sha
from noah_baseline.runtime import ArtifactNamespace

# Only identities, event coordinates, units, structured payload and demographics.
FIELDS=set('''subject_id hadm_id stay_id labevent_id emar_id emar_seq pharmacy_id poe_id poe_seq
itemid specimen_id microevent_id micro_specimen_id note_id note_type note_seq study_id dicom_id
path dicom_filepath acquisition_datetime ecg_time admittime dischtime deathtime intime outtime
starttime endtime stoptime charttime storetime chartdate storedate transfertime verifiedtime entertime
StudyDate StudyTime seq_num icd_code icd_version medication drug name event_txt value valuenum
valueuom amount amountuom rate rateuom orderid linkorderid statusdescription originalamount originalrate
test_itemid test_name spec_itemid spec_type_desc org_itemid org_name ab_itemid ab_name interpretation
careunit curr_service prev_service temperature heartrate resprate o2sat sbp dbp pain acuity gender
race insurance language marital_status anchor_age anchor_year'''.split())
CLOCK_FIELDS=set('''admittime dischtime deathtime intime outtime starttime endtime stoptime
charttime storetime chartdate storedate transfertime verifiedtime entertime acquisition_datetime
ecg_time StudyDate StudyTime'''.split())


def index_diagnostics(connection, target, columns):
    """Describe raw fields without selecting canonical occurrence/availability policies."""
    missing=lambda q: f"({q} IS NULL OR trim({q})='')"
    counts=connection.execute('''SELECT split,count(*) source_rows,count(DISTINCT subject_id) patients
        FROM read_parquet(?) GROUP BY split ORDER BY split''',[str(target)]).fetchdf().to_dict('records')
    clocks={};quality={}
    for name in sorted(set(columns)&CLOCK_FIELDS):
        q='"'+name+'"';empty=missing(q)
        date=f"regexp_full_match(trim({q}), '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}|[0-9]{{8}}')"
        full=f"(NOT {date} AND try_cast({q} AS TIMESTAMP) IS NOT NULL)"
        clocks[name]=connection.execute(f'''SELECT split,count(*) denominator,
            count(*) FILTER(WHERE {empty}) missing,
            count(*) FILTER(WHERE NOT {empty} AND {date}) date_only_strings,
            count(*) FILTER(WHERE NOT {empty} AND {full}) parseable_timestamp_strings,
            count(*) FILTER(WHERE NOT {empty} AND NOT {date} AND NOT {full}) other_nonempty_strings
            FROM read_parquet(?) GROUP BY split ORDER BY split''',[str(target)]).fetchdf().to_dict('records')
    for name in sorted(set(columns)&{'hadm_id','stay_id','itemid','valueuom','amountuom','rateuom','valuenum'}):
        q='"'+name+'"'
        quality[name]=connection.execute(f'''SELECT split,count(*) denominator,
            count(*) FILTER(WHERE {missing(q)}) missing,count(DISTINCT {q}) distinct_nonnull_values
            FROM read_parquet(?) GROUP BY split ORDER BY split''',[str(target)]).fetchdf().to_dict('records')
    return dict(counts=counts,raw_clock_completeness=clocks,raw_field_completeness=quality,
        availability_validation='PENDING: field presence/parseability does not establish prospective availability',
        canonical_events='NOT_COMPUTED',qualified_64_target_patients='NOT_COMPUTED',
        encounter_matching='PENDING',duplicates_and_content_validity='PENDING')


def sql_string(value):
    return "'"+str(value).replace("'","''")+"'"


def write_source_index(connection, source, target, splits, *, source_row_limit=None):
    if target.exists():raise FileExistsError('Source index overwrite forbidden')
    connection.register('cohort',pd.DataFrame(dict(subject_id=list(splits),split=list(splits.values()))))
    header=connection.execute('DESCRIBE SELECT * FROM read_csv(?,header=true,all_varchar=true,parallel=false)',[str(source)]).fetchall()
    columns=[r[0] for r in header]
    if 'subject_id' not in columns:raise ValueError('Patient-linked source table required')
    kept=[name for name in columns if name in FIELDS]
    projection=['c.split','t.source_csv_record_ordinal']+['t."'+name.replace('"','""')+'"' for name in kept]
    text_fields=[name for name in ['text','chiefcomplaint','comments','icd_title','etcdescription'] if name in columns]
    for name in text_fields:projection.append('sha256(t."'+name+'") AS "'+name+'_sha256"')
    limit=''
    if source_row_limit is not None:
        if not isinstance(source_row_limit,int) or source_row_limit<1:raise ValueError('Positive fixture source-row limit required')
        limit=f' LIMIT {source_row_limit}'
    query='''COPY (SELECT '''+','.join(projection)+''' FROM (
        SELECT row_number() OVER () source_csv_record_ordinal,* FROM read_csv('''+sql_string(source)+''',
        header=true,all_varchar=true,parallel=false,ignore_errors=false)'''+limit+''') t
        JOIN cohort c ON CAST(t.subject_id AS BIGINT)=c.subject_id)
        TO '''+sql_string(target)+''' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 32768)'''
    connection.execute(query)
    summary=index_diagnostics(connection,target,kept)
    return dict(**summary,included_fields=kept,text_body_policy='hash_only' if text_fields else 'not_applicable',hashed_payload_fields=text_fields,
                source_row_limit=source_row_limit,semantics='source rows, not canonical events or qualified model targets')


def source_index(root, *, out, source):
    proposal=json.loads((root/'configs/m2_c1.json').read_text())
    if proposal['execution_authorized'] is not True or not proposal['approval_reference']:
        raise RuntimeError('M2-C1 full-cohort preprocessing awaits the explicit resource approval required by M2 scope')
    if proposal['training_authorized'] or proposal['bulk_media_encoding_authorized'] or proposal['alpha_authorized']:
        raise ValueError('Source-index authorization must not activate other experiments')
    if proposal['limits']!={'cpus':2,'memory_GiB':16,'elapsed_seconds':3530,'new_output_GiB':8,'spill_GiB':1}:
        raise ValueError('Approved C1 hard resource limits must remain fixed')
    ArtifactNamespace(root,'reduced_pilot').owned(out)
    if sha(source/'patient_splits.csv')!=proposal['patient_splits_sha256']:raise ValueError('Fixed cohort split changed')
    # A persistent exclusive claim prevents retries, including after incomplete execution.
    claim=out.parent/'m2-c1-single-approved-submission.json'
    with claim.open('x') as f:json.dump(dict(job_id=os.environ.get('SLURM_JOB_ID'),output=str(out),approval=proposal['approval_reference']),f)
    out.mkdir(exist_ok=False);limits=proposal['limits'];started=time.monotonic()
    deadline=started+limits['elapsed_seconds']
    def check_budget():
        if time.monotonic()>=deadline:raise TimeoutError('M2-C1 elapsed cap; no automatic extension')
    def bounded_sha(path):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while data:=stream.read(2**20):check_budget();digest.update(data)
        return digest.hexdigest()
    with (source/'patient_splits.csv').open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    config=json.loads((root/'configs/m2.json').read_text())
    inventory=json.loads((source/'source_inventory.json').read_text());selected=[]
    cohort_identity=json.loads((source/'cohort_manifest.json').read_text())
    for info in inventory:
        path=Path(info['path']);table=path.name.removesuffix('.gz').removesuffix('.csv')
        if (info['dataset'],table) in SOURCE_CATEGORIES or table in {'patients','d_items','d_labitems','d_icd_diagnoses','d_icd_procedures','mimic-cxr-2.0.0-metadata'}:
            selected.append(dict(info,path=str(Path(config['roots'][info['dataset']])/path.name),table=table))
    capture=out/'executed-source';capture.mkdir()
    captured={}
    for path in [root/'configs/m2_c1.json',root/'configs/m2.json',Path(__file__),root/'tools/m2_run.py',
            root/'packages/noah_baseline/m2/adapters.py',root/'packages/noah_baseline/m2/inventory.py',
            root/'packages/noah_baseline/runtime.py',root/'sbatch/m2_source_index.sh']:
        target=capture/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        captured[str(path.relative_to(root))]=sha(target)
    identity=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root)),
        job_id=os.environ.get('SLURM_JOB_ID'),config_sha256=sha(root/'configs/m2_c1.json'),
        source_inventory_sha256=sha(source/'source_inventory.json'),executed_source_sha256=captured,
        versions=config['versions'],limits=limits,approval_reference=proposal['approval_reference'])
    dump(out/'experiment_manifest.json',identity)
    connection=duckdb.connect();connection.execute('SET threads=?',[limits['cpus']])
    connection.execute("SET memory_limit='8GB'");connection.execute('SET preserve_insertion_order=false')
    connection.execute('SET temp_directory=?',[str(out/'spill')]);connection.execute('SET max_temp_directory_size=?',[str(limits['spill_GiB'])+'GiB'])
    completed=[];source_hashes=[];status='INCOMPLETE';error=None;old_limit=resource.getrlimit(resource.RLIMIT_FSIZE)
    signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
    def interrupted(signum,frame):raise TimeoutError('C1 termination signal; no retry authorized')
    signal.signal(signal.SIGTERM,interrupted)
    active_source=None;error_detail=None
    try:
        for info in selected:
            active_source=info['dataset']+'/'+info['table']
            print(json.dumps(dict(status='INDEXING_SOURCE',source=active_source,elapsed_seconds=time.monotonic()-started)),flush=True)
            check_budget();path=Path(info['path'])
            if info['dataset']!='cxr' and (path.stat().st_size!=info['bytes'] or path.stat().st_mtime_ns!=info['mtime_ns']):
                raise ValueError('Source fingerprint changed since cohort inventory')
            before=bounded_sha(path)
            if info['dataset']=='hosp' and info['table'] in {'admissions','patients'} and before!=cohort_identity[info['table']+'_sha256']:
                raise ValueError('Cohort-defining source changed')
            if 'subject_id' not in info['header']:
                after=bounded_sha(path)
                if before!=after:raise ValueError('Source dictionary changed during indexing')
                source_hashes.append(dict(dataset=info['dataset'],path=str(path),sha256=after,role='dictionary'));continue
            # Single active output file: reserve a separate spill quota and32MiB for manifests.
            used=sum((out/r['file']).stat().st_size for r in completed)
            remaining=int((limits['new_output_GiB']-limits['spill_GiB'])*2**30)-32*2**20-used
            if remaining<32*2**20:raise OSError('M2-C1 output cap; no automatic extension')
            resource.setrlimit(resource.RLIMIT_FSIZE,(remaining,old_limit[1]))
            name=info['dataset']+'-'+info['table']+'.parquet';partial=out/(name+'.partial')
            timer=threading.Timer(max(.001,deadline-time.monotonic()),connection.interrupt);timer.start()
            try:summary=write_source_index(connection,path,partial,splits)
            finally:timer.cancel();resource.setrlimit(resource.RLIMIT_FSIZE,old_limit)
            after=bounded_sha(path)
            if before!=after:raise ValueError('Source changed during indexing; partial file is not publishable')
            target=out/name
            if target.exists():raise FileExistsError('Source index collision')
            output_sha=bounded_sha(partial)
            partial.rename(target)
            source_hashes.append(dict(dataset=info['dataset'],path=str(path),sha256=after,role='indexed_source'))
            completed.append(dict(dataset=info['dataset'],table=info['table'],file=name,file_sha256=output_sha,source_sha256=after,**summary))
            dump(out/('progress-'+str(len(completed))+'.json'),dict(status='PARTIAL_INDEX',completed_sources=completed))
        coverage={}
        for dataset in sorted({r['dataset'] for r in completed}):
            paths=[str(out/r['file']) for r in completed if r['dataset']==dataset]
            coverage[dataset]=connection.execute('''SELECT split,count(*) source_rows,count(DISTINCT subject_id) patients
                FROM read_parquet(?,union_by_name=true) GROUP BY split ORDER BY split''',[paths]).fetchdf().to_dict('records')
        dump(out/'coverage.json',dict(by_dataset=coverage,semantics='source-row union, not canonical modality events',
            denominator_patients={s:sum(v==s for v in splits.values()) for s in ['train','val','test']},
            modality_mapping='hosp/icu/ed: mixed categorical and numeric candidates; note:text; cxr:CXR; ecg:ECG; echo:ECHO',
            canonical_events='NOT_COMPUTED',qualified_64_target_patients='NOT_COMPUTED',readiness='HOLD'))
        status='SOURCE_INDEX_COMPLETE_NOT_MODEL_READY'
    except Exception as exc:
        error=type(exc).__name__;error_detail=str(exc)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE,old_limit)
        dump(out/'source_inventory.json',source_hashes)
        result=dict(status=status,error=error,error_detail_private=error_detail,active_or_failed_source=active_source,elapsed_seconds=time.monotonic()-started,completed_sources=completed,
            intended_source_files=len(selected),patient_splits_sha256=proposal['patient_splits_sha256'],
            source_index_is_not_model_ready=True,readiness='HOLD',retry_authorized=False,
            output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,job_id=identity['job_id'],
            training_launched=False,bulk_encoding_launched=False)
        dump(out/'execution.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in {'completed_sources','error_detail_private'}},indent=2))
    if status!='SOURCE_INDEX_COMPLETE_NOT_MODEL_READY':raise RuntimeError('Partial M2-C1 index retained; review execution receipt')
