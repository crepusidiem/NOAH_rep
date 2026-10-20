"""Streaming, columnar source index. This does not produce model-ready events.

Untimed records and date precision survive unchanged. Large note bodies are
represented by hashes and row identities. Execution requires a separate budget
approval; the bounded helper is independently testable on fictional CSV input.
"""
import csv
import hashlib
import json
from pathlib import Path
import resource
import signal
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
    counts=connection.execute('''SELECT split,count(*) source_rows,count(DISTINCT subject_id) patients
        FROM read_parquet(?) GROUP BY split ORDER BY split''',[str(target)]).fetchdf().to_dict('records')
    return dict(counts=counts,included_fields=kept,text_body_policy='hash_only' if text_fields else 'not_applicable',hashed_payload_fields=text_fields,
                source_row_limit=source_row_limit,semantics='source rows, not canonical events or qualified model targets')


def source_index(root, *, out, source):
    proposal=json.loads((root/'configs/m2_c1.json').read_text())
    if proposal['execution_authorized'] is not True or not proposal['approval_reference']:
        raise RuntimeError('M2-C1 full-cohort preprocessing awaits the explicit resource approval required by M2 scope')
    if proposal['training_authorized'] or proposal['bulk_media_encoding_authorized'] or proposal['alpha_authorized']:
        raise ValueError('Source-index authorization must not activate other experiments')
    ArtifactNamespace(root,'reduced_pilot').owned(out)
    if sha(source/'patient_splits.csv')!=proposal['patient_splits_sha256']:raise ValueError('Fixed cohort split changed')
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
    for path in [root/'configs/m2_c1.json',root/'configs/m2.json',Path(__file__),root/'tools/m2_run.py']:
        target=capture/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    connection=duckdb.connect();connection.execute('SET threads=?',[limits['cpus']])
    connection.execute("SET memory_limit='8GB'");connection.execute('SET preserve_insertion_order=false')
    connection.execute('SET temp_directory=?',[str(out/'spill')]);connection.execute('SET max_temp_directory_size=?',[str(limits['spill_GiB'])+'GiB'])
    completed=[];source_hashes=[];status='INCOMPLETE';error=None;old_limit=resource.getrlimit(resource.RLIMIT_FSIZE)
    signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
    try:
        for info in selected:
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
            completed.append(dict(file=name,file_sha256=output_sha,source_sha256=after,**summary))
            dump(out/('progress-'+str(len(completed))+'.json'),dict(status='PARTIAL_INDEX',completed_sources=completed))
        status='SOURCE_INDEX_COMPLETE_NOT_MODEL_READY'
    except Exception as exc:
        error=type(exc).__name__
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE,old_limit)
        dump(out/'source_inventory.json',source_hashes)
        result=dict(status=status,error=error,elapsed_seconds=time.monotonic()-started,completed_sources=completed,
            intended_source_files=len(selected),patient_splits_sha256=proposal['patient_splits_sha256'],
            source_index_is_not_model_ready=True,training_launched=False,bulk_encoding_launched=False)
        dump(out/'execution.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='completed_sources'},indent=2))
    if status!='SOURCE_INDEX_COMPLETE_NOT_MODEL_READY':raise RuntimeError('Partial M2-C1 index retained; review execution receipt')
