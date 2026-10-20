"""Bounded read-only source census; candidate rows are not canonical aggregated events."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import marshal
from pathlib import Path
import subprocess
import threading
import time

import duckdb
import pandas as pd

from noah_baseline.m1.data import select_indexes
from noah_baseline.m2.cohort import select_cohort


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(root,out):
    started=time.monotonic();config=json.loads((root/'configs/m2.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    capture=out/'executed-source';capture.mkdir()
    captured={}
    for name in ('configs/m2.json','tools/m2_run.py','packages/noah_baseline/m2/inventory.py',
                 'packages/noah_baseline/m2/cohort.py','packages/noah_baseline/m1/data.py'):
        target=capture/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
        captured[name]=sha(target)
    dump(out/'executed_source.json',dict(files=captured,run_bytecode_sha256=hashlib.sha256(marshal.dumps(run.__code__)).hexdigest()))
    c=duckdb.connect(':memory:');c.execute('SET threads=2');c.execute("SET memory_limit='10GB'")
    c.execute('SET temp_directory=?',[str(out/'duckdb-temp')])
    roots={k:Path(v) for k,v in config['roots'].items()}
    source_inventory=[]
    for kind,directory in roots.items():
        if not directory.is_dir():continue
        for p in sorted(directory.iterdir()):
            if not p.name.endswith(('.csv','.csv.gz')):continue
            opener=gzip.open if p.suffix=='.gz' else open
            with opener(p,'rt',newline='') as f:header=next(csv.reader(f))
            stat=p.stat();source_inventory.append(dict(dataset=kind,version=config['versions'][kind],path=str(p),
                bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,header=header,
                identity_status='stat/header fingerprint only; complete hash required before bulk extraction'))
    dump(out/'source_inventory.json',source_inventory)
    admissions=roots['hosp']/'admissions.csv';patients=roots['hosp']/'patients.csv'
    c.execute('CREATE TABLE admissions AS SELECT * FROM read_csv_auto(?)',[str(admissions)])
    source_subjects=[r[0] for r in c.execute('SELECT DISTINCT subject_id FROM admissions').fetchall()]
    original=root/config['legacy_splits']
    assert sha(original)=='0beadca1a4580c37abff93252c57b73ca5ea7dadabb26865f1e43b1d5f0cac7e'
    with original.open(newline='') as f:protected={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    sizes={k:config['cohort'][k] for k in ('train','val','test')}
    splits=select_cohort(source_subjects,protected,sizes,config['cohort']['seed'])
    with (out/'patient_splits.csv').open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['subject_id','split','protected_pilot'])
        writer.writerows((s,splits[s],int(s in protected)) for s in sorted(splits))
    c.register('cohort',pd.DataFrame([dict(subject_id=s,split=v) for s,v in splits.items()]))
    selected=c.execute('''SELECT a.subject_id,a.hadm_id,epoch(a.admittime)/3600. admittime,
        epoch(a.dischtime)/3600. dischtime,epoch(a.deathtime)/3600. deathtime,
        a.hospital_expire_flag,a.discharge_location FROM admissions a JOIN cohort USING(subject_id)''').fetchdf()
    selected=selected.astype(object).where(pd.notnull(selected),None).to_dict('records')
    indexes=select_indexes(selected,splits)
    dump(out/'index_cohort.json',indexes)
    counts={split:dict(patients=sizes[split],admissions=sum(splits[r['subject_id']]==split for r in selected),
        eligible_24h=sum(i['split']==split and i['eligible'] for i in indexes),
        eligible_24h_deaths=sum(i['split']==split and i['eligible'] and i['label']==1 for i in indexes),
        exclusions=dict(Counter(i['exclusion'] for i in indexes if i['split']==split and not i['eligible']))) for split in sizes}
    dump(out/'cohort_manifest.json',dict(schema_version=1,cohort=config['cohort'],counts=counts,
        source_universe_patients=len(source_subjects),admissions_sha256=sha(admissions),patients_sha256=sha(patients),
        protected_split_sha256=sha(original),patient_splits_sha256=sha(out/'patient_splits.csv'),
        index_sha256=sha(out/'index_cohort.json'),source_config_sha256=sha(root/'configs/m2.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'])),
        training_token_eligibility='not yet applied; no future-density-based cohort resampling',
        outcome_policy='same M1 first-admission/+24h risk set; source counts only, no prediction or test tuning'))
    # Query census counts only. Taxonomy mappings below are candidates pending canonical event construction.
    mappings={
        ('ed','edstays'):['Enter ED','Leave ED'],('ed','triage'):['Complaint','Triage Acuity'],
        ('ed','vitalsign'):['Vitals'],('ed','diagnosis'):['Diagnosis'],('ed','medrecon'):['Drug Previously'],
        ('ed','pyxis'):['Drug Start'],('hosp','admissions'):['Enter Hospitalization','Leave Hospitalization','Death'],
        ('hosp','diagnoses_icd'):['Diagnosis'],('hosp','procedures_icd'):['Procedure'],
        ('hosp','prescriptions'):['Drug Prescription'],('hosp','pharmacy'):['Drug Prescription'],
        ('hosp','emar'):['Drug Start'],('hosp','labevents'):['Lab Test'],
        ('hosp','microbiologyevents'):['Microbiology Test'],('hosp','services'):['Service'],
        ('hosp','transfers'):['Transfer','Leave Transfer'],('icu','icustays'):['Enter ICU','Leave ICU'],
        ('icu','chartevents'):['Vitals','Other Event'],('icu','inputevents'):['Body Input','Body Input End'],
        ('icu','outputevents'):['Body Output'],('icu','procedureevents'):['Procedure','Procedure End'],
        ('note','discharge'):['Note Dischargesummary'],('note','radiology'):['Note Radiologyreport'],
        ('cxr','cxr-record-list'):['X-ray'],('ecg','record_list'):['ECG'],('echo','echo-record-list'):['ECHO']}
    census=[];samples=[]
    priority={'cxr':0,'ecg':1,'echo':2,'ed':3,'note':4,'hosp':5,'icu':6}
    for info in sorted(source_inventory,key=lambda x:(priority[x['dataset']],x['path'])):
        kind=info['dataset'];p=Path(info['path']);name=p.name.removesuffix('.gz').removesuffix('.csv')
        if (kind,name) not in mappings:continue
        if kind in {'hosp','icu'} and samples and not (out/'small_source_samples.json').exists():
            dump(out/'small_source_samples.json',samples)
        entry=dict(dataset=kind,table=name,candidate_categories=mappings[kind,name],semantics='source rows, not post-aggregation event counts')
        if time.monotonic()-started>1080 or info['bytes']>config['limits']['source_table_size_limit_bytes']:
            entry['status']='DEFERRED_RESOURCE_BOUND';census.append(entry);continue
        timer=threading.Timer(config['limits']['per_query_seconds'],c.interrupt)
        timer.start();before=time.monotonic()
        try:
            item=next((k for k in ['itemid','drug','medication','icd_code','label'] if k in info['header']),None)
            extra=f',count(DISTINCT t."{item}") distinct_specific_items' if item else ',NULL distinct_specific_items'
            sql='SELECT c.split,count(*) source_rows,count(DISTINCT t.subject_id) contributing_patients'
            if 'hadm_id' in info['header']:sql+=',count(DISTINCT t.hadm_id) linked_hospital_admissions'
            else:sql+=',NULL linked_hospital_admissions'
            sql+=extra+' FROM read_csv_auto(?, sample_size=20000, ignore_errors=false) t JOIN cohort c USING(subject_id) GROUP BY c.split ORDER BY c.split'
            result=c.execute(sql,[str(p)]).fetchdf().astype(object)
            entry['counts']=result.where(pd.notnull(result),None).to_dict('records');entry['status']='COUNTED'
            if kind in {'cxr','ecg','echo','note'} or (kind=='ed' and name in {'vitalsign','edstays'}):
                source="read_csv_auto(?,sample_size=20000,ignore_errors=false)"
                if kind=='ed' and name=='vitalsign':
                    source="(SELECT row_number() OVER() source_csv_record_ordinal,* FROM read_csv_auto(?,parallel=false,ignore_errors=false))"
                chosen=c.execute("SELECT t.* FROM "+source+" t JOIN cohort c USING(subject_id) WHERE c.split='train' LIMIT 2",[str(p)]).fetchdf().astype(object)
                rows=chosen.where(pd.notnull(chosen),None).to_dict('records')
                # Keep source content private; string timestamps preserve source representation.
                for row in rows:
                    row={k:str(v) if v is not None else None for k,v in row.items()}
                    samples.append(dict(dataset=kind,source_path=str(p),table=name,row=row))
        except Exception as exc:
            entry['status']='QUERY_INTERRUPTED_OR_SOURCE_ERROR';entry['error']=str(exc)[:250]
        finally:timer.cancel()
        entry['elapsed_seconds']=time.monotonic()-before;census.append(entry)
        dump(out/f'census-{kind}-{name}.json',entry)
        print(json.dumps({k:entry[k] for k in ['dataset','table','status','elapsed_seconds']}),flush=True)
    dump(out/'census.json',census)
    if not (out/'small_source_samples.json').exists():dump(out/'small_source_samples.json',samples)
    dump(out/'execution.json',dict(status='BOUNDED_INVENTORY_COMPLETE',elapsed_seconds=time.monotonic()-started,
        cohort_counts=counts,tables_counted=sum(r['status']=='COUNTED' for r in census),
        tables_deferred=sum(r['status']!='COUNTED' for r in census),training_launched=False,bulk_encoding_launched=False))
    print(json.dumps(dict(cohort_counts=counts,elapsed_seconds=time.monotonic()-started)),flush=True)
