"""Resource-gated recovery orchestration; immutable references reuse accepted C1."""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import uuid

import duckdb
import pyarrow as pa

from noah_baseline.m2.stream_index import atomic_json,digest_file,stream_source


def recovery_plan(root, config):
    base=root/'artifacts/reduced_pilot/m2';prior=base/config['reuse_run']
    failed=json.loads((root/'reports/server/2026-10-20_m2_c1/failure.json').read_text())
    inventory=json.loads((base/config['inventory']/'source_inventory.json').read_text())
    roots=json.loads((root/'configs/m2.json').read_text())['roots']
    def order(r):
        category=('icu_chartevents' if r['table']=='chartevents' else 'icu_small') if r['dataset']=='icu' else r['dataset']
        return config['source_order'].index(category)
    remaining=[]
    for r in failed['source_status']:
        if r['status'] in {'INDEX_COMPLETE','DICTIONARY_HASH_VERIFIED'}:continue
        info=next(x for x in inventory if x['dataset']==r['dataset'] and Path(x['path']).name.removesuffix('.gz').removesuffix('.csv')==r['table'])
        remaining.append(dict(**r,path=str(Path(roots[r['dataset']])/Path(info['path']).name),
            version=info['version'],fingerprint=dict(bytes=info['bytes'],mtime_ns=info['mtime_ns']),
            dictionary='subject_id' not in info['header']))
    return prior,sorted(remaining,key=lambda r:(order(r),r['table']))


def validate_reference(result, references, dataset, table):
    expected={r['split']:r for r in references if r['dataset']==dataset and r['table']==table}
    if not expected:raise ValueError('Independent whole-source reference missing')
    for split in ['train','val','test']:
        wanted=expected.get(split,{'source_rows':0,'contributing_patients':0})
        if (result['split_rows'].get(split,0)!=int(wanted['source_rows']) or
                result['split_patients'].get(split,0)!=int(wanted['contributing_patients'])):
            raise ValueError('Independent source-row/patient coverage reference mismatch')


def recover(root, out):
    root=Path(root);out=Path(out);config=json.loads((root/'configs/m2_c1_recovery.json').read_text())
    if config.get('execution_authorized') is not True or not config.get('approval_reference'):
        raise RuntimeError('C1 recovery requires a new explicit resource approval; no output or source scan performed')
    if any(config[k] for k in ['training_authorized','bulk_media_encoding_authorized','alpha_authorized']):raise ValueError('Index-only scope required')
    if config['limits']!={'cpus':2,'memory_GiB':8,'elapsed_seconds':3530,'new_output_GiB':8,'duckdb_validation_memory':'256MB'}:
        raise ValueError('Recovery budget changed')
    if set(config['chunks'])!={'source_records_per_chunk','selected_rows_per_chunk','selected_bytes_per_chunk'}:
        raise ValueError('Full recovery forbids source-record subsampling limits')
    base=root/'artifacts/reduced_pilot/m2'
    if not out.resolve().is_relative_to(base.resolve()):raise ValueError('Recovery namespace escape')
    start=time.monotonic();limits=config['limits'];deadline=start+limits['elapsed_seconds'];pa.set_cpu_count(2);pa.set_io_thread_count(1)
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('C1 recovery deadline; no automatic continuation')
    prior,plan=recovery_plan(root,config);cohort=base/config['inventory']
    if digest_file(cohort/'patient_splits.csv')!=config['patient_splits_sha256']:raise ValueError('Patient split changed')
    inventory=base/'m2-c1-artifact-inventory-57006471.json'
    if digest_file(inventory)!=config['reuse_inventory_sha256']:raise ValueError('Historical inventory changed')
    if config['resume_directory'] is None:
        if out.exists():raise FileExistsError('Fresh recovery namespace required')
    elif out.resolve()!=(root/config['resume_directory']).resolve():raise ValueError('Unapproved resume directory')
    claim=base/('m2-c1-recovery-submission-'+hashlib.sha256(config['approval_reference'].encode()).hexdigest()[:16]+'.json')
    atomic_json(claim,dict(job_id=os.environ.get('SLURM_JOB_ID'),out=str(out),approval=config['approval_reference']))
    out.mkdir(exist_ok=config['resume_directory'] is not None)
    attempt=out/('attempt-'+uuid.uuid4().hex);attempt.mkdir()
    source_files=['configs/m2_c1_recovery.json','configs/m2.json','packages/noah_baseline/m2/stream_index.py',
        'packages/noah_baseline/m2/c1_recovery.py','tools/m2_recover.py','sbatch/m2_c1_recovery.sh']
    capture={}
    for name in source_files:
        p=attempt/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/name).read_bytes());capture[name]=digest_file(p)
    identity=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root)),job_id=os.environ.get('SLURM_JOB_ID'),
        config_sha256=digest_file(root/'configs/m2_c1_recovery.json'),patient_splits_sha256=config['patient_splits_sha256'],
        independent_census_sha256=digest_file(root/'reports/server/2026-10-20_m2/source_counts.csv'),
        original_inventory_sha256=digest_file(cohort/'source_inventory.json'),reused_inventory_sha256=digest_file(inventory),
        sources=capture,scope='source indexes only; no canonical event or prospective validity claim',readiness='HOLD')
    atomic_json(attempt/'manifest.json',identity)
    with (cohort/'patient_splits.csv').open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    with (root/'reports/server/2026-10-20_m2/source_counts.csv').open(newline='') as f:references=list(csv.DictReader(f))
    completed=[];failures=[];reused=[];old_limit=resource.getrlimit(resource.RLIMIT_FSIZE)
    signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
    def terminate(signum,frame):raise TimeoutError('C1 recovery terminated; no continuation authorized')
    signal.signal(signal.SIGTERM,terminate)
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    # The remaining quota is applied to each next chunk file; a metadata reserve
    # plus the per-source check prevents aggregate growth exceeding the hard cap.
    def quota():
        check();remaining=int(limits['new_output_GiB']*2**30)-used-64*2**20
        if remaining<1:raise OSError('Recovery output quota exhausted')
        resource.setrlimit(resource.RLIMIT_FSIZE,(remaining,old_limit[1]))
    def published(directory):
        nonlocal used
        used+=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    try:
        entries=json.loads(inventory.read_text())
        for name,expected in entries.items():
            check()
            if digest_file(base/name,check)!=expected:raise ValueError('Accepted C1 evidence changed')
        for r in json.loads((prior/'execution.json').read_text())['completed_sources']:
            source_info=next(i for i in json.loads((cohort/'source_inventory.json').read_text()) if i['dataset']==r['dataset'] and Path(i['path']).name.removesuffix('.gz').removesuffix('.csv')==r['table'])
            stat=Path(source_info['path']).stat()
            if stat.st_size!=source_info['bytes'] or stat.st_mtime_ns!=source_info['mtime_ns']:raise ValueError('Reused source fingerprint changed')
            reused.append(dict(dataset=r['dataset'],table=r['table'],path=str(prior/r['file']),
                sha256=r['file_sha256'],source_sha256=r['source_sha256'],counts=r['counts'],
                raw_clock_precision_overlay='DATE-named fields remain date precision, including midnight-formatted values; raw availability fields are not clinical availability certification'))
        reused.extend(dict(dataset=r['dataset'],path=r['path'],source_sha256=r['sha256'],status='REUSED_DICTIONARY_HASH_RECEIPT')
            for r in json.loads((prior/'source_inventory.json').read_text()) if r['role']=='dictionary')
        atomic_json(attempt/'reused_sources.json',reused)
        for item in plan:
            try:
                check();path=Path(item['path']);fingerprint=item['fingerprint'];stat=path.stat()
                if item['dataset']!='cxr' and (stat.st_size!=fingerprint['bytes'] or stat.st_mtime_ns!=fingerprint['mtime_ns']):raise ValueError('Source fingerprint changed')
                target=out/(item['dataset']+'-'+item['table'])
                if item['dictionary']:
                    before=digest_file(path,check);after=digest_file(path,check)
                    if before!=after:raise ValueError('Dictionary changed')
                    completed.append(dict(dataset=item['dataset'],table=item['table'],status='DICTIONARY_HASH_VERIFIED',sha256=after));continue
                result=stream_source(path,target,splits,dataset=item['dataset'],table=item['table'],version=item['version'],check=check,
                    before_write=quota,on_published=published,**config['chunks'])
                if result['status']!='SOURCE_COMPLETE_NOT_MODEL_READY':raise ValueError('Bounded prefix cannot complete a full source')
                independent=references
                if item['dataset']=='cxr' and item['table']=='mimic-cxr-2.0.0-metadata':
                    # This small auxiliary table was not in the original26-table census.
                    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='256MB'");c.execute("SET max_temp_directory_size='0B'")
                    c.register('cohort',pa.table({'subject_id':list(splits),'split':list(splits.values())}))
                    values=c.execute('''SELECT c.split,count(*),count(DISTINCT t.subject_id)
                        FROM read_csv(?,header=true,all_varchar=true,parallel=false) t
                        JOIN cohort c ON CAST(t.subject_id AS BIGINT)=c.subject_id GROUP BY c.split''',[str(path)]).fetchall();c.close()
                    independent=[dict(dataset=item['dataset'],table=item['table'],split=s,source_rows=n,contributing_patients=p) for s,n,p in values]
                    if digest_file(path,check)!=result['source_identity']['source_sha256']:raise ValueError('CXR metadata changed during reference')
                validate_reference(result,independent,item['dataset'],item['table'])
                # Independent bounded SQL check of the first committed chunk.
                chunk=next(iter(sorted((target/'chunks').glob('*/rows.parquet'))),None)
                if chunk:
                    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='256MB'");c.execute("SET max_temp_directory_size='0B'")
                    actual=c.execute('SELECT _split,count(*),count(DISTINCT subject_id) FROM read_parquet(?) GROUP BY _split',[str(chunk)]).fetchall();c.close()
                    receipt=json.loads((chunk.parent/'receipt.json').read_text())
                    if {s:(n,p) for s,n,p in actual}!={s:(n,len(receipt['split_patients'][s])) for s,n in receipt['split_rows'].items()}:raise ValueError('Independent chunk SQL mismatch')
                completed.append(dict(dataset=item['dataset'],table=item['table'],result=result,reference='PASS_INDEPENDENT_WHOLE_SOURCE_COUNTS_AND_PATIENT_COVERAGE'))
            except (TimeoutError,OSError):raise
            except Exception as error:
                used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
                failures.append(dict(dataset=item['dataset'],table=item['table'],error=type(error).__name__))
                atomic_json(attempt/('failure-'+item['dataset']+'-'+item['table']+'.json'),dict(type=type(error).__name__,detail_private=str(error)))
            atomic_json(attempt/('progress-'+str(len(completed)+len(failures))+'.json'),dict(completed=completed,failures=failures))
    except Exception as error:
        failures.append(dict(scope='global',error=type(error).__name__))
        atomic_json(attempt/'global-error.json',dict(type=type(error).__name__,detail_private=str(error)))
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE,old_limit)
        result=dict(status='SOURCE_INDEX_COMPLETE_NOT_MODEL_READY' if len(completed)==len(plan) and not failures else 'INCOMPLETE',
            reused=reused,completed=completed,failures=failures,planned_remaining_sources=len(plan),elapsed_seconds=time.monotonic()-start,
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,readiness='HOLD',training=False,encoding=False,retry=False)
        atomic_json(attempt/'execution.json',result)
    if result['status']=='INCOMPLETE':raise RuntimeError('Recovery incomplete; preserved outputs; no automatic retry')
    return result
