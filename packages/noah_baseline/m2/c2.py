"""One authorized C2 execution: accepted indexes to candidate events and queues."""
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from datetime import datetime, timezone

import pyarrow as pa

from noah_baseline.m2.adapters import Context
from noah_baseline.m2.c2_io import EVENT_SCHEMA,DEFER_SCHEMA,Shards,Queues,load_sources,source_rows,hash_json
from noah_baseline.m2.c2_mapping import map_record
from noah_baseline.m2.c2_finalize import connect,aggregate_vitals,finalize
from noah_baseline.m2.stream_index import atomic_json,digest_file


RECIPE_KEYS=['source_mappings', 'numeric', 'encounters', 'time', 'encoder_candidates', 'embedding_queues', 'windows', 'minimum_scored_targets', 'identity', 'eligibility']
FROZEN_RECIPE_SHA256='0490bec6a1999989890f218f68b398ba396881ad63338e2d8c988300f196be28'


def validate_config(config):
    if config['schema_version']!='m2-c2-candidate-v1':raise ValueError('Frozen C2 schema required')
    if config['limits']!={'cpus':4,'memory_GiB':32,'elapsed_seconds':7110,'output_GiB':48,'duckdb_memory':'1GB','spill_GiB':8}:
        raise ValueError('Unapproved C2 resource change')
    if any(config[k] for k in ['gpu_encoding_authorized','training_authorized','alpha_authorized']):raise ValueError('CPU preparation scope only')
    if config['numeric']['aggregation']['span_hours']!=2 or config['numeric']['aggregation']['stride_hours']!=1:raise ValueError('Aggregation implementation differs from recipe')
    if config['minimum_scored_targets']!=64 or config['windows']!={'length':64,'prefix':6,'overlap':16,'information_profile':'occurrence_diagnostic','terminal_gap':'unknown_masked','future_gap_at_landmark':'forbidden'}:
        raise ValueError('Unfrozen target/window contract')
    if len(config['approval_attachment_sha256'])!=64 or config['maximum_submissions']!=1:raise ValueError('Explicit single-job approval required')
    from noah_baseline.m2.adapters import SOURCE_CATEGORIES
    expected={'.'.join(key) for key in SOURCE_CATEGORIES}|{'hosp.patients','cxr.mimic-cxr-2.0.0-metadata'}
    if set(config['source_mappings'])!=expected:raise ValueError('Missing or unexpected source policy')
    if hash_json({k:config[k] for k in RECIPE_KEYS})!=FROZEN_RECIPE_SHA256:
        raise ValueError('Frozen mapping/recipe requires a new implementation revision')


def build_context(sources,splits,dictionaries,config,check):
    by={(s['dataset'],s['table']):s for s in sources};admissions={};ed={};metadata={}
    for row,ref in source_rows(by['hosp','admissions'],splits,check=check):
        hadm=int(row['hadm_id']);sid=int(row['subject_id'])
        if hadm in admissions and admissions[hadm]!=sid:raise ValueError('Admission ID has conflicting patient identity')
        admissions[hadm]=sid
    for row,ref in source_rows(by['ed','edstays'],splits,check=check):
        key=int(row['stay_id']);value=(int(row['subject_id']),int(row['hadm_id']) if row.get('hadm_id') else None)
        if key in ed and ed[key]!=value:raise ValueError('ED stay linkage ambiguity')
        ed[key]=value
    for row,ref in source_rows(by['cxr','mimic-cxr-2.0.0-metadata'],splits,check=check):metadata.setdefault(row['dicom_id'],[]).append((row,ref))
    labels={};dictionary_evidence=[]
    for d in dictionaries:
        path=Path(d['path'])
        if not path.exists():
            alt=path.with_suffix('') if path.suffix=='.gz' else path.with_suffix('.csv.gz')
            path=alt
        if digest_file(path,check)!=d['sha256']:raise ValueError('Source dictionary changed')
        dictionary_evidence.append(dict(path=str(path),sha256=d['sha256']))
        if path.name.removesuffix('.gz') not in {'d_items.csv','d_labitems.csv'}:continue
        dataset='icu' if path.name.startswith('d_items.') else 'hosp';opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',newline='') as f:
            for row in csv.DictReader(f):
                key=(dataset,row['itemid'])
                if key in labels and labels[key]!=row['label']:raise ValueError('Ambiguous item dictionary')
                labels[key]=row['label']
    if not all(('icu',i) in labels for i in config['numeric']['icu_vital_itemids']):raise ValueError('Unverified vital item mapping')
    context=Context(admissions,ed_stays=ed,item_labels=labels,
        chart_categories={i:28 for i in config['numeric']['icu_vital_itemids']},
        ed_units=config['source_mappings']['ed.vitalsign']['units'],rx_primary_source=None)
    return context,metadata,dictionary_evidence


def run(root,out):
    root=Path(root);out=Path(out);config=json.loads((root/'configs/m2_c2.json').read_text());validate_config(config)
    if config['execution_authorized'] is not True:raise RuntimeError('C2 authorization consumed or absent')
    from noah_baseline.runtime import ArtifactNamespace
    ArtifactNamespace(root,'reduced_pilot').owned(out)
    if out.exists():raise FileExistsError('Fresh C2 output namespace required')
    preflight=root/'artifacts/reduced_pilot/m2/m2-c2-preflight-20261021-06/PASS.json'
    pre=json.loads(preflight.read_text())
    if pre['status']!='PASS' or pre['config_sha256']!=digest_file(root/'configs/m2_c2.json'):raise ValueError('Passed preflight must bind exact effective configuration')
    code=pre['source_files']
    if any(digest_file(root/p)!=h for p,h in code.items()):raise ValueError('Source changed after preflight')
    base=root/'artifacts/reduced_pilot/m2';claim=base/'m2-c2-single-job-claim.json'
    atomic_json(claim,dict(job_id=os.environ.get('SLURM_JOB_ID'),started_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=pre['config_sha256'],approval=config['approval_attachment_sha256']))
    out.mkdir();start=time.monotonic();used=0;queue=None;stage='IDENTITY';completed=[];source_counts=[];failure=None
    original_limit=resource.getrlimit(resource.RLIMIT_FSIZE);signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
    def terminate(signum,frame):raise TimeoutError('C2 hard deadline; no continuation authorized')
    signal.signal(signal.SIGTERM,terminate)
    def check():
        if time.monotonic()-start>=config['limits']['elapsed_seconds']:raise TimeoutError('C2 elapsed cap')
        dynamic=sum(p.stat().st_size for p in out.glob('encoding-queues.sqlite*') if p.is_file())
        dynamic+=sum(p.stat().st_size for p in (out/'scratch').glob('*') if p.is_file())
        remaining=48*2**30-used-dynamic-64*2**20
        if remaining<=0:raise OSError('C2 aggregate output cap')
        resource.setrlimit(resource.RLIMIT_FSIZE,(remaining,original_limit[1]))
    def published(n):
        nonlocal used
        used+=n
    capture={}
    for name in code:
        dest=out/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes());capture[name]=digest_file(dest)
    used=sum(p.stat().st_size for p in out.rglob('*') if not p.name.startswith('.nfs') and p.is_file())
    atomic_json(out/'manifest.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root)),source_files=capture,
        job_id=os.environ.get('SLURM_JOB_ID'),started_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=pre['config_sha256'],patient_splits_sha256=config['patient_splits_sha256'],
        source_manifest_sha256=config['source_manifest_sha256'],effective_config=config,readiness='HOLD',model='unchanged candidate; no model loaded or trained'))
    pa.set_cpu_count(4);pa.set_io_thread_count(2)
    events=Shards(out,'events',EVENT_SCHEMA,check=check,published=published)
    deferred=Shards(out,'deferred',DEFER_SCHEMA,check=check,published=published)
    vitals=Shards(out,'vital-inputs',EVENT_SCHEMA,check=check,published=published)
    try:
        _,splits,sources,dictionaries=load_sources(root,config)
        context,metadata,dictionary_evidence=build_context(sources,splits,dictionaries,config,check)
        atomic_json(out/'dictionaries.json',dictionary_evidence)
        roots=json.loads((root/'configs/m2.json').read_text())['roots'];assets=json.loads((root/'configs/encoder_assets.json').read_text())['assets']
        recipes={}
        for kind,asset in [('text','biolord'),('specifics','biolord'),('cxr','rad_dino'),('ecg','otis'),('echo','echoprime')]:
            recipes[kind]=dict(candidate=config['encoder_candidates']['text' if kind=='specifics' else kind],
                checkpoint_sha256=assets[asset]['weight_sha256'],asset_manifest=assets[asset]['manifest_path'],
                integration_recipe_sha256=digest_file(root/'configs/m2_integration.json'),
                encoder_code_sha256={name:capture[name] for name in capture if any(x in name for x in ['encoders.py','otis.py','echo.py','media_smoke.py'])})
        queue=Queues(out,recipes)
        # Finalize smaller sources/content tasks first; labs and chart rows last.
        sources.sort(key=lambda s:(2 if s['table']=='chartevents' else 1 if s['table']=='labevents' else 0,s['dataset'],s['table']))
        for source in sources:
            stage='SOURCE:'+source['dataset']+'.'+source['table'];counts=Counter()
            for row,ref in source_rows(source,splits,batch_rows=config['batch_rows'],check=check):
                counts['source_rows']+=1
                result,failures,is_vital=map_record(row,ref,context,config,queue,metadata,roots)
                for e in result:(vitals if is_vital else events).add(e)
                for d in failures:deferred.add(d)
                counts['raw_candidate_events']+=len(result);counts['deferred_records']+=len(failures)
                if source['table'] in {'patients','mimic-cxr-2.0.0-metadata'}:counts['auxiliary_rows']+=1
            if counts['source_rows']!=source['expected_rows']:raise ValueError('C2 source consumption count mismatch')
            queue.commit();events.flush();deferred.flush();vitals.flush()
            source_counts.append(dict(dataset=source['dataset'],table=source['table'],**counts));completed.append(stage)
            atomic_json(out/('progress-'+str(len(completed))+'.json'),dict(stage=stage,sources=source_counts,
                published_events=events.count,published_vital_inputs=vitals.count,published_deferred=deferred.count,elapsed_seconds=time.monotonic()-start))
        stage='VITAL_AGGREGATION';inputs=vitals.close();c=connect(out,config)
        # Interrupt potentially long SQL operations at the same controller deadline.
        import threading
        stop=threading.Event()
        def watchdog():
            while not stop.wait(1):
                if time.monotonic()-start>=config['limits']['elapsed_seconds']:c.interrupt();return
        watcher=threading.Thread(target=watchdog,daemon=True);watcher.start()
        try:
            aggregates=aggregate_vitals(c,inputs,events,out,check=check,published=published)
            event_manifest=events.close();deferred_manifest=deferred.close();stage='OWNERSHIP_AND_COUNTS'
            if deferred_manifest['rows']:
                c.read_parquet([r['path'] for r in deferred_manifest['shards']]).create_view('deferred_records')
                from noah_baseline.m2.c2_finalize import copy_atomic
                deferred_manifest['reason_support']=copy_atomic(c,'''SELECT dataset,"table",split,reason,count(*) inventory_entries,
                    count(DISTINCT source_ordinal) source_records,count(DISTINCT subject_id) patients
                    FROM deferred_records GROUP BY dataset,"table",split,reason''',out/'deferred-reasons.parquet',check,published)
            outputs=finalize(c,event_manifest,out,splits,config,check=check,published=published)
            stage='QUEUE_EXPORT';queues=queue.export(out,check=check,published=published);check()
            atomic_json(out/'dataset-manifest.json',dict(status='CANDIDATE_CANONICAL_PREPARATION_COMPLETE_EMBEDDINGS_PENDING',
                events=event_manifest,deferred=deferred_manifest,vital_inputs=inputs,vital_aggregation=aggregates,
                encoding_queues=queues,source_counts=source_counts,**outputs,
                config_sha256=pre['config_sha256'],source_manifest_sha256=config['source_manifest_sha256'],
                patient_splits_sha256=config['patient_splits_sha256'],readiness='HOLD',strict_prospective='NOT_CERTIFIED',paper_exact='NOT_CERTIFIED'))
        finally:stop.set();c.close()
        stage='COMPLETE'
    except Exception as error:
        failure=dict(type=type(error).__name__,stage=stage)
        atomic_json(out/'failure-private.json',dict(**failure,detail=str(error)))
    finally:
        if queue is not None:queue.commit();queue.db.close()
        resource.setrlimit(resource.RLIMIT_FSIZE,original_limit)
        result=dict(status='COMPLETE_PREPARATION_NOT_MODEL_READY' if stage=='COMPLETE' else 'INCOMPLETE',stage=stage,failure=failure,
            elapsed_seconds=time.monotonic()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            output_bytes=sum(p.stat().st_size for p in out.rglob('*') if not p.name.startswith('.nfs') and p.is_file()),completed_sources=completed,
            source_counts=source_counts,published_events=events.count,published_vital_inputs=vitals.count,published_deferred=deferred.count,
            uncommitted_batches=dict(events=len(events.batch),vitals=len(vitals.batch),deferred=len(deferred.batch)),
            retry=False,encoding=False,training=False,alpha=False,readiness='HOLD')
        atomic_json(out/'execution.json',result)
    if result['status']=='INCOMPLETE':raise RuntimeError('C2 incomplete; retain atomic outputs; no automatic retry')
    return result
