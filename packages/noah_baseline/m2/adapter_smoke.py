"""Bounded two-row-per-source adapter checks; unknown clocks remain deferred."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import threading
import time

import duckdb
import pandas as pd

from noah_baseline.data.schema import EventCategory as C
from noah_baseline.m2.adapters import Context,SOURCE_CATEGORIES,adapt_row
from noah_baseline.m2.inventory import dump,sha


def adapter_smoke(root, *, out, source):
    out.mkdir(exist_ok=False);started=time.monotonic()
    capture=out/'executed-source';capture.mkdir()
    for name in ['packages/noah_baseline/m2/adapter_smoke.py','packages/noah_baseline/m2/adapters.py','configs/m2.json']:
        p=capture/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/name).read_bytes())
    config=json.loads((root/'configs/m2.json').read_text())
    inventory=json.loads((source/'source_inventory.json').read_text())
    with (source/'patient_splits.csv').open(newline='') as f:splits=list(csv.DictReader(f))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='4GB'")
    c.register('cohort',pd.DataFrame(dict(subject_id=[int(r['subject_id']) for r in splits],split=[r['split'] for r in splits])))
    admissions={}
    with (Path(config['roots']['hosp'])/'admissions.csv').open(newline='') as f:
        for row in csv.DictReader(f):admissions[int(row['hadm_id'])]=int(row['subject_id'])
    stays={}
    with (Path(config['roots']['ed'])/'edstays.csv').open(newline='') as f:
        for row in csv.DictReader(f):stays[int(row['stay_id'])]=(int(row['subject_id']),int(row['hadm_id']) if row['hadm_id'] else None)
    labels={};chart_categories={}
    known_vitals={'heart rate','respiratory rate','o2 saturation pulseoxymetry','non invasive blood pressure systolic',
        'non invasive blood pressure diastolic','non invasive blood pressure mean','temperature fahrenheit','temperature celsius'}
    for info in inventory:
        name=Path(info['path']).name.removesuffix('.gz')
        if name not in {'d_items.csv','d_labitems.csv'}:continue
        p=Path(info['path']);opener=gzip.open if p.suffix=='.gz' else open
        with opener(p,'rt',newline='') as f:
            for row in csv.DictReader(f):
                labels[info['dataset'],row['itemid']]=row['label']
                if info['dataset']=='icu' and row['label'].lower() in known_vitals:chart_categories[row['itemid']]=C.VITALS
    context=Context(admissions,stays,labels,chart_categories,{'heartrate':'bpm'},None,'retain_untimed')
    summaries=[];private=[]
    for info in inventory:
        p=Path(info['path']);dataset=info['dataset'];table=p.name.removesuffix('.gz').removesuffix('.csv')
        if (dataset,table) not in SOURCE_CATEGORIES or dataset in {'cxr','ecg','echo'}:continue
        if time.monotonic()-started>240:raise RuntimeError('Four-minute adapter-smoke wall guard')
        timer=threading.Timer(45,c.interrupt);timer.start()
        try:
            query='''SELECT t.* FROM (SELECT row_number() OVER () source_csv_record_ordinal,*
                FROM read_csv(?,header=true,all_varchar=true,parallel=false,ignore_errors=false)) t
                JOIN cohort c ON CAST(t.subject_id AS BIGINT)=c.subject_id WHERE c.split='train' LIMIT2'''.replace('LIMIT2','LIMIT 2')
            rows=c.execute(query,[str(p)]).fetchdf().astype(object)
            rows=rows.where(pd.notnull(rows),None).to_dict('records')
        finally:timer.cancel()
        events=[];deferred=[]
        for row in rows:
            row_id=str(row['source_csv_record_ordinal'])
            result=adapt_row(dataset,table,row,source_path=str(p),source_row_id='csv_record_'+row_id,
                version=config['versions'][dataset],context=context)
            events.extend(result['events']);deferred.extend(result['deferred'])
        private.append(dict(dataset=dataset,table=table,source_rows=rows,events=events,deferred=deferred))
        summaries.append(dict(dataset=dataset,table=table,source_rows=len(rows),timed_candidate_events=len(events),
            deferred_candidate_events=len(deferred),deferred_reasons=dict(Counter(r['reason'] for r in deferred)),
            categories=dict(Counter(str(r['event']['category']) for r in events)),
            observed_availability=sum(r['event']['available_at'] is not None for r in events)))
    dump(out/'private_records.json',private)
    aggregate=dict(status='PASS_BOUNDED_ADAPTER_CHECK_NOT_FULL_MAPPING_APPROVAL',
        tables=len(summaries),source_rows=sum(r['source_rows'] for r in summaries),
        timed_candidate_events=sum(r['timed_candidate_events'] for r in summaries),
        deferred_candidate_events=sum(r['deferred_candidate_events'] for r in summaries),
        source_rows_are_not_event_counts=True,tables_summary=summaries,
        mapping_classification='ASSUMED candidates; untimed/unclassified rows retained, never silently ordered',
        elapsed_seconds=time.monotonic()-started,source_sha256=sha(Path(__file__)))
    dump(out/'aggregate.json',aggregate);print(json.dumps({k:v for k,v in aggregate.items() if k!='tables_summary'},indent=2))
