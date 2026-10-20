"""Inexpensive actual-record identities, media readability and optional bounded lab census."""
from collections import Counter
from datetime import datetime
import csv
import gzip
import json
from pathlib import Path
import time
import threading
from dataclasses import asdict

import duckdb
import numpy as np
import pandas as pd
from PIL import Image
import wfdb

from noah_baseline.m1.data import hours
from noah_baseline.m2.inventory import dump,sha
from noah_baseline.m2.events import Event,EmbeddingCache
from noah_baseline.data.numeric import DyadicFourierEncoder


def source_smoke(root, *, out=None, source=None):
    start=time.monotonic();config=json.loads((root/'configs/m2.json').read_text())
    a=source or root/'artifacts/reduced_pilot/m2/m2-a-20261020-01'
    out=out or root/'artifacts/reduced_pilot/m2/m2-source-smoke-20261020-03'
    out.mkdir(exist_ok=False);samples=json.loads((a/'small_source_samples.json').read_text())
    c=duckdb.connect(':memory:');c.execute('SET threads=2');c.execute("SET memory_limit='4GB'")
    admission_path=Path(config['roots']['hosp'])/'admissions.csv'
    c.execute('CREATE TABLE admissions AS SELECT subject_id,hadm_id,admittime,dischtime FROM read_csv_auto(?)',[str(admission_path)])
    cxr_ids={s['row']['dicom_id'] for s in samples if s['dataset']=='cxr'};metadata={}
    meta=Path(config['roots']['cxr'])/'mimic-cxr-2.0.0-metadata.csv.gz'
    with gzip.open(meta,'rt',newline='') as f:
        for row in csv.DictReader(f):
            if row['dicom_id'] in cxr_ids:
                if row['dicom_id'] in metadata:raise ValueError('Duplicate CXR identity')
                metadata[row['dicom_id']]=row
    events=[];checks=[];cache=EmbeddingCache(root);dataset_sha=sha(a/'source_inventory.json')
    def number(value):return int(float(value)) if value not in (None,'','nan') else None
    for s in samples:
        r=s['row'];kind=s['dataset'];sid=number(r['subject_id']);hadm=number(r.get('hadm_id'))
        availability=None;av_source='unknown';units=None;specific=None;source_id=None
        if kind=='ed' and s['table']=='edstays':
            occurrence=hours(r['intime']);modality='categorical';category=22;source_id=r['stay_id']+':intime'
        elif kind=='ed':
            if r.get('heartrate') is None:continue
            occurrence=hours(r['charttime']);modality='numeric';category=28;units='bpm';specific='heartrate'
            source_id='csv_record_'+r['source_csv_record_ordinal']+':heartrate'
            # ED stay to hospital mapping is source-observed, never an interval guess.
            ed=Path(config['roots']['ed'])/'edstays.csv'
            linked=c.execute('SELECT hadm_id FROM read_csv_auto(?) WHERE subject_id=? AND stay_id=?',[str(ed),sid,number(r['stay_id'])]).fetchall()
            if len(linked)!=1:raise ValueError('ED source join not unique')
            hadm=linked[0][0]
            value=DyadicFourierEncoder().encode(float(r['heartrate']))
            ref=cache.put(value,kind='numeric',content_sha256=__import__('hashlib').sha256(str(r['heartrate']).encode()).hexdigest(),
                dataset_sha256=dataset_sha,checkpoint_sha256=sha(root/'packages/noah_baseline/data/numeric.py'),
                recipe={'transform_revision':'m2-numeric-m1-candidate-v1','item':'heartrate','units':'bpm','normalization':'none; unchanged M1','phase':1.,'scales':[-7,14],'layout':'sin then cos'})
            np.testing.assert_array_equal(cache.get(ref),value)
            checks.append(dict(kind='numeric',source_id=source_id,shape=[44],cache=ref))
        elif kind=='note':
            occurrence=hours(r['charttime']);availability=hours(r.get('storetime'));av_source='storetime' if availability is not None else 'unknown'
            modality='text';category=6 if s['table']=='discharge' else 3;source_id=r['note_id'];specific=r.get('note_type')
        elif kind=='cxr':
            m=metadata[r['dicom_id']]
            if number(m['subject_id'])!=sid or number(m['study_id'])!=number(r['study_id']):raise ValueError('CXR metadata identity join mismatch')
            if not m['StudyTime']:raise ValueError('Unknown CXR time cannot be silently midnight')
            rawtime=m['StudyTime'].split('.')[0].zfill(6)
            occurrence=hours(datetime.strptime(m['StudyDate']+rawtime,'%Y%m%d%H%M%S').isoformat(' '))
            modality='cxr';category=1;source_id=r['dicom_id']
            path=(Path(config['roots']['cxr'])/r['path']).with_suffix('.jpg')
            with Image.open(path) as im:
                im.load();checks.append(dict(kind='cxr',source_id=source_id,mode=im.mode,size=list(im.size),file_sha256=sha(path),physical_decode='PASS',resize_provenance='UNKNOWN'))
        elif kind=='ecg':
            occurrence=hours(r['ecg_time']);modality='ecg';category=0;source_id=r['study_id']
            path=Path(config['roots']['ecg'])/r['path']
            try:
                waveform,info=wfdb.rdsamp(str(path))
                checks.append(dict(kind='ecg',source_id=source_id,physical_decode='PASS',shape=list(waveform.shape),fs=info['fs'],leads=info['sig_name'],finite=bool(np.isfinite(waveform).all()),encoder='NOT_RUN_NO_VERIFIED_OTIS'))
            except Exception as exc:checks.append(dict(kind='ecg',source_id=source_id,physical_decode='FAIL',error=type(exc).__name__))
        elif kind=='echo':
            occurrence=hours(r['acquisition_datetime']);modality='echo';category=2;source_id=r['dicom_filepath']
            path=Path(config['roots']['echo'])/r['dicom_filepath']
            checks.append(dict(kind='echo',source_id=source_id,source_file_exists=path.is_file(),bytes=path.stat().st_size if path.is_file() else None,decode='NOT_RUN_NO_DICOM_DECODER',encoder='NOT_RUN_NO_ECHOPRIME'))
        else:continue
        if hadm is not None:
            if c.execute('SELECT count(*) FROM admissions WHERE subject_id=? AND hadm_id=?',[sid,hadm]).fetchone()[0]!=1:
                raise ValueError('Cross-patient source encounter link')
            linkage='source'
        else:
            matches=c.execute('SELECT hadm_id FROM admissions WHERE subject_id=? AND epoch(admittime)/3600<=? AND epoch(dischtime)/3600>=?',[sid,occurrence,occurrence]).fetchall()
            hadm=matches[0][0] if len(matches)==1 else None
            linkage='unique_interval' if len(matches)==1 else 'ambiguous' if matches else 'unlinked'
        event=Event(f'{kind}:{source_id}',sid,hadm,linkage,kind,config['versions'][kind],source_id,s['source_path'],occurrence,
            availability,av_source,modality,category,specific,units,'m2-source-smoke-v1').validate()
        events.append(asdict(event))
    dump(out/'events.json',events);dump(out/'private_checks.json',checks)
    dump(out/'aggregate.json',dict(status='PASS',events_by_modality=dict(Counter(e['modality'] for e in events)),
        encounter_links=dict(Counter(e['linkage_status'] for e in events)),availability=dict(Counter(e['availability_source'] for e in events)),
        numeric_cache_replay='PASS',cxr_physical_decodes=sum(r['kind']=='cxr' and r.get('physical_decode')=='PASS' for r in checks),
        ecg_physical_decodes=sum(r['kind']=='ecg' and r.get('physical_decode')=='PASS' for r in checks),
        echo_files_present=sum(r['kind']=='echo' and r.get('source_file_exists') for r in checks),
        elapsed_seconds=time.monotonic()-start,encoders_run=False,source_sha256=sha(Path(__file__))))
