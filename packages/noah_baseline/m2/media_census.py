"""Bounded file-existence census distinguishes metadata references from accessible media."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

from noah_baseline.m2.inventory import dump,sha


def media_census(root, *, out=None, source=None):
    start=time.monotonic();config=json.loads((root/'configs/m2.json').read_text())
    a=source or root/'artifacts/reduced_pilot/m2/m2-a-20261020-01'
    out=out or root/'artifacts/reduced_pilot/m2/m2-media-census-20261020-02';out.mkdir(exist_ok=False)
    with (a/'patient_splits.csv').open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    records={'cxr':'cxr-record-list.csv.gz','ecg':'record_list.csv','echo':'echo-record-list.csv'}
    results={};examples=[]
    for kind,name in records.items():
        directory=Path(config['cxr_previous_derivative'] if kind=='cxr' else config['roots'][kind]);p=directory/name;opener=gzip.open if p.suffix=='.gz' else open
        shards={v.name for v in (directory/'files').iterdir() if v.is_dir()}
        file_lists={}
        def available(file):
            try:parts=file.relative_to(directory).parts
            except ValueError:return False
            if len(parts)>1 and parts[0]=='files' and parts[1] not in shards:return False
            if file.parent not in file_lists:
                try:file_lists[file.parent]={v.name for v in file.parent.iterdir() if v.is_file()}
                except FileNotFoundError:file_lists[file.parent]=set()
            return file.name in file_lists[file.parent]
        counters=Counter();patients={};available_examples=[];missing_examples=[]
        with opener(p,'rt',newline='') as f:
            for row in csv.DictReader(f):
                sid=int(row['subject_id'])
                if sid not in splits:continue
                if time.monotonic()-start>270:raise RuntimeError('Media census270s cap reached; no bulk file read')
                split=splits[sid]
                if kind=='cxr':
                    file=(directory/row['path']).with_suffix('.png')
                    exists=available(file)
                    if not exists:
                        alternative=file.with_suffix('.jpg')
                        if available(alternative):file=alternative;exists=True
                elif kind=='ecg':
                    file=directory/row['path'];exists=available(file.with_suffix('.hea')) and available(file.with_suffix('.dat'))
                else:
                    file=directory/row['dicom_filepath'];exists=available(file)
                counters[(split,'index_records')]+=1;counters[(split,'files_present' if exists else 'files_missing')]+=1
                patients.setdefault((split,'index_patients'),set()).add(sid)
                patients.setdefault((split,'available_patients' if exists else 'patients_with_missing_records'),set()).add(sid)
                target=available_examples if exists else missing_examples
                if split=='train' and len(target)<2:target.append(dict(dataset=kind,table=name.removesuffix('.gz').removesuffix('.csv'),source_path=str(p),row=row,physical_file=str(file),file_present=exists))
        results[kind]={split:{**{label:counters[split,label] for label in ('index_records','files_present','files_missing')},
            **{label:len(patients.get((split,label),set())) for label in ('index_patients','available_patients','patients_with_missing_records')}} for split in ('train','val','test')}
        examples.extend(available_examples+missing_examples)
        dump(out/(kind+'-counts.json'),results[kind]);print(json.dumps({kind:results[kind]}),flush=True)
    dump(out/'representative_sources.json',examples)
    dump(out/'aggregate.json',dict(status='PASS',counts=results,elapsed_seconds=time.monotonic()-start,
        source_sha256=sha(Path(__file__)),scope='existence checks only; no bulk media decoding or encoding'))
