"""Census of the accessible original JPEG tree with content-identical metadata."""
import csv
import gzip
import json
import os
from pathlib import Path
from collections import Counter
import time
from noah_baseline.m2.inventory import dump,sha


def cxr_census(root, *, out=None, source=None):
    start=time.monotonic();config=json.loads((root/'configs/m2.json').read_text());directory=Path(config['roots']['cxr'])
    a=source or root/'artifacts/reduced_pilot/m2/m2-a-20261020-01'
    out=out or root/'artifacts/reduced_pilot/m2/m2-cxr-physical-20261020-01';out.mkdir(exist_ok=False)
    with (a/'patient_splits.csv').open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    counters=Counter();patients={};file_lists={}
    with gzip.open(directory/'cxr-record-list.csv.gz','rt',newline='') as f:
        for row in csv.DictReader(f):
            sid=int(row['subject_id'])
            if sid not in splits:continue
            if time.monotonic()-start>270:raise RuntimeError('CXR270s bounded census limit')
            file=(directory/row['path']).with_suffix('.jpg');split=splits[sid]
            if file.parent not in file_lists:
                try:
                    with os.scandir(file.parent) as d:file_lists[file.parent]={x.name for x in d if x.is_file()}
                except FileNotFoundError:file_lists[file.parent]=set()
            found=file.name in file_lists[file.parent]
            counters[split,'index_records']+=1;counters[split,'files_present' if found else 'files_missing']+=1
            patients.setdefault((split,'index_patients'),set()).add(sid)
            patients.setdefault((split,'available_patients' if found else 'patients_with_missing_records'),set()).add(sid)
    result={s:{**{k:counters[s,k] for k in ('index_records','files_present','files_missing')},
        **{k:len(patients.get((s,k),set())) for k in ('index_patients','available_patients','patients_with_missing_records')}} for s in ('train','val','test')}
    dump(out/'aggregate.json',dict(status='PASS',counts=result,elapsed_seconds=time.monotonic()-start,source_sha256=sha(Path(__file__)),
        source_root=str(directory),scope='original JPEG physical availability; pixels not bulk decoded'))
    print(json.dumps(result,indent=2))
