"""Bounded CSV-record indexing; source records never become canonical events here.

One writer owns a source directory. Each atomic chunk is self-describing and
independently checked. Gzip resume reparses the prefix, but never rewrites it.
"""
from collections import Counter, defaultdict
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

VERSION='c1-stream-v1'
TEXT_FIELDS={'text','chiefcomplaint','comments','icd_title','etcdescription'}
DATE_FIELDS={'chartdate','storedate','StudyDate','dod','expirationdate'}
AVAILABILITY_FIELDS={'storetime','verifiedtime','entertime'}


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest_file(path, check=lambda:None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(2**20):check();h.update(data)
    return h.hexdigest()


def atomic_json(path, value):
    """Publish new immutable metadata without replacing an existing receipt."""
    path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(encode(value)+b'\n');f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path)
    finally:tmp.unlink()
    fsync_dir(path.parent)


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def clock_field(name):
    return name in DATE_FIELDS or name.endswith(('time','datetime')) or name=='StudyTime'


def precision(name,value):
    if value is None or not value.strip():return 'unknown'
    if name in DATE_FIELDS or re.fullmatch(r'\d{4}-\d{2}-\d{2}|\d{8}',value):return 'date'
    if name=='StudyTime':return 'time_of_day_component'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?',value):
        return 'fractional_second' if '.' in value else 'second'
    return 'unparsed_preserved'


def schema_for(header):
    names=['_source_csv_record_ordinal','_split','_source_row_sha256','_encounter_attribution','_availability_observation']
    names += [n+'_sha256' if n in TEXT_FIELDS else n for n in header]
    names += [n+'__precision' for n in header if clock_field(n)]
    if len(set(names))!=len(names):raise ValueError('Source/reserved column collision')
    return pa.schema([(n,pa.int64() if n=='_source_csv_record_ordinal' else pa.string()) for n in names])


def project(header, row, ordinal, split):
    values=dict(zip(header,row))
    out=dict(_source_csv_record_ordinal=ordinal,_split=split,
        _source_row_sha256=hashlib.sha256(encode(row)).hexdigest(),
        _encounter_attribution='source_hadm_id' if values.get('hadm_id','').strip() else 'unknown',
        _availability_observation='raw_clock_present' if any(values.get(n,'').strip() for n in AVAILABILITY_FIELDS) else 'unknown')
    for n,v in values.items():out[n+'_sha256' if n in TEXT_FIELDS else n]=hashlib.sha256(v.encode()).hexdigest() if n in TEXT_FIELDS else v
    for n,v in values.items():
        if clock_field(n):out[n+'__precision']=precision(n,v)
    return out


def verify_chunk(directory, schema, splits, *, expected_identity=None):
    """Re-read bounded Parquet batches; validate identities, values digest and coverage."""
    receipt=json.loads((directory/'receipt.json').read_text())
    if expected_identity and receipt['source_identity']!=expected_identity:raise ValueError('Chunk source identity mismatch')
    data=directory/'rows.parquet'
    if digest_file(data)!=receipt['parquet_sha256']:raise ValueError('Chunk file tampered')
    pf=pq.ParquetFile(data)
    if not pf.schema_arrow.equals(schema,check_metadata=False):raise ValueError('Chunk schema mismatch')
    h=hashlib.sha256();counts=Counter();patients=defaultdict(set);previous=receipt['first_source_record']-1;rows=0
    for batch in pf.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            ordinal=row['_source_csv_record_ordinal'];subject=int(row['subject_id'])
            if not previous<ordinal<=receipt['last_source_record'] or splits.get(subject)!=row['_split']:
                raise ValueError('Chunk order, patient or split mismatch')
            previous=ordinal;rows+=1;h.update(encode(row));h.update(b'\n')
            counts[row['_split']]+=1;patients[row['_split']].add(subject)
    coverage={s:sorted(v) for s,v in patients.items()}
    if (rows!=receipt['selected_rows'] or h.hexdigest()!=receipt['selected_digest']
            or dict(counts)!=receipt['split_rows'] or coverage!=receipt['split_patients']):
        raise ValueError('Chunk content, count or patient coverage mismatch')
    return receipt


def stream_source(source, out, splits, *, dataset, table, version, source_sha256=None,
        source_records_per_chunk=250000, selected_rows_per_chunk=25000, selected_bytes_per_chunk=16*2**20,
        max_source_records=None, check=lambda:None, fault=None,before_write=lambda:None,on_published=lambda p:None):
    """Index all records unless an explicit bounded verification limit is supplied.

    Complete-source status requires EOF and before/after file-hash equality.
    Interrupted commits remain provisional. Resume verifies committed chunks and
    the replayed source-prefix digest before it accepts any new range.
    """
    if any(not isinstance(n,int) or n<1 for n in [source_records_per_chunk,selected_rows_per_chunk,selected_bytes_per_chunk]):
        raise ValueError('Positive explicit chunk limits required')
    if max_source_records is not None and (not isinstance(max_source_records,int) or max_source_records<1):raise ValueError('Invalid bounded verification limit')
    if not splits or not set(splits.values())<={'train','val','test'}:raise ValueError('Invalid patient split')
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check();file_sha=digest_file(source,check)
        if source_sha256 is not None and file_sha!=source_sha256:raise ValueError('Source content changed')
        identity=dict(schema_version=VERSION,dataset=dataset,table=table,version=version,source_path=str(source.resolve()),
            source_sha256=file_sha,splits_sha256=hashlib.sha256(encode(sorted(splits.items()))).hexdigest(),
            implementation_sha256=digest_file(Path(__file__)),chunk_limits=[source_records_per_chunk,selected_rows_per_chunk,selected_bytes_per_chunk],
            bounded_verification_max_source_records=max_source_records)
        opener=gzip.open if source.suffix=='.gz' else open
        csv.field_size_limit(16*2**20)
        with opener(source,'rt',newline='',encoding='utf-8') as stream:
            reader=csv.reader(stream,strict=True);header=next(reader)
            if len(header)!=len(set(header)) or 'subject_id' not in header:raise ValueError('Invalid patient-linked CSV header')
            schema=schema_for(header);identity['source_header']=header
            source_receipt=out/'source.json'
            if source_receipt.exists():
                if json.loads(source_receipt.read_text())!=identity:raise ValueError('Resume identity/recipe changed')
            else:atomic_json(source_receipt,identity)
            chunks=out/'chunks';chunks.mkdir(exist_ok=True)
            # Interrupted temporary directories are quarantined, never consumed or overwritten.
            for p in chunks.glob('.pending-*'):
                quarantine=out/'incomplete';quarantine.mkdir(exist_ok=True);p.rename(quarantine/p.name)
            committed=sorted(p for p in chunks.iterdir() if not p.name.startswith('.'))
            expected_start=1;verified_counts=Counter();verified_patients=defaultdict(set)
            for p in committed:
                receipt=verify_chunk(p,schema,splits,expected_identity=identity)
                if receipt['first_source_record']!=expected_start:raise ValueError('Missing or overlapping source chunk')
                if receipt['last_source_record']<expected_start:raise ValueError('Invalid source chunk range')
                if receipt['source_records']!=receipt['last_source_record']-expected_start+1:raise ValueError('Invalid source range count')
                verified_counts.update(receipt['split_rows'])
                for s,ids in receipt['split_patients'].items():verified_patients[s].update(ids)
                expected_start=receipt['last_source_record']+1
            completion=out/'COMPLETE.json'
            existing_completion=completion if max_source_records is None else out/'BOUNDED.json'
            if existing_completion.exists():
                done=json.loads(existing_completion.read_text())
                if (done['source_records']!=expected_start-1 or done['chunks']!=len(committed)
                        or done['selected_rows']!=sum(verified_counts.values()) or done['split_rows']!=dict(verified_counts)
                        or done['split_patients']!={s:len(v) for s,v in verified_patients.items()} or done['source_identity']!=identity):
                    raise ValueError('Invalid completion receipt')
                return done
            start=1;ordinal=0;selected=[];size=0;digest=hashlib.sha256();count=Counter();patients=defaultdict(set)
            source_subject_index=header.index('subject_id');replayed=0;new_chunks=0

            def commit(end):
                nonlocal start,selected,size,digest,count,patients,replayed,new_chunks
                check();name=f'{start:012d}-{end:012d}';destination=chunks/name
                coverage={s:sorted(v) for s,v in patients.items()}
                current=dict(first_source_record=start,last_source_record=end,source_records=end-start+1,
                    selected_rows=len(selected),selected_digest=digest.hexdigest(),split_rows=dict(count),split_patients=coverage)
                if destination.exists():
                    old=json.loads((destination/'receipt.json').read_text())
                    if any(old[k]!=v for k,v in current.items()):raise ValueError('Resume source-prefix mismatch')
                    replayed+=1
                else:
                    # A later committed range can never be silently bypassed by a changed boundary.
                    if start<expected_start:raise ValueError('Resume chunk boundary mismatch')
                    before_write()
                    pending=chunks/('.pending-'+name+'-'+uuid.uuid4().hex);pending.mkdir()
                    if fault:fault('before_write',start)
                    path=pending/'rows.parquet'
                    pq.write_table(pa.Table.from_pylist(selected,schema=schema),path,compression='zstd',row_group_size=4096)
                    with path.open('rb') as f:os.fsync(f.fileno())
                    current.update(source_identity=identity,parquet_sha256=digest_file(path),status='VERIFIED_CHUNK_PROVISIONAL_UNTIL_SOURCE_COMPLETE')
                    atomic_json(pending/'receipt.json',current)
                    verify_chunk(pending,schema,splits,expected_identity=identity)
                    if fault:fault('before_publish',start)
                    pending.rename(destination);fsync_dir(chunks)
                    for p in destination.iterdir():p.chmod(p.stat().st_mode&~0o222)
                    on_published(destination)
                    if fault:fault('after_publish',start)
                    new_chunks+=1
                start=end+1;selected=[];size=0;digest=hashlib.sha256();count=Counter();patients=defaultdict(set)

            eof=True
            for row in reader:
                if ordinal%4096==0:check()
                if max_source_records is not None and ordinal>=max_source_records:eof=False;break
                ordinal+=1
                if len(row)!=len(header):raise ValueError('Malformed source record; no silent row loss')
                if sum(map(len,row))>32*2**20:raise ValueError('Source record exceeds explicit bounded parser budget')
                raw_subject=row[source_subject_index]
                if not re.fullmatch(r'[0-9]+',raw_subject):raise ValueError('Invalid source subject ID')
                subject=int(raw_subject)
                if subject in splits:
                    record=project(header,row,ordinal,splits[subject]);payload=encode(record)
                    if len(payload)>selected_bytes_per_chunk:raise ValueError('Oversized record retained in source; chunk budget cannot encode it')
                    if selected and size+len(payload)>selected_bytes_per_chunk:commit(ordinal-1)
                    selected.append(record);size+=len(payload);digest.update(payload);digest.update(b'\n')
                    count[splits[subject]]+=1;patients[splits[subject]].add(subject)
                if ordinal-start+1>=source_records_per_chunk or len(selected)>=selected_rows_per_chunk or size>=selected_bytes_per_chunk:
                    commit(ordinal)
            if ordinal>=start:commit(ordinal)
            if replayed!=len(committed):raise ValueError('Source ended before committed resume prefix')
            if digest_file(source,check)!=file_sha:raise ValueError('Source changed during indexing')
            receipts=[json.loads((p/'receipt.json').read_text()) for p in sorted(chunks.iterdir()) if not p.name.startswith('.')]
            counts=Counter();all_patients=defaultdict(set)
            for r in receipts:
                counts.update(r['split_rows'])
                for s,ids in r['split_patients'].items():all_patients[s].update(ids)
            complete_source=eof and max_source_records is None
            result=dict(status='SOURCE_COMPLETE_NOT_MODEL_READY' if complete_source else 'BOUNDED_CHECK_NOT_FULL_SOURCE',source_identity=identity,
                source_records=ordinal,selected_rows=sum(counts.values()),split_rows=dict(counts),
                split_patients={s:len(ids) for s,ids in all_patients.items()},chunks=len(receipts),
                completeness='EOF_AND_SOURCE_HASH_VERIFIED' if complete_source else 'BOUNDED_CHECK_ONLY',readiness='HOLD')
            atomic_json(completion if complete_source else out/'BOUNDED.json',result)
            return result
