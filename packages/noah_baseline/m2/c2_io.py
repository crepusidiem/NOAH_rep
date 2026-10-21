"""Bounded C2 source reads, atomic shards and content-addressed work queues."""
from collections import OrderedDict
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from noah_baseline.data.numeric import DyadicFourierEncoder
from noah_baseline.m2.stream_index import atomic_json,digest_file,encode,fsync_dir

EVENT_SCHEMA=pa.schema([(n,t) for n,t in [
 ('event_id',pa.string()),('subject_id',pa.int64()),('split',pa.string()),('encounter_id',pa.int64()),
 ('stay_id',pa.string()),('linkage_status',pa.string()),('dataset',pa.string()),('table',pa.string()),
 ('dataset_version',pa.string()),('source_path',pa.string()),('source_sha256',pa.string()),
 ('source_row_id',pa.string()),('source_ordinal',pa.int64()),('source_row_sha256',pa.string()),
 ('index_path',pa.string()),('occurred_at',pa.float64()),('raw_occurrence',pa.string()),('precision',pa.string()),
 ('available_at',pa.float64()),('observed_available_at',pa.float64()),('availability_source',pa.string()),
 ('raw_storetime',pa.string()),('category',pa.int16()),('modality',pa.string()),('specific_item',pa.string()),
 ('units',pa.string()),('raw_numeric',pa.float64()),('numeric_vector',pa.list_(pa.float32())),
 ('content_key',pa.string()),('specifics_key',pa.string()),('model_ready',pa.bool_()),
 ('auxiliary_provenance',pa.string()),('semantics',pa.string()),('transform_revision',pa.string()),('aggregate_group',pa.string())]])
DEFER_SCHEMA=pa.schema([(n,t) for n,t in [
 ('subject_id',pa.int64()),('split',pa.string()),('encounter_id',pa.string()),('dataset',pa.string()),
 ('table',pa.string()),('source_path',pa.string()),('source_sha256',pa.string()),('source_ordinal',pa.int64()),
 ('source_row_sha256',pa.string()),('index_path',pa.string()),('reason',pa.string()),('category',pa.int16()),
 ('raw_occurrence',pa.string()),('precision',pa.string())]])


def hash_json(value):return hashlib.sha256(encode(value)).hexdigest()


class Shards:
    def __init__(self,out,name,schema,*,rows=20000,check=lambda:None,published=lambda n:None):
        self.path=Path(out)/name;self.path.mkdir();self.schema=schema;self.limit=rows
        self.check=check;self.published=published;self.batch=[];self.files=[];self.count=0
    def add(self,row):
        self.batch.append(row)
        if len(self.batch)>=self.limit:self.flush()
    def flush(self):
        if not self.batch:return
        self.check();batch=self.batch
        if 'numeric_vector' in self.schema.names:
            numeric=[i for i,r in enumerate(batch) if r.get('raw_numeric') is not None]
            if numeric:
                with np.errstate(over='ignore',invalid='ignore'):
                    vectors=DyadicFourierEncoder().encode(np.array([batch[i]['raw_numeric'] for i in numeric]))
                if not np.isfinite(vectors).all():raise ValueError('Nonfinite numeric basis cannot be published')
                for i,v in zip(numeric,vectors):batch[i]['numeric_vector']=v.tolist()
        temp=self.path/('.pending-'+str(len(self.files)));temp.mkdir()
        p=temp/'rows.parquet';pq.write_table(pa.Table.from_pylist(batch,schema=self.schema),p,compression='zstd',row_group_size=4096)
        with p.open('rb') as f:os.fsync(f.fileno())
        if pq.ParquetFile(p).metadata.num_rows!=len(batch):raise ValueError('Shard readback count mismatch')
        receipt=dict(rows=len(batch),sha256=digest_file(p),schema_sha256=hash_json(str(self.schema)))
        atomic_json(temp/'receipt.json',receipt);fsync_dir(temp)
        # Account only the two published artifacts, never transient NFS unlink tombstones.
        published_bytes=p.stat().st_size+(temp/'receipt.json').stat().st_size
        final=self.path/f'{len(self.files):06d}';temp.rename(final);fsync_dir(self.path)
        self.files.append(dict(path=str(final/'rows.parquet'),**receipt));self.count+=len(batch)
        self.batch=[]
        self.published(published_bytes);self.check()
    def close(self):self.flush();return dict(rows=self.count,shards=self.files)


class Queues:
    def __init__(self,out,recipes):
        self.db=sqlite3.connect(Path(out)/'encoding-queues.sqlite');self.recipes=recipes;self.seen=OrderedDict()
        self.db.execute('CREATE TABLE tasks(id TEXT PRIMARY KEY,kind TEXT,content_sha256 TEXT,recipe_sha256 TEXT,locator TEXT,recipe TEXT)')
    def put(self,kind,content,locator):
        recipe=self.recipes[kind];rsha=hash_json(recipe);key=hash_json([kind,content,rsha])
        if key not in self.seen:
            self.db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?)',
                (key,kind,content,rsha,json.dumps(locator,sort_keys=True),json.dumps(recipe,sort_keys=True)))
            self.seen[key]=None
            if len(self.seen)>100000:self.seen.popitem(last=False)
        else:self.seen.move_to_end(key)
        return key
    def commit(self):self.db.commit()
    def export(self,out,*,check=lambda:None,published=lambda n:None):
        self.commit();schema=pa.schema([(k,pa.string()) for k in ['id','kind','content_sha256','recipe_sha256','locator','recipe']])
        result={}
        for kind,n in self.db.execute('SELECT kind,count(*) FROM tasks GROUP BY kind').fetchall():
            check();path=Path(out)/('queue-'+kind+'.parquet');temp=path.with_suffix('.pending');writer=pq.ParquetWriter(temp,schema,compression='zstd')
            cursor=self.db.execute('SELECT * FROM tasks WHERE kind=? ORDER BY id',(kind,))
            try:
                while rows:=cursor.fetchmany(4096):
                    check();writer.write_table(pa.Table.from_pylist([dict(zip(schema.names,row)) for row in rows],schema=schema))
            finally:writer.close()
            temp.rename(path);published(path.stat().st_size);check()
            result[kind]=dict(tasks=n,path=str(path),sha256=digest_file(path))
        return result


def load_sources(root,config):
    root=Path(root);manifest_path=root/config['source_manifest']
    if digest_file(manifest_path)!=config['source_manifest_sha256']:raise ValueError('Accepted C1 manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='ALL_PLANNED_SOURCE_INDEXES_ACCEPTED_NOT_MODEL_READY':raise ValueError('Accepted C1 required')
    split_path=root/config['patient_splits']
    if digest_file(split_path)!=config['patient_splits_sha256']:raise ValueError('Patient split changed')
    with split_path.open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    roots=json.loads((root/'configs/m2.json').read_text())['roots'];sources=[];dictionaries=[]
    for item in manifest['source_entries']:
        if 'table' not in item:
            dictionaries.append(dict(path=item['path'],sha256=item['source_sha256']));continue
        if item.get('status')=='DICTIONARY_HASH_VERIFIED':
            dictionaries.append(dict(path=str(Path(roots[item['dataset']])/(item['table']+'.csv.gz')),sha256=item['sha256']));continue
        if 'completion_path' in item:
            path=Path(item['completion_path'])
            if digest_file(path)!=item['completion_sha256']:raise ValueError('Source completion changed')
            complete=json.loads(path.read_text());identity=complete['source_identity'];units=[];last=0;selected=0
            if complete['status']!='SOURCE_COMPLETE_NOT_MODEL_READY':raise ValueError('Source EOF completion required')
            for receipt_path in sorted((path.parent/'chunks').glob('*/receipt.json')):
                receipt=json.loads(receipt_path.read_text())
                if receipt['source_identity']!=identity or receipt['first_source_record']!=last+1:raise ValueError('C1 receipt identity/range changed')
                last=receipt['last_source_record'];selected+=receipt['selected_rows']
                units.append(dict(path=str(receipt_path.parent/'rows.parquet'),sha256=receipt['parquet_sha256']))
            if len(units)!=complete['chunks'] or last!=complete['source_records'] or selected!=complete['selected_rows']:
                raise ValueError('C1 chunk missing or source counts changed')
            source=dict(expected_rows=selected,dataset=item['dataset'],table=item['table'],source_path=identity['source_path'],
                source_sha256=identity['source_sha256'],version=identity['version'],units=units)
        else:
            raw=Path(roots[item['dataset']])/(item['table']+'.csv')
            if not raw.exists():raw=raw.with_suffix('.csv.gz')
            source=dict(expected_rows=sum(r['source_rows'] for r in item['counts']),dataset=item['dataset'],table=item['table'],source_path=str(raw),source_sha256=item['source_sha256'],
                version='2.2',units=[dict(path=item['path'],sha256=item['sha256'])])
        sources.append(source)
    return manifest,splits,sources,dictionaries


def source_rows(source,splits,*,batch_rows=8192,check=lambda:None):
    previous=0
    for unit in source['units']:
        path=Path(unit['path']);check()
        if digest_file(path,check)!=unit['sha256']:raise ValueError('C1 index content changed')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                subject=int(row['subject_id']);split=row.get('_split',row.get('split'))
                if splits.get(subject)!=split:raise ValueError('C1 source crosses frozen patient split')
                ordinal=int(row.get('_source_csv_record_ordinal',row.get('source_csv_record_ordinal')))
                if ordinal<=previous:raise ValueError('Source ordinals duplicated or reordered')
                previous=ordinal
                yield row,dict(**{k:v for k,v in source.items() if k!='units'},index_path=str(path),
                    ordinal=ordinal,split=split,subject_id=subject,row_sha256=row.get('_source_row_sha256'))
            check()
        if digest_file(path,check)!=unit['sha256']:raise ValueError('C1 index changed during preparation')
