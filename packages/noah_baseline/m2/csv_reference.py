"""Strict independent CSV references for the small CXR metadata inventory.

No source reindexing occurs here. Explicit RFC-style comma/quote settings avoid
sample-dependent DuckDB dialect detection on late quoted commas.
"""
from collections import Counter,defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from noah_baseline.m2.stream_index import atomic_json,digest_file,encode,schema_for,verify_chunk

CSV_OPTIONS="header=true,all_varchar=true,parallel=false,delim=',',quote='\"',escape='\"'"


def connection(splits):
    c=duckdb.connect()
    c.execute('SET threads=2');c.execute("SET memory_limit='256MB'");c.execute("SET max_temp_directory_size='0B'")
    c.register('cohort',pa.table({'subject_id':list(splits),'split':list(splits.values())}))
    return c


def source_counts(path,splits):
    c=connection(splits)
    try:
        rows=c.execute(f'''SELECT c.split,count(*),count(DISTINCT t.subject_id)
            FROM read_csv(?,{CSV_OPTIONS}) t JOIN cohort c
            ON CAST(t.subject_id AS BIGINT)=c.subject_id GROUP BY c.split''',[str(path)]).fetchall()
        return dict(split_rows={s:n for s,n,p in rows},split_patients={s:p for s,n,p in rows})
    finally:c.close()


def attest_existing_index(source,index,splits,out):
    """Read-only full small-source comparison; publish a separate attestation."""
    source=Path(source);index=Path(index);out=Path(out)
    if out.exists():raise FileExistsError('Post-validation receipts cannot be replaced')
    complete=json.loads((index/'COMPLETE.json').read_text());identity=complete['source_identity']
    if complete['status']!='SOURCE_COMPLETE_NOT_MODEL_READY':raise ValueError('Source EOF receipt required')
    if digest_file(source)!=identity['source_sha256']:raise ValueError('Original source changed')
    if hashlib.sha256(encode(sorted(splits.items()))).hexdigest()!=identity['splits_sha256']:raise ValueError('Patient split changed')
    header=identity['source_header'];schema=schema_for(header);chunks=[]
    def indexed_rows():
        for directory in sorted((index/'chunks').iterdir()):
            receipt=verify_chunk(directory,schema,splits,expected_identity=identity)
            chunks.append(dict(path=str(directory.relative_to(index)),sha256=receipt['parquet_sha256']))
            for batch in pq.ParquetFile(directory/'rows.parquet').iter_batches(batch_size=1024):yield from batch.to_pylist()
    c=connection(splits);counts=Counter();patients=defaultdict(set);count=0
    try:
        result=c.execute(f'''SELECT t.*,c.split FROM
            (SELECT row_number() OVER () AS __ordinal,* FROM read_csv(?,{CSV_OPTIONS})) t
            JOIN cohort c ON CAST(t.subject_id AS BIGINT)=c.subject_id ORDER BY __ordinal''',[str(source)])
        def raw_rows():
            while batch:=result.fetchmany(1024):yield from batch
        for raw,row in itertools.zip_longest(raw_rows(),indexed_rows()):
            if raw is None or row is None:raise ValueError('Independent source/index row count mismatch')
            ordinal,*values,split=raw;values=['' if v is None else v for v in values]
            if ordinal!=row['_source_csv_record_ordinal'] or split!=row['_split']:
                raise ValueError('Independent global ordinal/split mismatch')
            if hashlib.sha256(encode(values)).hexdigest()!=row['_source_row_sha256']:
                raise ValueError('Independent parsed source identity mismatch')
            for name,value in zip(header,values):
                if row[name]!=value:raise ValueError('Independent raw field preservation mismatch')
            counts[split]+=1;patients[split].add(int(row['subject_id']));count+=1
    finally:c.close()
    if counts!=Counter(complete['split_rows']) or {s:len(p) for s,p in patients.items()}!=complete['split_patients']:
        raise ValueError('Independent source completion count/coverage mismatch')
    if digest_file(source)!=identity['source_sha256']:raise ValueError('Source changed during post-validation')
    value=dict(status='PASS_POST_RUN_INDEPENDENT_VALIDATION',source_sha256=identity['source_sha256'],
        index_completion_sha256=digest_file(index/'COMPLETE.json'),selected_rows=count,split_rows=dict(counts),
        split_patients={s:len(p) for s,p in patients.items()},chunks=chunks,
        checks=['all selected raw fields','global CSV ordinals','source row hashes','exact patient splits',
                'chunk digests','whole-source selected counts and patient coverage','source before/after SHA256'],
        csv_options=CSV_OPTIONS,duckdb_memory='256MB',reindexed=False,original_job_status_unchanged=True,readiness='HOLD')
    atomic_json(out,value)
    return value
