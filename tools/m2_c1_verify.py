"""Inexpensive C1 recovery checks: real prefixes only, no full-source controller."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root/'packages'))
    if not sys.flags.isolated or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use isolated Python inside the existing CPU allocation')
    for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='2'
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['stream','legacy']);p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--production-chunks',action='store_true');p.add_argument('--remaining-prefixes',action='store_true')
    args=p.parse_args();base=root/'artifacts/reduced_pilot/m2';start=time.monotonic()
    with (base/'m2-a-20261020-01/patient_splits.csv').open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    from noah_baseline.m2.stream_index import stream_source,encode,digest_file
    import pyarrow.parquet as pq
    if args.remaining_prefixes:
        from noah_baseline.m2.stream_index import TEXT_FIELDS
        import duckdb
        args.out.mkdir(exist_ok=False);roots=json.loads((root/'configs/m2.json').read_text())['roots'];checks=[];csv.field_size_limit(16*2**20)
        for dataset,table,limit in [('cxr','cxr-record-list',5000),('cxr','mimic-cxr-2.0.0-metadata',5000),('ecg','record_list',5000),('echo','echo-record-list',5000),('note','discharge',500),('note','radiology',500)]:
            source=Path(roots[dataset])/(table+'.csv')
            if not source.exists():source=source.with_suffix('.csv.gz')
            stat=source.stat();prefix=args.out/(dataset+'-'+table+'.csv');opener=gzip.open if source.suffix=='.gz' else open
            with opener(source,'rt',newline='') as f,prefix.open('x',newline='') as target:
                reader=csv.reader(f,strict=True);header=next(reader);writer=csv.writer(target,lineterminator='\n');writer.writerow(header)
                n=0
                for row in reader:
                    writer.writerow(row);n+=1
                    if n==limit:break
            assert (stat.st_size,stat.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns)
            out=args.out/(dataset+'-'+table)
            result=stream_source(prefix,out,splits,dataset=dataset+'_bounded_prefix',table=table,version='source prefix only')
            actual=[r for q in sorted(out.glob('chunks/*/rows.parquet')) for r in pq.read_table(q).to_pylist()]
            c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='256MB'");c.execute("SET max_temp_directory_size='0B'")
            c.execute('CREATE TABLE cohort(subject_id BIGINT,split VARCHAR)');c.executemany('INSERT INTO cohort VALUES (?,?)',list(splits.items()))
            ref=c.execute('''SELECT c.split,t.* FROM (SELECT row_number() OVER() ordinal,* FROM read_csv(?,header=true,all_varchar=true,parallel=false,nullstr='__C1_NO_NULL_SENTINEL__')) t JOIN cohort c ON CAST(t.subject_id AS BIGINT)=c.subject_id ORDER BY ordinal''',[str(prefix)])
            names=[v[0] for v in ref.description];count=0
            while values:=ref.fetchmany(1024):
                for values_row in values:
                    raw=dict(zip(names,values_row));got=actual[count];count+=1
                    assert got['_source_csv_record_ordinal']==raw['ordinal'] and got['_split']==raw['split']
                    for field in header:
                        if field in TEXT_FIELDS:assert got[field+'_sha256']==hashlib.sha256(raw[field].encode()).hexdigest()
                        else:assert got[field]==raw[field]
            assert count==len(actual)==result['selected_rows'];c.close()
            checks.append(dict(dataset=dataset,table=table,prefix_records=n,selected_rows=count,split_rows=result['split_rows'],split_patients=result['split_patients'],independent_raw_field_and_text_hash_reference='PASS',full_original_source_indexed=False))
        result=dict(status='PASS_BOUNDED_REMAINING_PREFIXES',checks=checks,elapsed_seconds=time.monotonic()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.out/'aggregate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    if args.worker:
        if args.worker=='stream':
            count=0
            def fault(stage,ordinal):
                nonlocal count
                if stage=='after_publish' and count==0:count+=1;raise RuntimeError('intentional real-prefix resume check')
            options=dict(dataset='icu_bounded_prefix',table='chartevents',version='2.2 prefix only',source_records_per_chunk=50000,
                selected_rows_per_chunk=5000,selected_bytes_per_chunk=4*2**20)
            if args.production_chunks:options.update(source_records_per_chunk=250000,selected_rows_per_chunk=25000,selected_bytes_per_chunk=16*2**20)
            try:stream_source(args.source,args.out,splits,fault=fault,**options)
            except RuntimeError as error:
                if str(error)!='intentional real-prefix resume check':raise
            before={str(x.relative_to(args.out)):digest_file(x) for x in args.out.glob('chunks/*/*')}
            result=stream_source(args.source,args.out,splits,**options)
            assert all(digest_file(args.out/n)==d for n,d in before.items())
            result['resume_preserved_chunks']=True
        else:
            import duckdb,pandas as pd
            from noah_baseline.m2.source_index import write_source_index
            c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='256MB'");c.execute("SET max_temp_directory_size='0B'");c.execute('SET preserve_insertion_order=false')
            stages=[]
            class Connection:
                def register(self,*a,**k):return c.register(*a,**k)
                def execute(self,sql,*a,**k):
                    stage='COPY_TO_PARQUET' if sql.startswith('COPY') else 'DESCRIBE' if sql.startswith('DESCRIBE') else 'POST_WRITE_DIAGNOSTIC'
                    stages.append(stage);return c.execute(sql,*a,**k)
            try:
                args.out.mkdir();summary=write_source_index(Connection(),args.source,args.out/'legacy.parquet',splits)
                result=dict(status='PASS_BOUNDED_LEGACY',stages=stages,counts=summary['counts'])
            except Exception as error:result=dict(status='FAILED_BOUNDED_LEGACY',stages=stages,error=type(error).__name__,detail_private=str(error))
        result.update(elapsed_seconds=time.monotonic()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps(result));return
    args.out.mkdir(exist_ok=False);capture=args.out/'executed-source';capture.mkdir()
    for name in ['tools/m2_c1_verify.py','packages/noah_baseline/m2/stream_index.py','packages/noah_baseline/m2/c1_recovery.py','configs/m2_c1_recovery.json','tests/test_m2_stream_index.py']:
        target=capture/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
    roots=json.loads((root/'configs/m2.json').read_text())['roots'];source=Path(roots['icu'])/'chartevents.csv.gz'
    prefixes={50000:args.out/'chartevents-first50000.csv',500000:args.out/'chartevents-first500000.csv'}
    handles={n:path.open('x',newline='') for n,path in prefixes.items()};writers={n:csv.writer(f,lineterminator='\n') for n,f in handles.items()}
    before=source.stat()
    try:
        with gzip.open(source,'rt',newline='') as f:
            reader=csv.reader(f);header=next(reader)
            for writer in writers.values():writer.writerow(header)
            for ordinal,row in enumerate(reader,1):
                for n,writer in writers.items():
                    if ordinal<=n:writer.writerow(row)
                if ordinal==500000:break
    finally:
        for handle in handles.values():handle.close()
    after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    reports=[]
    for n,path in prefixes.items():
        child=subprocess.run([sys.executable,'-I',__file__,'--worker','stream','--source',str(path),'--out',str(args.out/f'stream-{n}')],capture_output=True,text=True,timeout=120,check=True)
        report=json.loads(child.stdout);report['prefix_source_records']=n;report['prefix_bytes']=path.stat().st_size
        # Separate SQL parser and independent field-by-field raw reference.
        import duckdb
        c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='256MB'");c.execute("SET max_temp_directory_size='0B'")
        c.execute('CREATE TABLE cohort(subject_id BIGINT,split VARCHAR)');c.executemany('INSERT INTO cohort VALUES (?,?)',list(splits.items()))
        query='''SELECT c.split,t.* FROM (SELECT row_number() OVER() ordinal,* FROM read_csv(?,header=true,all_varchar=true,parallel=false,nullstr='__C1_NO_NULL_SENTINEL__')) t JOIN cohort c ON CAST(t.subject_id AS BIGINT)=c.subject_id ORDER BY ordinal'''
        reference=c.execute(query,[str(path)]);names=[v[0] for v in reference.description];actual=[]
        for q in sorted((args.out/f'stream-{n}/chunks').glob('*/rows.parquet')):actual.extend(pq.read_table(q).to_pylist())
        count=0;patients={};expected_counts={}
        while batch:=reference.fetchmany(1024):
            for values in batch:
                row=dict(zip(names,values));got=actual[count];count+=1
                assert got['_split']==row['split'] and got['_source_csv_record_ordinal']==row['ordinal']
                for field in header:assert got[field]==row[field],field
                assert got['_source_row_sha256']==hashlib.sha256(encode([row[field] for field in header])).hexdigest()
                expected_counts[row['split']]=expected_counts.get(row['split'],0)+1;patients.setdefault(row['split'],set()).add(row['subject_id'])
        assert count==len(actual)==report['selected_rows'] and expected_counts==report['split_rows']
        assert {s:len(v) for s,v in patients.items()}==report['split_patients'];c.close()
        report['independent_raw_field_ordinal_split_count_coverage_reference']='PASS';reports.append(report)
    legacy=subprocess.run([sys.executable,'-I',__file__,'--worker','legacy','--source',str(prefixes[500000]),'--out',str(args.out/'legacy-500000')],capture_output=True,text=True,timeout=90,check=True)
    legacy_result=json.loads(legacy.stdout);(args.out/'legacy-private.json').write_text(json.dumps(legacy_result,indent=2))
    public_legacy={k:v for k,v in legacy_result.items() if k!='detail_private'}
    result=dict(status='PASS_BOUNDED_PREFIX_CHECKS',full_source_indexing=False,source='icu/chartevents first500000 CSV records only',
        original_source_identity='original stat fingerprint only; prefix copies have separate full hashes, not original full-source hashes',
        prefix_checks=reports,legacy_256MB_check=public_legacy,elapsed_seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'])
    (args.out/'aggregate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
