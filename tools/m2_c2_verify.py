"""Bounded post-run C2 artifact, numeric and ownership replay; no preprocessing."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    if not sys.flags.isolated or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Existing CPU allocation required')
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    import json
    import sqlite3
    import time
    import numpy as np
    import pyarrow.parquet as pq
    from noah_baseline.m2.stream_index import atomic_json,digest_file
    run=root/'artifacts/reduced_pilot/m2/m2-c2-canonical-20261020-01';manifest=run/'dataset-manifest.json'
    data=json.loads(manifest.read_text());execution=json.loads((run/'execution.json').read_text())
    if execution['status']!='COMPLETE_PREPARATION_NOT_MODEL_READY':raise ValueError('Cannot certify incomplete C2')
    out=root/'artifacts/reduced_pilot/m2/m2-c2-replay-20261021-01';out.mkdir(exist_ok=False)
    capture=out/'executed-source';capture.mkdir();(capture/'m2_c2_verify.py').write_bytes(Path(__file__).read_bytes())
    start=time.monotonic();files=0
    def check():
        if time.monotonic()-start>240:raise TimeoutError('Post-run verification cap, no bulk continuation')
    for key in ['events','deferred','vital_inputs','candidate_windows']:
        count=0
        for shard in data[key]['shards']:
            check();p=Path(shard['path'])
            if digest_file(p,check)!=shard['sha256']:raise ValueError('Published C2 artifact changed')
            count+=pq.ParquetFile(p).metadata.num_rows;files+=1
        if count!=data[key]['rows']:raise ValueError('Manifest count disagrees with actual retained shards')
    source_count=sum(s['source_rows'] for s in data['source_counts'])
    if source_count!=46625949 or len(data['source_counts'])!=28:raise ValueError('Not all accepted C1 rows accounted for')
    q=sqlite3.connect('file:'+str(run/'encoding-queues.sqlite')+'?mode=ro',uri=True)
    numeric=0;bindings=0;windows=0
    try:
        shards=data['events']['shards'];selected=sorted({0,len(shards)//2,len(shards)-1})
        for i in selected:
            batch=next(pq.ParquetFile(shards[i]['path']).iter_batches(batch_size=128))
            for row in batch.to_pylist():
                if row['modality']=='numeric':
                    arg=np.float32(row['raw_numeric'])*np.power(np.float32(2),np.arange(-7,15,dtype=np.float32))
                    expected=np.concatenate([np.sin(arg),np.cos(arg)])
                    actual=np.asarray(row['numeric_vector'],dtype=np.float32)
                    if actual.shape!=(44,) or not np.isfinite(actual).all():raise ValueError('Invalid saved numeric representation')
                    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7);numeric+=1
                for kind,key in [('specifics',row['specifics_key']),(row['modality'],row['content_key'])]:
                    if key:
                        task=q.execute('SELECT kind,content_sha256 FROM tasks WHERE id=?',(key,)).fetchone()
                        if task is None or task[0]!=kind or row['model_ready']:raise ValueError('Pending event/cache task binding violated')
                        bindings+=1
                check()
        shards=data['candidate_windows']['shards']
        for i in sorted({0,len(shards)//2,len(shards)-1}):
            batch=next(pq.ParquetFile(shards[i]['path']).iter_batches(batch_size=128))
            for row in batch.to_pylist():
                expected=row['event_ids'][0 if row['window_id']==0 else 16:]
                if expected!=row['owned_target_ids'] or len(expected)!=row['owned_targets'] or not row['last_gap_unknown']:
                    raise ValueError('Saved shifted target/window terminal-gap contract violated')
                windows+=1;check()
    finally:q.close()
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root)),
        verification_source_sha256=digest_file(capture/'m2_c2_verify.py'),status='PASS_ARTIFACT_AND_BOUNDED_VALUE_OWNERSHIP_REPLAY',c2_dataset_manifest_sha256=digest_file(manifest),
        source_table_rows=source_count,source_tables=28,hashed_shards=files,numeric_sample_rows=numeric,
        pending_bindings_checked=bindings,window_samples=windows,elapsed_seconds=time.monotonic()-start,
        scope='All published shard hashes/counts; first128 rows of first/middle/last event and window shards; independent numeric/shifted-index oracle; not full author-ETL fidelity',
        job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),readiness='HOLD')
    atomic_json(out/'verification.json',result);assert_local_modules(root);print(json.dumps(result))


if __name__=='__main__':main()
