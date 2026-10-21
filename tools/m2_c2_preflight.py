"""Inexpensive fixture receipt plus bounded real C1 mapping/source checks."""
from pathlib import Path
import os
import sys


def main():
    if not sys.flags.isolated or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Existing CPU allocation and isolated Python required')
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    import itertools
    import json
    import time
    from collections import Counter
    from noah_baseline.m2.c2 import validate_config,build_context
    from noah_baseline.m2.c2_io import load_sources,Queues
    from noah_baseline.m2.c2_mapping import map_record
    from noah_baseline.m2.c2_io import source_rows
    from noah_baseline.m2.stream_index import atomic_json,digest_file
    out=root/'artifacts/reduced_pilot/m2/m2-c2-preflight-20261021-06';out.mkdir(exist_ok=True)
    if (out/'PASS.json').exists():raise FileExistsError('Preflight evidence already finalized')
    config=json.loads((root/'configs/m2_c2.json').read_text());validate_config(config)
    log=(out/'regression.txt').read_text()
    if '132 passed' not in log or 'failed' in log.lower():raise ValueError('Current successful full fixture receipt required')
    start=time.monotonic()
    def check():
        if time.monotonic()-start>180:raise TimeoutError('Bounded real preflight limit')
    manifest,splits,sources,dictionaries=load_sources(root,config)
    assert Counter(splits.values())=={'train':10500,'val':2250,'test':2250}
    context,metadata,dict_evidence=build_context(sources,splits,dictionaries,config,check)
    roots=json.loads((root/'configs/m2.json').read_text())['roots']
    assets=json.loads((root/'configs/encoder_assets.json').read_text())['assets']
    for asset in assets.values():
        check()
        if digest_file(asset['path'],check)!=asset['weight_sha256']:raise ValueError('Pinned encoder weight changed')
    q=Queues(out,{k:dict(preflight=True,kind=k) for k in ['text','specifics','cxr','ecg','echo']});results=[]
    try:
        for source in sources:
            n=1 if source['dataset'] in {'cxr','ecg','echo'} else 32;counter=Counter()
            for row,ref in itertools.islice(source_rows(source,splits,check=check),n):
                es,ds,raw=map_record(row,ref,context,config,q,metadata,roots)
                counter['selected_source_rows']+=1;counter['events']+=len(es);counter['deferred']+=len(ds)
                for event in es:
                    assert event['source_sha256']==source['source_sha256'] and event['split']==splits[event['subject_id']]
                    if event['content_key'] or event['specifics_key']:assert event['model_ready'] is False
                    if event['available_at'] is None:assert event['availability_source']=='unknown'
                if source['dataset']=='note':assert all(e['modality']=='text' for e in es)
            results.append(dict(dataset=source['dataset'],table=source['table'],**counter));check()
        q.commit()
    finally:q.db.close()
    files=['configs/m2_c2.json','configs/m2.json','configs/m2_integration.json','configs/encoder_assets.json',
        'packages/noah_baseline/m2/c2.py','packages/noah_baseline/m2/c2_io.py','packages/noah_baseline/m2/c2_mapping.py','packages/noah_baseline/m2/c2_finalize.py',
        'packages/noah_baseline/m2/adapters.py','packages/noah_baseline/m2/events.py','packages/noah_baseline/m2/stream_index.py',
        'packages/noah_baseline/m2/encoders.py','packages/noah_baseline/m2/otis.py','packages/noah_baseline/m2/echo.py','packages/noah_baseline/m2/media_smoke.py',
        'packages/noah_baseline/data/numeric.py','packages/noah_baseline/data/schema.py','packages/noah_baseline/data/window_builder.py',
        'packages/noah_baseline/m1/data.py','packages/noah_baseline/runtime.py','tools/m2_c2.py','tools/m2_c2_preflight.py','sbatch/m2_c2.sh','tests/test_m2_c2.py','tests/test_m2_csv_reference.py']
    captures={}
    for name in files:
        dest=out/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes());captures[name]=digest_file(dest)
    atomic_json(out/'real-checks.json',dict(status='PASS_BOUNDED_SOURCE_CHECKS',tables=results,dictionaries=dict_evidence,
        elapsed_seconds=time.monotonic()-start,scope='first32 indexed selected rows/table, first1 media row; not population support'))
    atomic_json(out/'PASS.json',dict(status='PASS',config_sha256=digest_file(root/'configs/m2_c2.json'),source_files=captures,
        source_manifest_sha256=config['source_manifest_sha256'],patient_splits_sha256=config['patient_splits_sha256'],
        fixture_log_sha256=digest_file(out/'regression.txt'),real_checks_sha256=digest_file(out/'real-checks.json'),
        job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID')))
    assert_local_modules(root);print('PASS: frozen C2 recipe,132 fixtures,28 source-table checks, exact source/split identities and all four pinned weights')


if __name__=='__main__':main()
