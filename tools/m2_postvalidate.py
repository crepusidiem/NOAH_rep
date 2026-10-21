"""Bounded independent post-validation of an existing CXR metadata index only."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    if not sys.flags.isolated or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use isolated Python in an existing CPU Slurm allocation')
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',required=True);args=parser.parse_args()
    if Path(args.output_name).name!=args.output_name or args.output_name in {'.','..'}:raise ValueError('A fresh run name is required')
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    import csv
    import resource
    from noah_baseline.m2.csv_reference import attest_existing_index
    from noah_baseline.m2.stream_index import atomic_json,digest_file
    base=root/'artifacts/reduced_pilot/m2';run=base/'m2-c1-recovery-20261020-01'
    attempt=next(run.glob('attempt-*'))
    # The original job's terminal receipt must remain untouched.
    original=json.loads((attempt/'execution.json').read_text())
    index=run/'cxr-mimic-cxr-2.0.0-metadata';completion=json.loads((index/'COMPLETE.json').read_text())
    source=Path(completion['source_identity']['source_path'])
    if source.stat().st_size>64*2**20:raise ValueError('This inexpensive check is limited to the small metadata source')
    with (base/'m2-a-20261020-01/patient_splits.csv').open(newline='') as f:
        splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    out=base/args.output_name;out.mkdir(exist_ok=False);start=time.monotonic()
    capture={}
    for name in ['tools/m2_postvalidate.py','packages/noah_baseline/m2/csv_reference.py']:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes());capture[name]=digest_file(dest)
    atomic_json(out/'manifest.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),source_files=capture,
        job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),
        original_job_id='57044273',original_execution_status=original['status'],
        original_execution_sha256=digest_file(attempt/'execution.json'),index_rebuilt=False))
    proof=attest_existing_index(source,index,splits,out/'cxr-metadata-attestation.json')
    atomic_json(out/'resources.json',dict(elapsed_seconds=time.monotonic()-start,
        peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,duckdb_memory='256MB',
        selected_rows=proof['selected_rows'],new_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))
    assert_local_modules(root)
    print(json.dumps(dict(status=proof['status'],selected_rows=proof['selected_rows'],original_job_status_unchanged=True)))


if __name__=='__main__':main()
