"""Isolated M2 source/encoder checks; no model training or bulk encoder launcher."""
import os
import argparse
import re
from pathlib import Path
import sys


def main():
    if not sys.flags.isolated:raise RuntimeError('Use python -I tools/m2_run.py')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('M2 scans/encoders require a SLURM allocation')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['inventory','encoder-smoke','source-smoke','media-census','lab-census','cxr-census','media-smoke','integration-smoke','adapter-smoke','readiness','source-index'])
    parser.add_argument('--output-name',required=True,help='Fresh run directory; existing evidence is never overwritten')
    parser.add_argument('--inventory-name',default='m2-a-20261020-01')
    parser.add_argument('--media-name',default='m2-media-real-20261020-02')
    parser.add_argument('--integration-name',default='m2-integration-real-20261020-05')
    args=parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) for name in (args.output_name,args.inventory_name,args.media_name,args.integration_name)):
        raise ValueError('Run names must be single directory names')
    from noah_baseline.runtime import ArtifactNamespace
    namespace=ArtifactNamespace(root,'reduced_pilot')
    out=namespace.owned(root/'artifacts/reduced_pilot/m2'/args.output_name)
    source=namespace.owned(root/'artifacts/reduced_pilot/m2'/args.inventory_name)
    if out.exists():raise FileExistsError('Choose a new output name; prior evidence is immutable')
    if args.phase=='source-index':
        from noah_baseline.m2.source_index import source_index
        source_index(root,out=out,source=source)
    elif args.phase=='readiness':
        from noah_baseline.m2.readiness import check_materialized_run
        integration=namespace.owned(root/'artifacts/reduced_pilot/m2'/args.integration_name)
        if not check_materialized_run(root,out=out,source=source,integration=integration):raise SystemExit(2)
    elif args.phase=='adapter-smoke':
        from noah_baseline.m2.adapter_smoke import adapter_smoke
        adapter_smoke(root,out=out,source=source)
    elif args.phase=='integration-smoke':
        from noah_baseline.m2.integration import integration_smoke
        media=namespace.owned(root/'artifacts/reduced_pilot/m2'/args.media_name)
        integration_smoke(root,out=out,source=source,media_run=media)
    elif args.phase=='media-smoke':
        dependencies=root/'artifacts/reduced_pilot/m2/runtime-deps'
        if not dependencies.is_dir():raise RuntimeError('Pinned isolated M2 media dependencies are required')
        sys.path.insert(0,str(dependencies))
        from noah_baseline.m2.media_smoke import media_smoke
        media_smoke(root,out=out,source=source)
    elif args.phase=='inventory':
        from noah_baseline.m2.inventory import run
        run(root,out)
    elif args.phase=='encoder-smoke':
        from noah_baseline.m2.encoders import smoke
        smoke(root,out=out,source=source)
    elif args.phase=='source-smoke':
        from noah_baseline.m2.source_smoke import source_smoke
        source_smoke(root,out=out,source=source)
    elif args.phase=='media-census':
        from noah_baseline.m2.media_census import media_census
        media_census(root,out=out,source=source)
    elif args.phase=='lab-census':
        from noah_baseline.m2.lab_census import lab_census
        lab_census(root,out=out,source=source)
    elif args.phase=='cxr-census':
        from noah_baseline.m2.cxr_census import cxr_census
        cxr_census(root,out=out,source=source)
    else:raise RuntimeError('Unknown phase; bulk encoding and training are not implemented here')
    assert_local_modules(root)


if __name__=='__main__':main()
