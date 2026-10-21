"""Write an unapproved GPU resource proposal from the actual completed C2 queues."""
from pathlib import Path
import argparse
import sys
import subprocess


def main():
    if not sys.flags.isolated:raise RuntimeError('Use isolated Python')
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-name',default='m2-c2-canonical-20261020-01')
    parser.add_argument('--output-name',required=True);args=parser.parse_args()
    for name in [args.dataset_name,args.output_name]:
        if Path(name).name!=name or name in {'.','..'}:raise ValueError('Explicit clean namespace names required')
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    from noah_baseline.m2.gpu_plan import proposal
    from noah_baseline.m2.stream_index import atomic_json,digest_file
    base=root/'artifacts/reduced_pilot/m2';result=proposal(base/args.dataset_name/'dataset-manifest.json')
    out=base/args.output_name;out.mkdir(exist_ok=False)
    result['planner_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    result['planner_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root))
    result['planner_sources']={}
    for name in ['tools/m2_gpu_plan.py','packages/noah_baseline/m2/gpu_plan.py']:
        dest=out/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes());result['planner_sources'][name]=digest_file(dest)
    atomic_json(out/'proposal.json',result)
    print('PROPOSED_NOT_AUTHORIZED: actual queue counts saved; no Slurm submission or encoder inference')
    assert_local_modules(root)


if __name__=='__main__':main()
