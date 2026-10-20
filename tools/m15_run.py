"""Isolated entrypoint for the single researcher-approved, hard-capped M1.5 run."""
from pathlib import Path
import os
import resource
import sys


def main():
    if not sys.flags.isolated:
        raise RuntimeError('Invoke python -I tools/m15_run.py')
    # Set before importing numerical libraries; never initialize CUDA.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='2'
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    root=Path(__file__).resolve().parents[1]
    os.chdir(root);sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from noah_baseline.m15.controller import run,replay_checkpoint
    if len(sys.argv)==4 and sys.argv[1]=='--replay':
        replay_checkpoint(root,Path(sys.argv[2]),int(sys.argv[3]))
    elif len(sys.argv)==2 and sys.argv[1]=='m15-real-20261020-01':
        run(root,sys.argv[1])
    else:
        raise RuntimeError('Only the single approved run identifier or explicit checkpoint replay is supported')
    assert_local_modules(root)


if __name__=='__main__':
    main()
