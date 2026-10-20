"""Explicitly authorized C1 recovery only; no automatic full-source execution."""
from pathlib import Path
import os
import sys


def main():
    if not sys.flags.isolated:raise RuntimeError('Use python -I tools/m2_recover.py')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('C1 recovery requires a CPU Slurm allocation')
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    from noah_baseline.m2.c1_recovery import recover
    recover(root,root/'artifacts/reduced_pilot/m2/m2-c1-recovery-20261020-01')
    assert_local_modules(root)


if __name__=='__main__':main()
