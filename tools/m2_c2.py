"""Isolated entry point for the single approved C2 CPU preparation job."""
from pathlib import Path
import os
import sys


def main():
    if not sys.flags.isolated or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Isolated CPU Slurm execution required')
    root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    from noah_baseline.m2.c2 import run
    run(root,root/'artifacts/reduced_pilot/m2/m2-c2-canonical-20261020-01')
    assert_local_modules(root)


if __name__=='__main__':main()
