#!/bin/bash
#SBATCH --job-name=noah_m2_source_index
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/reduced_pilot/m2/logs/%x_%j.out
#SBATCH --error=artifacts/reduced_pilot/m2/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
exec timeout --signal=TERM --kill-after=10s 3550s /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m2_run.py source-index --output-name "${M2_RUN_ID:-m2-source-index-${SLURM_JOB_ID}}"
