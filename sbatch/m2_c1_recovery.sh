#!/bin/bash
#SBATCH --job-name=noah_m2_c1_recovery
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=artifacts/reduced_pilot/m2/logs/%x_%j.out
#SBATCH --error=artifacts/reduced_pilot/m2/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
exec timeout --signal=TERM --kill-after=10s 3550s /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m2_recover.py
