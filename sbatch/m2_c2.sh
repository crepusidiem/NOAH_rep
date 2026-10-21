#!/bin/bash
#SBATCH --job-name=noah_m2_c2
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=artifacts/reduced_pilot/m2/logs/%x_%j.out
#SBATCH --error=artifacts/reduced_pilot/m2/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
exec timeout --signal=TERM --kill-after=10s 7180s /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m2_c2.py
