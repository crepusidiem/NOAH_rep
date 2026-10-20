#!/bin/bash
#SBATCH --job-name=noah_m2_encoder_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/reduced_pilot/m2/logs/%x_%j.out
#SBATCH --error=artifacts/reduced_pilot/m2/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec timeout --signal=TERM --kill-after=10s 550s /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m2_run.py encoder-smoke --output-name "${M2_RUN_ID:-m2-encoder-smoke-${SLURM_JOB_ID}}"
