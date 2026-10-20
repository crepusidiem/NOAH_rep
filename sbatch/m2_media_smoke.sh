#!/bin/bash
#SBATCH --job-name=noah_m2_media_smoke
#SBATCH --partition=gpu,gpu_quad,gpu_requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=gpuquad_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=artifacts/reduced_pilot/m2/logs/%x_%j.out
#SBATCH --error=artifacts/reduced_pilot/m2/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec timeout --signal=TERM --kill-after=10s 280s /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/m2_run.py media-smoke --output-name "${M2_RUN_ID:-m2-media-smoke-${SLURM_JOB_ID}}"
