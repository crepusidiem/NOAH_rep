#!/bin/bash
# ==============================================================================
# SLURM GPU Batch Job: Pilot Feature Extraction & Prototype Training
# Cluster: Harvard Medical School O2 (Rajpurkar Lab)
# Usage:
#   sbatch sbatch/run_pilot_gpu.sh
#   sbatch --gres=gpu:a100:1 -p gpu sbatch/run_pilot_gpu.sh
# ==============================================================================

#SBATCH -p gpu                         # Partition: gpu
#SBATCH --gres=gpu:1                   # Request 1 GPU (optionally: gpu:a100:1 or gpu:rtx8000:1)
#SBATCH -c 4                           # 4 CPU cores
#SBATCH --mem=32G                      # 32 GB RAM
#SBATCH -t 08:00:00                    # 8 hours wall time
#SBATCH -J noah_gpu                    # Job name
#SBATCH -o logs/%x_%j.out              # Standard output
#SBATCH -e logs/%x_%j.err              # Standard error

# Create logs directory
mkdir -p logs

echo "========================================================================"
echo "NOAH GPU Pilot Run"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Running on:    $(hostname)"
echo "Started at:    $(date)"
echo "CUDA Device:   ${CUDA_VISIBLE_DEVICES:-Not Set}"
echo "========================================================================"

# 1. Load System Modules (GCC 14.2 + CUDA 12.8)
module purge
module load gcc/14.2.0 cuda/12.8

# 2. Cache Redirection to Scratch
export HF_HOME="$HOME/scratch/cache/huggingface"
export TORCH_HOME="$HOME/scratch/cache/torch"
export UV_CACHE_DIR="$HOME/scratch/cache/uv"
export TMPDIR="$HOME/scratch/tmp"
export PYTHONPATH="."

# Check GPU status
nvidia-smi

# 3. Execute with persistent environment
VENV_PYTHON="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python"

# Placeholder for prototype training entrypoint
echo "Ready for GPU execution with: ${VENV_PYTHON}"

echo "Finished at: $(date)"
