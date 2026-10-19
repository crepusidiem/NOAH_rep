#!/bin/bash
# ==============================================================================
# SLURM GPU Batch Job: NOAH Phase 1 Architecture Scaling (150M Parameters)
# Cluster: Harvard Medical School O2 (Rajpurkar Lab)
# Steps executed sequentially:
#   1. Train NOAH at full paper dimensions (d_model=768, d_latent=384, 12 layers, 152M params)
#   2. Run Comprehensive Verification & Reconstruction Audit on held-out test split
#   3. Run Downstream Clinical Probing Suite (Mortality, Readmission, LOS>7d, NEWS2)
# ==============================================================================

#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J noah_150m
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

mkdir -p logs checkpoints/phase1_paper_scale reports/server

echo "========================================================================"
echo "NOAH PHASE 1 ARCHITECTURE SCALING PIPELINE (150M PARAMETERS)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Running on:    $(hostname)"
echo "Started at:    $(date)"
echo "CUDA Device:   ${CUDA_VISIBLE_DEVICES:-Not Set}"
echo "========================================================================"

# 1. Load System Modules
module purge
module load gcc/14.2.0 cuda/12.8

# 2. Cache & Environment Setup
export HF_HOME="$HOME/scratch/cache/huggingface"
export TORCH_HOME="$HOME/scratch/cache/torch"
export UV_CACHE_DIR="$HOME/scratch/cache/uv"
export TMPDIR="/tmp"
export PYTHONPATH="."
export PYTHONUNBUFFERED="1"

# Ensure PyTorch finds bundled nvidia CUDA & cuDNN 9 sublibraries
NVIDIA_VENV="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_VENV}/cudnn/lib:${NVIDIA_VENV}/cu13/lib:${NVIDIA_VENV}/nccl/lib:${NVIDIA_VENV}/cusparselt/lib:${LD_LIBRARY_PATH}"

# Check GPU status
nvidia-smi

VENV_PYTHON="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python"

# ------------------------------------------------------------------------------
# Stage 1: Train 152M Parameter NOAH on GPU
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 1/3] Training 150M NOAH on GPU (10 Epochs)..."
${VENV_PYTHON} scripts/train_pilot.py \
    --config configs/paper_scale_config.yaml \
    --train_data data/processed_with_cxr/train \
    --val_data data/processed_with_cxr/val \
    --output_dir checkpoints/phase1_paper_scale \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --lr 3e-4 \
    --warmup_epochs 2 \
    --kl_warmup_epochs 5 \
    --device cuda \
    --use_wandb \
    --wandb_project noah-phase1 \
    --wandb_name noah_150m_paper_scale_10ep \
    --wandb_mode ${WANDB_MODE:-online}

if [ $? -ne 0 ]; then
    echo "Error: Stage 1 (Model Training) failed!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 2: Comprehensive Verification & Reconstruction Audit on Held-Out Test Set
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 2/3] Running Verification & Reconstruction Audit..."
${VENV_PYTHON} scripts/verify_pilot_results.py \
    --checkpoint checkpoints/phase1_paper_scale/pilot_checkpoint_epoch_10.pt \
    --config configs/paper_scale_config.yaml \
    --data_dir data/processed_with_cxr/test \
    --output_report reports/server/audit_phase1_paper_scale.json \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Stage 2 (Verification Audit) failed!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 3: Downstream Clinical Probing Suite
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 3/3] Evaluating Downstream Clinical Probes (Mortality, Readmission, LOS, NEWS2)..."
${VENV_PYTHON} scripts/evaluate_clinical_probes.py \
    --checkpoint checkpoints/phase1_paper_scale/pilot_checkpoint_epoch_10.pt \
    --config configs/paper_scale_config.yaml \
    --data_dir data/processed_with_cxr/test \
    --output_report reports/server/clinical_probes_phase1_paper_scale.json \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Stage 3 (Clinical Probes) failed!"
    exit 1
fi

echo "========================================================================"
echo "PHASE 1 PIPELINE FULLY COMPLETED AT $(date)"
echo "========================================================================"
