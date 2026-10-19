#!/bin/bash
# ==============================================================================
# SLURM GPU Batch Job: Complete NOAH CXR Ablation Pipeline
# Cluster: Harvard Medical School O2 (Rajpurkar Lab)
# Steps executed sequentially:
#   1. Pre-cache all 6,763 CXRs on GPU using RAD-DINO
#   2. Run Data Curation ETL with attached CXR representations (data/processed_with_cxr)
#   3. Train NOAH for 10 epochs on GPU with full CXR representations
#   4. Run Comprehensive Verification & Reconstruction Audit on held-out test split
#   5. Run Downstream Clinical Probing Suite (Mortality, Readmission, LOS>7d, NEWS2)
# ==============================================================================

#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J noah_cxr_exp
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

mkdir -p logs data/processed_with_cxr checkpoints/pilot_with_cxr reports/server

echo "========================================================================"
echo "NOAH COMPLETE MULTIMODAL CXR ABLATION PIPELINE"
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
# Stage 1: Pre-cache CXR Embeddings with RAD-DINO on GPU
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 1/5] Pre-caching CXR Embeddings with RAD-DINO on GPU..."
${VENV_PYTHON} scripts/precache_cxrs.py \
    --manifest data/pilot_cohort_manifest.csv \
    --cxr_base_dir /home/wey371/rajpurkarlab/datasets/cxr/MIMIC-CXR/raw_jpg \
    --cache_dir data/cache/cxr \
    --batch_size 64 \
    --num_workers 4 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Stage 1 (CXR Pre-caching) failed!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 2: Multimodal Data Curation ETL (N-05) with Attached CXRs
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 2/5] Running Multimodal Data Curation ETL (N-05)..."
${VENV_PYTHON} scripts/run_curation_pipeline.py \
    --manifest data/pilot_cohort_manifest.csv \
    --output_dir data/processed_with_cxr \
    --max_seq_len 1024 \
    --overlap_len 256 \
    --min_events 64 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Stage 2 (Data Curation ETL) failed!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 3: Train NOAH on GPU with Full CXR Representations (N-07)
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 3/5] Training NOAH on GPU with Full CXR Representations (N-07)..."
${VENV_PYTHON} scripts/train_pilot.py \
    --train_data data/processed_with_cxr/train \
    --val_data data/processed_with_cxr/val \
    --output_dir checkpoints/pilot_with_cxr \
    --batch_size 8 \
    --epochs 10 \
    --lr 6e-4 \
    --device cuda \
    --use_wandb \
    --wandb_project noah-pilot \
    --wandb_name noah_pilot_with_cxr_10ep \
    --wandb_mode ${WANDB_MODE:-online}

if [ $? -ne 0 ]; then
    echo "Error: Stage 3 (Model Training) failed!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 4: Comprehensive Verification & Reconstruction Audit on Held-Out Test Set
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 4/5] Auditing Reconstruction & Hurdle Metrics on Held-Out Test Shards..."
${VENV_PYTHON} scripts/verify_pilot_results.py \
    --checkpoint checkpoints/pilot_with_cxr/pilot_checkpoint_epoch_10.pt \
    --data_dir data/processed_with_cxr/test \
    --device cuda \
    --output_report reports/server/audit_with_cxr.json

if [ $? -ne 0 ]; then
    echo "Error: Stage 4 (Verification Audit) failed!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 5: Downstream Clinical Probing Evaluation
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 5/5] Evaluating Downstream Clinical Probes (Mortality, Readmission, LOS, NEWS2)..."
${VENV_PYTHON} scripts/evaluate_clinical_probes.py \
    --checkpoint checkpoints/pilot_with_cxr/pilot_checkpoint_epoch_10.pt \
    --config configs/pilot_config.yaml \
    --data_dir data/processed_with_cxr/test \
    --device cuda \
    --output_report reports/server/clinical_probes_with_cxr.json

if [ $? -ne 0 ]; then
    echo "Error: Stage 5 (Clinical Probes) failed!"
    exit 1
fi

echo -e "\n========================================================================"
echo "PIPELINE FULLY COMPLETED AT $(date)"
echo "========================================================================"
