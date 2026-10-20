#!/bin/bash
# ==============================================================================
# SLURM GPU Array Job: Controlled Class Weighting Strength (Alpha-Sweep) Ablation
# Evaluates w_c = (N_c + 1)^(-alpha) for alpha in {0.00, 0.25, 0.50}
# across 3 paired random seeds {42, 43, 44} (9 total runs).
#
# Array Tasks: 0 to 8
# Concurrency limit: %3 (at most 3 jobs simultaneously)
# ==============================================================================

#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J noah_alpha_grid
#SBATCH --array=0-8%3
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Controlled remediation hold: methodology decisions and experiment gates are pending.
echo "BLOCKED: alpha-sweep is not approved; see docs/ARCHITECTURE_AUDIT.md remediation matrix." >&2
exit 78

mkdir -p logs checkpoints/ablation_alpha_sweep reports/server

# Define the 9 experimental configurations: (alpha, seed, run_name)
CONFIGS=(
  "0.00 42 alpha00_seed42"
  "0.00 43 alpha00_seed43"
  "0.00 44 alpha00_seed44"
  "0.25 42 alpha25_seed42"
  "0.25 43 alpha25_seed43"
  "0.25 44 alpha25_seed44"
  "0.50 42 alpha50_seed42"
  "0.50 43 alpha50_seed43"
  "0.50 44 alpha50_seed44"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CFG=(${CONFIGS[$TASK_ID]})
ALPHA=${CFG[0]}
SEED=${CFG[1]}
RUN_NAME=${CFG[2]}

echo "========================================================================"
echo "NOAH CONTROLLED ALPHA-SWEEP ABLATION: RUN ${TASK_ID}/8 (${RUN_NAME})"
echo "Job Array ID:  ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Weight Alpha:  ${ALPHA}"
echo "Random Seed:   ${SEED}"
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

nvidia-smi

VENV_PYTHON="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python"

# 3. Determine weighting flags
WEIGHT_FLAGS=""
if [ "$ALPHA" != "0.00" ]; then
    WEIGHT_FLAGS="--use_class_weights --class_weight_alpha ${ALPHA}"
fi

OUTPUT_DIR="checkpoints/ablation_alpha_sweep/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------------------
# Stage 1: Train NOAH Model (10 Epochs)
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 1/3] Training NOAH (${RUN_NAME}: alpha=${ALPHA}, seed=${SEED})..."
${VENV_PYTHON} scripts/train_pilot.py \
    --config configs/paper_scale_config.yaml \
    --train_data data/processed_dynamic_demo/train \
    --val_data data/processed_dynamic_demo/val \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --lr 3e-4 \
    --warmup_epochs 2 \
    --kl_warmup_epochs 5 \
    --seed ${SEED} \
    ${WEIGHT_FLAGS} \
    --device cuda \
    --use_wandb \
    --wandb_project noah-alpha-ablation \
    --wandb_name "${RUN_NAME}" \
    --wandb_mode ${WANDB_MODE:-offline}

if [ $? -ne 0 ]; then
    echo "Error: Stage 1 (Training) failed for ${RUN_NAME}!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 2: Verification Audit on Held-Out Test Set
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 2/3] Running Verification & Reconstruction Audit..."
${VENV_PYTHON} scripts/verify_pilot_results.py \
    --checkpoint "${OUTPUT_DIR}/pilot_checkpoint_epoch_10.pt" \
    --data_dir data/processed_dynamic_demo/test \
    --config configs/paper_scale_config.yaml \
    --output_report "reports/server/audit_${RUN_NAME}.json" \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Stage 2 (Verification Audit) failed for ${RUN_NAME}!"
    exit 1
fi

# ------------------------------------------------------------------------------
# Stage 3: Downstream Clinical Probing Suite
# ------------------------------------------------------------------------------
echo -e "\n>>> [Stage 3/3] Evaluating Downstream Clinical Probes..."
${VENV_PYTHON} scripts/evaluate_clinical_probes.py \
    --checkpoint "${OUTPUT_DIR}/pilot_checkpoint_epoch_10.pt" \
    --data_dir data/processed_dynamic_demo/test \
    --config configs/paper_scale_config.yaml \
    --output_report "reports/server/clinical_probes_${RUN_NAME}.json" \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Stage 3 (Clinical Probing) failed for ${RUN_NAME}!"
    exit 1
fi

echo -e "\n========================================================================"
echo "SUCCESSFULLY COMPLETED PIPELINE FOR ${RUN_NAME}"
echo "Completed at: $(date)"
echo "========================================================================"
