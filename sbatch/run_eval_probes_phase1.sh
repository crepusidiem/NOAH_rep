#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J eval_probes_150m
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
module load gcc/14.2.0 cuda/12.8

export HF_HOME="$HOME/scratch/cache/huggingface"
export TORCH_HOME="$HOME/scratch/cache/torch"
export UV_CACHE_DIR="$HOME/scratch/cache/uv"
export TMPDIR="/tmp"
export PYTHONPATH="."
export PYTHONUNBUFFERED="1"

NVIDIA_VENV="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_VENV}/cudnn/lib:${NVIDIA_VENV}/cu13/lib:${NVIDIA_VENV}/nccl/lib:${NVIDIA_VENV}/cusparselt/lib:${LD_LIBRARY_PATH}"

VENV_PYTHON="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python"

echo "Running Clinical Probes on GPU for 152M NOAH..."
${VENV_PYTHON} scripts/evaluate_clinical_probes.py \
    --checkpoint checkpoints/phase1_paper_scale/pilot_checkpoint_epoch_10.pt \
    --config configs/paper_scale_config.yaml \
    --data_dir data/processed_with_cxr/test \
    --output_report reports/server/clinical_probes_phase1_paper_scale.json \
    --device cuda

echo "Evaluation finished with code $?"
