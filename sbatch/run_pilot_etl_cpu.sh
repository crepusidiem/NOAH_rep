#!/bin/bash
# ==============================================================================
# SLURM CPU Batch Job: Pilot Subcohort ETL for NOAH Replication
# Cluster: Harvard Medical School O2 (Rajpurkar Lab)
# Usage:
#   sbatch sbatch/run_pilot_etl_cpu.sh
# ==============================================================================

#SBATCH -p short                       # Partition: short (up to 12 hrs)
#SBATCH -c 8                           # 8 CPU cores for fast table scanning
#SBATCH --mem=64G                      # 64 GB RAM for large table chunks
#SBATCH -t 04:00:00                    # 4 hours max wall time
#SBATCH -J noah_pilot_etl              # Job name
#SBATCH -o logs/%x_%j.out              # Standard output
#SBATCH -e logs/%x_%j.err              # Standard error

# Create logs directory
mkdir -p logs data

echo "========================================================================"
echo "NOAH Pilot Subcohort ETL"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Running on:     $(hostname)"
echo "Started at:     $(date)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "========================================================================"

# Cache redirection
export UV_CACHE_DIR="$HOME/scratch/cache/uv"
export TMPDIR="$HOME/scratch/tmp"
export PYTHONPATH="."
export PYTHONUNBUFFERED="1"

# Run ETL with persistent virtual environment
VENV_PYTHON="/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python"

echo "Executing Complete NOAH Data Curation Pipeline (N-05)..."
${VENV_PYTHON} scripts/run_curation_pipeline.py \
    --manifest data/pilot_cohort_manifest.csv \
    --output_dir data/processed \
    --max_seq_len 1024 \
    --overlap_len 256 \
    --min_events 64

echo "ETL Finished at: $(date)"
