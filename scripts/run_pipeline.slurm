#!/bin/bash

#SBATCH --job-name=te_pipeline
#SBATCH --array=0-25
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs_eval/te_pipeline_%A_%a.out
#SBATCH --error=logs_eval/te_pipeline_%A_%a.err

echo "Job started: $(date)"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# ── environment ──────────────────────────────────────────────────────────
source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

# ── run ──────────────────────────────────────────────────────────────────
python run_pipeline.py

echo "Job finished: $(date)"