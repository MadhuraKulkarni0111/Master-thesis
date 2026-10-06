#!/bin/bash
#SBATCH --job-name=te_nt_embed
#SBATCH --output=logs/te_nt_embed_%A_%a.out
#SBATCH --error=logs/te_nt_embed_%A_%a.err
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
##SBATCH --partition=<gpu-partition>   # uncomment and set if your cluster needs it

set -e

# -----------------------------------------------------------------------------
# nt_embed.sbatch
# ---------------
# Step 1 of the NT-v2 runs: compute and cache embeddings on a GPU.
#   task 0 -> human, task 1 -> mouse
# Prerequisite (once, on the login node): python download_nt_model.py
# -----------------------------------------------------------------------------

mkdir -p logs

source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

DATASETS=(human mouse)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID  ->  dataset=$DATASET"
nvidia-smi || true

python run_nt.py --embed-only --dataset "$DATASET"
