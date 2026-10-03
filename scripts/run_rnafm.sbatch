#!/bin/bash
#SBATCH --job-name=te_rnafm
#SBATCH --output=logs/te_rnafm_%A_%a.out
#SBATCH --error=logs/te_rnafm_%A_%a.err
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1        # only needed if embeddings are NOT already cached;
                            # remove this line if you copied the cache from your laptop

set -e

# -----------------------------------------------------------------------------
# run_rnafm.sbatch
# -----------------
# Normal (non-ablation) pipeline with RNA-FM features:
#   task 0 -> human, task 1 -> mouse
# each runs both feature sets (rnafm only, and hand-crafted + rnafm).
#
# Requires the hand-crafted feature cache (cache_features.sbatch) for the
# "combined" feature set. Chain them like this:
#
#   CACHE_JOB=$(sbatch --parsable cache_features.sbatch)
#   sbatch --dependency=afterok:$CACHE_JOB run_rnafm.sbatch
# -----------------------------------------------------------------------------

mkdir -p logs

source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

DATASETS=(human mouse)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID  ->  dataset=$DATASET"

python run_rnafm.py --dataset "$DATASET"
