#!/bin/bash
#SBATCH --job-name=te_nt
#SBATCH --output=logs/te_nt_%A_%a.out
#SBATCH --error=logs/te_nt_%A_%a.err
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -e

# -----------------------------------------------------------------------------
# run_nt.sbatch
# -------------
# Step 2 of the NT-v2 runs: models on CPU, using the cached embeddings.
#   task 0 -> human, task 1 -> mouse
# Each task runs both feature sets (nt only, and hand-crafted + nt).
# Needs the hand-crafted feature cache (cache_features.sbatch) for "combined".
#
# Chain everything:
#   EMBED=$(sbatch --parsable nt_embed.sbatch)
#   sbatch --dependency=afterok:$EMBED run_nt.sbatch
# -----------------------------------------------------------------------------

mkdir -p logs

source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

DATASETS=(human mouse)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID  ->  dataset=$DATASET"

python run_nt.py --dataset "$DATASET"
