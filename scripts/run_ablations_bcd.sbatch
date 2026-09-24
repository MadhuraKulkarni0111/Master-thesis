#!/bin/bash
#SBATCH --job-name=te_ablation_bcd
#SBATCH --output=logs/te_ablation_bcd_%A_%a.out
#SBATCH --error=logs/te_ablation_bcd_%A_%a.err
#SBATCH --array=0-23
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -e

# -----------------------------------------------------------------------------
# run_ablations_bcd.sbatch
# --------------------------
# Runs Family B (Location), Family C (Mechanism), Family D (Representation):
#   B1-B5 (5) + C1-C3 (3) + D1-D4 (4) = 12 ablations x 2 datasets = 24 tasks
#   -> --array=0-23
#
# Requires the feature cache to already exist (built once by
# cache_features.sbatch) — reuses the same cached full feature matrix,
# does not rebuild anything. Column selection (including the new
# region-based filtering for Family B) is resolved at load time from
# column names, so no cache changes were needed for this to work.
# -----------------------------------------------------------------------------

mkdir -p logs

source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

ABLATIONS=(B1 B2 B3 B4 B5 C1 C2 C3 D1 D2 D3 D4)
DATASETS=(human mouse)

N_ABLATIONS=${#ABLATIONS[@]}
A_IDX=$(( SLURM_ARRAY_TASK_ID % N_ABLATIONS ))
D_IDX=$(( SLURM_ARRAY_TASK_ID / N_ABLATIONS ))

ABLATION=${ABLATIONS[$A_IDX]}
DATASET=${DATASETS[$D_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID  ->  ablation=$ABLATION  dataset=$DATASET"

python run_pipeline.py --ablation "$ABLATION" --dataset "$DATASET"
