#!/bin/bash
#SBATCH --job-name=te_ablation
#SBATCH --output=logs/te_ablation_%A_%a.out
#SBATCH --error=logs/te_ablation_%A_%a.err
#SBATCH --array=0-25
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# -----------------------------------------------------------------------------
# run_ablations.sbatch
# ---------------------
# Runs every (ablation x dataset) combination as one array task each.
# 13 ablations (A1-A13) x 2 datasets (human, mouse) = 26 tasks -> --array=0-25
#
# IMPORTANT: run the feature cache build ONCE, serially, before submitting
# this array — otherwise every task races to build the same cache file.
#
#   python run_pipeline.py --cache-only
#   sbatch run_ablations.sbatch
#
# or chain them so the array only starts after caching finishes:
#
#   CACHE_JOB=$(sbatch --parsable cache_features.sbatch)
#   sbatch --dependency=afterok:$CACHE_JOB run_ablations.sbatch
# -----------------------------------------------------------------------------

mkdir -p logs

# Activate your environment here (conda/venv — adjust to your cluster setup)
# source ~/miniconda3/bin/activate te_pipeline
# module load python/3.11

ABLATIONS=(A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13)
DATASETS=(human mouse)

N_ABLATIONS=${#ABLATIONS[@]}
A_IDX=$(( SLURM_ARRAY_TASK_ID % N_ABLATIONS ))
D_IDX=$(( SLURM_ARRAY_TASK_ID / N_ABLATIONS ))

ABLATION=${ABLATIONS[$A_IDX]}
DATASET=${DATASETS[$D_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID  ->  ablation=$ABLATION  dataset=$DATASET"

python run_pipeline.py --ablation "$ABLATION" --dataset "$DATASET"
