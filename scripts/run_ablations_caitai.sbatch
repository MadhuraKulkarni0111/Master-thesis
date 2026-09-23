#!/bin/bash
#SBATCH --job-name=te_ablation_caitai
#SBATCH --output=logs/te_ablation_caitai_%A_%a.out
#SBATCH --error=logs/te_ablation_caitai_%A_%a.err
#SBATCH --array=0-35
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -e

# -----------------------------------------------------------------------------
# run_ablations_caitai.sbatch
# -----------------------------
# Runs the CAI/tAI sub-study: for each base ablation A1-A6, the three added
# variants (+cai, +tai, +cai+tai) x 2 datasets (human, mouse) = 18 x 2 = 36
# tasks -> --array=0-35
#
# Separate from run_ablations.sbatch (the main A1-A13 array) so it doesn't
# disturb array indices you're already tracking for that job.
#
# Requires the feature cache to already exist (built once by
# cache_features.sbatch) — this sub-study reuses the same cached full
# feature matrix, it does not rebuild anything.
# -----------------------------------------------------------------------------

mkdir -p logs

source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

ABLATIONS=(A1_cai A1_tai A1_cai_tai A2_cai A2_tai A2_cai_tai A3_cai A3_tai A3_cai_tai A4_cai A4_tai A4_cai_tai A5_cai A5_tai A5_cai_tai A6_cai A6_tai A6_cai_tai)
DATASETS=(human mouse)

N_ABLATIONS=${#ABLATIONS[@]}
A_IDX=$(( SLURM_ARRAY_TASK_ID % N_ABLATIONS ))
D_IDX=$(( SLURM_ARRAY_TASK_ID / N_ABLATIONS ))

ABLATION=${ABLATIONS[$A_IDX]}
DATASET=${DATASETS[$D_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID  ->  ablation=$ABLATION  dataset=$DATASET"

python run_pipeline.py --ablation "$ABLATION" --dataset "$DATASET"
