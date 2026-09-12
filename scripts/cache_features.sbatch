#!/bin/bash
#SBATCH --job-name=te_feature_cache
#SBATCH --output=logs/te_feature_cache_%j.out
#SBATCH --error=logs/te_feature_cache_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# -----------------------------------------------------------------------------
# cache_features.sbatch
# ----------------------
# Builds and caches the FULL feature matrix (every feature group, incl. the
# expensive MFE/ViennaRNA folding step) once per dataset. Run this before
# run_ablations.sbatch so the array tasks all hit a warm cache instead of
# rebuilding it 26 times in parallel.
#
# Chain it with the array job like this:
#
#   CACHE_JOB=$(sbatch --parsable cache_features.sbatch)
#   sbatch --dependency=afterok:$CACHE_JOB run_ablations.sbatch
# -----------------------------------------------------------------------------

mkdir -p logs

# Activate your environment here (conda/venv — adjust to your cluster setup)
# source ~/miniconda3/bin/activate te_pipeline
# module load python/3.11

python run_pipeline.py --cache-only
