#!/bin/bash
#SBATCH --job-name=te_pipeline
#SBATCH --array=0-25
#SBATCH --cpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/te_pipeline_%j.out
#SBATCH --error=logs/te_pipeline_%j.err

set -euo pipefail

# ── environment ─────────────────────────────────────────────────────────
mkdir -p logs_eval

# Adjust to how conda/mamba is set up on your cluster — some clusters need
# `module load miniconda3` (or similar) before this will work.
source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

# ── sanity checks ───────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU visible to job:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (nvidia-smi not found — check --gres=gpu request)"
echo "Python: $(which python)"
python -c "import sklearn, lightgbm, xgboost, RNA; print('deps OK')"

# ── run ──────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"     # run from wherever you submitted the job, i.e. your scripts/ dir
python run_pipeline.py

echo "Job finished: $(date)"
