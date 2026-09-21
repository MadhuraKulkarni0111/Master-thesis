#!/bin/bash
#SBATCH --job-name=te_pipeline_full
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/te_pipeline_full_%j.out
#SBATCH --error=logs/te_pipeline_full_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"

# -- directories -----------------------------------------------------------
mkdir -p logs

# -- environment -------------------------------------------------------------
source /home/mkulkar/software/miniforge3/etc/profile.d/conda.sh
conda activate master-thesis

# -- run ----------------------------------------------------------------------
python run_pipeline.py

echo "Job finished: $(date)"