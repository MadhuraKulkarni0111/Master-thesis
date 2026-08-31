"""
config.py
=========
Central configuration for the TE prediction pipeline.

This is the file we will need to edit when:
  - moving data files to a different location
  - adding a new dataset / cell line
  - changing output directories
"""

from pathlib import Path
import os

# ── Base paths ─────────────────────────────────────────────────────────────
# All paths below are derived from this file's location, so the pipeline
# runs correctly regardless of which machine or directory it's cloned into.
#
# __file__ is .../feature_engineering/scripts/config.py
# .parent        → .../feature_engineering/scripts/
# .parent.parent → .../feature_engineering/          ← where data/ and results/ live
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# ── Dataset definitions ───────────────────────────────────────────────────────
# Each entry is a tuple of:
#   (excel_path, te_column, label, output_prefix, species)
#
#   excel_path   : full path to the input Excel file
#   te_column    : name of the column containing translation efficiency values
#   label        : human-readable name used in plot titles and CSV headers
#   output_prefix: path prefix for all output files produced for this dataset
#                  (PNGs and CSVs will be saved as <output_prefix>_*.png / *.csv)
#   species      : "human" or "mouse" — selects CAI/TAI codon weight columns

DATASETS = [
    (
        str(DATA_DIR / "Human_data.xlsx"),
        "TE_HCT116",
        "Human_HCT116",
        str(RESULTS_DIR) + os.sep,
        "human"
    ),
    (
        str(DATA_DIR / "Mouse_data.xlsx"),
        "TE_4T1",
        "Mouse_4T1",
        str(RESULTS_DIR) + os.sep,
        "mouse"
    ),
]

# ── Sampling settings ─────────────────────────────────────────────────────────
# Set to an integer to run on a subset of sequences (useful for quick testing).
# Set to None to run on the full dataset.
MAX_SAMPLES = 100

# ── Feature engineering settings ─────────────────────────────────────────────

# Codons that terminate translation — excluded from codon usage features
STOP_CODONS = {"TAA", "TAG", "TGA"}
# STOP_CODONS = { }

# ── Visualisation settings ────────────────────────────────────────────────────

# Number of top features to show in the bar chart plots
TOP_N_BAR = 30

# Number of top features (per model) to include in the comparison heatmap
TOP_N_HEATMAP = 25

# Number of top features to save per model in the CSV output
TOP_N_CSV = 50

# Colour assigned to each model in all plots
MODEL_COLORS = {
    "Lasso":        "#4E79A7",   # blue
    "ElasticNet":   "#F28E2B",   # orange
    "RandomForest": "#59A14F",   # green
    "LightGBM":     "#B07AA1",   # purple
    "XGBoost":      "#E15759", # red
    "LinearSVM" :   "#1B9E77", # teal
    "SVR":          "#76B7B2",   # light teal s
}

# ── Model hyperparameters ─────────────────────────────────────────────────────
# Kept here so you can tune them without touching the model definition code.

# Regression parameters 
LASSO_ALPHA        = 0.001
ELASTICNET_ALPHA   = 0.001
ELASTICNET_L1RATIO = 0.5

# Random forest parameters 
RF_N_ESTIMATORS    = 300
RF_MAX_FEATURES    = "sqrt"
RF_RANDOM_STATE    = 42

# LGBM parameters 
LGBM_N_ESTIMATORS    = 1000
LGBM_LEARNING_RATE   = 0.05
LGBM_NUM_LEAVES      = 63
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_SUBSAMPLE       = 0.8
LGBM_COLSAMPLE       = 0.8
LGBM_RANDOM_STATE    = 42

# XGBoost Parameters
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_RANDOM_STATE = 42

# SVM parameters
SVM_C       = 1.0
SVM_EPSILON = 0.1
SVM_MAX_ITER = 10000
SVM_RANDOM_STATE = 42
SVM_KERNEL  = "rbf"
SVM_GAMMA   = "scale"
SVM_MAX_ITER = -1

# -----------------------------------------------------------------------------
# Codon Adaptation Index (CAI) / tRNA Adaptation Index (TAI) configuration
# -----------------------------------------------------------------------------
#
# Species-specific codon weights used for CAI/TAI calculation.
#
# CSV columns:
#   codon
#   amino_acid
#   cai_weight_human
#   cai_weight_mouse
# -----------------------------------------------------------------------------

CAI_WEIGHTS_FILE = DATA_DIR / "combined_cai_weights.csv"
TAI_WEIGHTS_FILE = DATA_DIR / "combined_tai_weights.csv"

# ------ config settings for the window size for MFE calculation ------------------

START_WINDOW_UPSTREAM = 30
START_WINDOW_DOWNSTREAM = 30