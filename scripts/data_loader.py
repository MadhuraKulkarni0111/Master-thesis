"""
data_loader.py
==============
Loads an Excel dataset, calls feature engineering (once, cached), slices
out the requested feature groups for the current ablation, and returns
clean numpy arrays ready for model training.

Caching
-------
The FULL feature matrix (every group) is built once per dataset and
written to FEATURE_CACHE_DIR as a parquet file, alongside a JSON file
mapping each column name to the feature group that produced it. Every
subsequent call — for any ablation — reads the cached matrix and just
selects columns, instead of re-running build_features() (which includes
the expensive MFE/ViennaRNA folding step).

If you change MAX_SAMPLES, or the underlying Excel file, or add/rename a
feature group in sequence_features.py, delete the cache directory (or
the relevant cache files) so it gets rebuilt.

Functions
---------
load_and_prepare(path, te_col, species, feature_groups=None)
    → X_arr, y, folds, feature_names

make_predefined_splits(folds)
    → list of (train_idx, test_idx) tuples
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from config import RESULTS_DIR, MAX_SAMPLES
from sequence_features import build_features

FEATURE_CACHE_DIR = Path(RESULTS_DIR) / "feature_cache"


def _cache_paths(path, species):
    stem = Path(path).stem
    suffix = f"n{MAX_SAMPLES}" if MAX_SAMPLES is not None else "full"
    base = FEATURE_CACHE_DIR / f"{stem}_{species}_{suffix}"
    return base.with_suffix(".parquet"), base.with_suffix(".groups.json")


def load_and_prepare(path, te_col, species, feature_groups=None):
    """
    Load an Excel file and return a feature matrix (sliced to
    `feature_groups`), target vector, fold assignments, and feature names.

    Parameters
    ----------
    path           : str   path to the Excel file
    te_col         : str   name of the translation efficiency column to predict
    species        : str   "human" or "mouse" — selects CAI/TAI weight columns
    feature_groups : list of str or None
        Feature groups to include (see sequence_features.FEATURE_FUNCS).
        None means "all groups" (equivalent to the old full pipeline / A13).

    Returns
    -------
    X_arr         : np.ndarray  shape (n_genes, n_selected_features)
    y             : np.ndarray  shape (n_genes,)
    folds         : np.ndarray  shape (n_genes,)
    feature_names : list of str
    """
    print(f"\nLoading {path}  →  target: {te_col}")
    df = pd.read_excel(path)
    df = df.dropna(subset=[te_col, "tx_sequence"])
    print(f"  Rows after dropping NA in target/sequence: {len(df)}")

    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)
        print(f"  Subsampled to {MAX_SAMPLES} sequences (MAX_SAMPLES is set in config.py)")

    # preserve fold assignments before feature engineering changes the index
    folds = df["fold"].values

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file, groups_file = _cache_paths(path, species)

    if cache_file.exists() and groups_file.exists():
        print(f"  Loading cached full feature matrix: {cache_file}")
        X_full = pd.read_parquet(cache_file)
        with open(groups_file) as fh:
            col_to_group = json.load(fh)
        # guard against a stale cache (different rows than current df)
        if not X_full.index.equals(df.index):
            print("  Cache index does not match current dataframe — rebuilding cache.")
            X_full, col_to_group = build_features(df, species, groups=None, return_groups=True)
            X_full.to_parquet(cache_file)
            with open(groups_file, "w") as fh:
                json.dump(col_to_group, fh)
    else:
        print("  No cache found — building full feature matrix (all groups, incl. MFE)...")
        X_full, col_to_group = build_features(df, species, groups=None, return_groups=True)
        X_full.to_parquet(cache_file)
        with open(groups_file, "w") as fh:
            json.dump(col_to_group, fh)
        print(f"  Cached: {cache_file}")

    if feature_groups is not None:
        keep_cols = [c for c in X_full.columns if col_to_group.get(c) in feature_groups]
        X = X_full[keep_cols]
    else:
        X = X_full

    y             = df[te_col].values
    feature_names = X.columns.tolist()

    # constant-fill imputation — fills NaN with 0 (unchanged from original)
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_arr   = imputer.fit_transform(X)

    print(f"  Feature matrix shape: {X_arr.shape}  (groups: {feature_groups or 'ALL'})")
    print(f"  Fold distribution: "
          f"{dict(zip(*np.unique(folds, return_counts=True)))}")

    return X_arr, y, folds, feature_names


def make_predefined_splits(folds):
    """
    Convert the dataset's fold column into sklearn-compatible CV splits.

    Parameters
    ----------
    folds : np.ndarray  shape (n_genes,)
        Integer fold assignments, e.g. 0-9 for 10-fold CV.

    Returns
    -------
    list of (train_idx, test_idx) tuples
    """
    splits = []
    for held_out in sorted(np.unique(folds)):
        train_idx = np.where(folds != held_out)[0]
        test_idx  = np.where(folds == held_out)[0]
        splits.append((train_idx, test_idx))
    return splits
