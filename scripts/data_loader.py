"""
data_loader.py
==============
Loads an Excel dataset, calls feature engineering (once, cached), slices
out the requested feature groups/regions for the current ablation, and
returns clean numpy arrays ready for model training.

Caching
-------
The FULL feature matrix (every group) is built once per dataset and
written to FEATURE_CACHE_DIR as a parquet file. Every subsequent call —
for any ablation — reads the cached matrix and just selects columns,
instead of re-running build_features() (which includes the expensive
MFE/ViennaRNA folding step).

Column selection is resolved fresh on every call via
sequence_features.classify_column(), which tags each column with its
(group, region) purely from its name — this is cheap (string parsing
over ~300 column names) so there's no separate tags cache to go stale.

If you change MAX_SAMPLES, or the underlying Excel file, or add/rename a
feature group in sequence_features.py, delete the cache directory (or
the relevant cache file) so it gets rebuilt.

Functions
---------
load_and_prepare(path, te_col, species, ablation_spec=None)
    → X_arr, y, folds, feature_names

make_predefined_splits(folds)
    → list of (train_idx, test_idx) tuples
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from config import RESULTS_DIR, MAX_SAMPLES
from sequence_features import build_features, classify_column

FEATURE_CACHE_DIR = Path(RESULTS_DIR) / "feature_cache"


def _cache_path(path, species):
    stem = Path(path).stem
    suffix = f"n{MAX_SAMPLES}" if MAX_SAMPLES is not None else "full"
    return FEATURE_CACHE_DIR / f"{stem}_{species}_{suffix}.parquet"


def _select_columns(columns, ablation_spec):
    """
    Filter a list of column names down to those matching an ablation spec.

    ablation_spec : dict {"groups": [...], "regions": [...] or None}
        or a plain list of group names (back-compat: treated as
        {"groups": that list, "regions": None})
        or None (take every column, unfiltered — the full feature set).
    """
    if ablation_spec is None:
        return list(columns)

    if isinstance(ablation_spec, list):
        ablation_spec = {"groups": ablation_spec, "regions": None}

    groups  = ablation_spec.get("groups")
    regions = ablation_spec.get("regions")

    selected = []
    for c in columns:
        g, r = classify_column(c)
        if groups is not None and g not in groups:
            continue
        if regions is not None and r not in regions:
            continue
        selected.append(c)
    return selected


def load_and_prepare(path, te_col, species, ablation_spec=None):
    """
    Load an Excel file and return a feature matrix (sliced to
    `ablation_spec`), target vector, fold assignments, and feature names.

    Parameters
    ----------
    path          : str   path to the Excel file
    te_col        : str   name of the translation efficiency column to predict
    species       : str   "human" or "mouse" — selects CAI/TAI weight columns
    ablation_spec : dict {"groups": [...], "regions": [...] or None}, or a
        plain list of group names, or None.
        None means "every feature group, every region" (the full model).

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
    cache_file = _cache_path(path, species)

    if cache_file.exists():
        print(f"  Loading cached full feature matrix: {cache_file}")
        X_full = pd.read_parquet(cache_file)
        if not X_full.index.equals(df.index):
            print("  Cache index does not match current dataframe — rebuilding cache.")
            X_full = build_features(df, species, groups=None)
            X_full.to_parquet(cache_file)
    else:
        print("  No cache found — building full feature matrix (all groups, incl. MFE)...")
        X_full = build_features(df, species, groups=None)
        X_full.to_parquet(cache_file)
        print(f"  Cached: {cache_file}")

    keep_cols = _select_columns(X_full.columns, ablation_spec)
    if not keep_cols:
        raise ValueError(
            f"Ablation spec {ablation_spec} matched zero columns in the cached "
            f"feature matrix — check the group/region names against "
            f"sequence_features.FEATURE_FUNCS / VALID_REGIONS."
        )
    X = X_full[keep_cols]

    y             = df[te_col].values
    feature_names = X.columns.tolist()

    # constant-fill imputation — fills NaN with 0 (unchanged from original)
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_arr   = imputer.fit_transform(X)

    print(f"  Feature matrix shape: {X_arr.shape}  (spec: {ablation_spec or 'ALL'})")
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