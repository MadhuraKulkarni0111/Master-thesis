"""
data_loader.py
==============
Loads an Excel dataset, calls feature engineering, and returns
clean numpy arrays ready for model training.

Functions
---------
load_and_prepare(path, te_col)
    → X_arr, y, folds, feature_names

make_predefined_splits(folds)
    → list of (train_idx, test_idx) tuples
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from config import RESULTS_DIR, MAX_SAMPLES   # add MAX_SAMPLES to the existing import

from sequence_features import build_features


def load_and_prepare(path, te_col):
    """
    Load an Excel file and return a feature matrix, target vector,
    fold assignments, and feature names.

    Steps performed
    ---------------
    1. Read the Excel file into a DataFrame.
    2. Drop rows where the TE target or tx_sequence is missing —
       we cannot train on genes without a sequence or a measured TE value.
    3. Call build_features() to engineer all 133 sequence features.
    4. Extract the pre-defined fold column from the dataset — these are
       the cross-validation split assignments that came with the data,
       ensuring our CV results are reproducible and comparable with
       published benchmarks that used the same splits.
    5. Impute any remaining NaN values using the column median.
       NaNs arise when a region is empty (e.g. a gene with no annotated
       5'UTR produces NaN for all utr5_* frequency features).

    Parameters
    ----------
    path   : str   path to the Excel file
    te_col : str   name of the translation efficiency column to predict

    Returns
    -------
    X_arr        : np.ndarray  shape (n_genes, 133)  — feature matrix
    y            : np.ndarray  shape (n_genes,)       — TE target values
    folds        : np.ndarray  shape (n_genes,)       — fold assignments (0–9)
    feature_names: list of str — column names matching X_arr columns
    """
    print(f"\nLoading {path}  →  target: {te_col}")
    df = pd.read_excel(path)
    df = df.dropna(subset=[te_col, "tx_sequence"])
    print(f"  Rows after dropping NA in target/sequence: {len(df)}")

    # ── Subset for quick testing ───────────────────────────────────────────────
    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)
        print(f"  Subsampled to {MAX_SAMPLES} sequences (MAX_SAMPLES is set in config.py)")

    # preserve fold assignments before feature engineering changes the index
    folds = df["fold"].values

    X             = build_features(df)
    y             = df[te_col].values
    feature_names = X.columns.tolist()

    # median imputation — fills NaN with the median of each feature column
    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)

    print(f"  Feature matrix shape: {X_arr.shape}")
    print(f"  Fold distribution: "
          f"{dict(zip(*np.unique(folds, return_counts=True)))}")

    return X_arr, y, folds, feature_names


def make_predefined_splits(folds):
    """
    Convert the dataset's fold column into sklearn-compatible CV splits.

    Each unique fold value is held out once as the test set while all
    remaining folds form the training set. For 10 folds this produces
    10 iterations, each testing on ~1,000 genes and training on ~9,000.

    Parameters
    ----------
    folds : np.ndarray  shape (n_genes,)
        Integer fold assignments, e.g. 0–9 for 10-fold CV.

    Returns
    -------
    list of (train_idx, test_idx) tuples
        Each tuple contains numpy arrays of integer indices into the
        full dataset. Pass directly to sklearn's cv parameter.

    Example
    -------
    splits = make_predefined_splits(folds)
    scores = cross_val_score(model, X, y, cv=splits, scoring="r2")
    """
    splits = []
    for held_out in sorted(np.unique(folds)):
        train_idx = np.where(folds != held_out)[0]
        test_idx  = np.where(folds == held_out)[0]
        splits.append((train_idx, test_idx))
    return splits
