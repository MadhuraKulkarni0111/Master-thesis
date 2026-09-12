"""
model_results.py
================
Export cross-validated R² scores to a CSV file inside the current
ablation's output directory.

Functions
---------
save_r2_results(cv_scores, label, out_dir)
    → str  (path to saved CSV)
"""

import os

import numpy as np
import pandas as pd


def save_r2_results(cv_scores, label, out_dir):
    """
    Save per-fold, out-of-fold (OOF), and summary R² scores to a CSV file.

    Creates out_dir if it doesn't already exist. The output file is named
    <label>_r2_results.csv, e.g. "Human_HCT116_r2_results.csv", written
    inside out_dir (one directory per ablation, e.g. results/A7/).

    Output columns
    --------------
    dataset   : the label passed in (e.g. "Human_HCT116")
    model     : model name (Lasso, ElasticNet, RandomForest, LightGBM, ...)
    fold      : fold index (0, 1, 2, ...) for per-fold rows,
                "mean" / "std" for per-fold summary rows,
                or "oof" for the single pooled out-of-fold R²
    r2        : the R² value for that row

    Parameters
    ----------
    cv_scores : dict  {model_name: {"oof_r2": float,
                                    "per_fold_r2": np.ndarray,
                                    "oof_predictions": np.ndarray}}
        Returned by fit_models() in models.py.
    label     : str
        Dataset label, used both in the output filename and the
        'dataset' column of the CSV.
    out_dir   : str
        Directory this ablation's outputs are written to.

    Returns
    -------
    str : path to the saved CSV file
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for model_name, result in cv_scores.items():
        per_fold_r2 = result["per_fold_r2"]
        oof_r2      = result["oof_r2"]

        for fold_idx, score in enumerate(per_fold_r2):
            rows.append({
                "dataset": label,
                "model":   model_name,
                "fold":    fold_idx,
                "r2":      score,
            })
        rows.append({
            "dataset": label, "model": model_name,
            "fold": "mean", "r2": np.mean(per_fold_r2)
        })
        rows.append({
            "dataset": label, "model": model_name,
            "fold": "std", "r2": np.std(per_fold_r2)
        })
        rows.append({
            "dataset": label, "model": model_name,
            "fold": "oof", "r2": oof_r2
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{label}_r2_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"  Saved: {csv_path}")
    return csv_path
