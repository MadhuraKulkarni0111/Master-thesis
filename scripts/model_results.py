"""
results.py
==========
Export cross-validated evaluation scores to a CSV file in the results folder.

Functions
---------
save_r2_results(cv_scores, label)
    → str  (path to saved CSV)
    Legacy, R²-only export. Left untouched so any ablation scripts that
    import this keep their existing behaviour/output format.

compute_metrics(y_true, y_pred)
    → dict  {"r2", "mae", "rmse", "pearson_r", "spearman_r"}

save_full_results(cv_scores, y, folds, label)
    → str  (path to saved CSV)
    Full-pipeline-only evaluation export (R² + MAE + RMSE + Pearson r +
    Spearman ρ), per fold and pooled OOF. Not used by the ablation runs.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from config import RESULTS_DIR
from data_loader import make_predefined_splits


def save_r2_results(cv_scores, label):
    """
    Save per-fold, out-of-fold (OOF), and summary R² scores to a CSV file.
 
    Not applicable folder already exists:
    Creates RESULTS_DIR (from config.py) if it doesn't already exist.
    The output file is named <label>_r2_results.csv, e.g.
    "Human_HCT116_r2_results.csv".
 
    Output columns
    --------------
    dataset   : the label passed in (e.g. "Human_HCT116")
    model     : model name (Lasso, ElasticNet, RandomForest, LightGBM)
    fold      : fold index (0, 1, 2, ...) for per-fold rows,
                "mean" / "std" for per-fold summary rows,
                or "oof" for the single pooled out-of-fold R²
    r2        : the R² value for that row
 
    The "oof" row is the headline number to report — it is the R² computed
    once across every gene's out-of-fold prediction, rather than an average
    of 10 separate fold scores. The per-fold rows are kept alongside it so
    you can still see how stable the model is across different subsets
    of genes.
 
    Parameters
    ----------
    cv_scores : dict  {model_name: {"oof_r2": float,
                                    "per_fold_r2": np.ndarray,
                                    "oof_predictions": np.ndarray}}
        Returned by fit_models() in models.py.
    label     : str
        Dataset label, used both in the output filename and the
        'dataset' column of the CSV.
 
    Returns
    -------
    str : path to the saved CSV file
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    rows = []
    for model_name, result in cv_scores.items():
        per_fold_r2 = result["per_fold_r2"]
        oof_r2      = result["oof_r2"]
 
        # one row per fold
        for fold_idx, score in enumerate(per_fold_r2):
            rows.append({
                "dataset": label,
                "model":   model_name,
                "fold":    fold_idx,
                "r2":      score,
            })
        # per-fold summary rows
        rows.append({
            "dataset": label, "model": model_name,
            "fold": "mean", "r2": np.mean(per_fold_r2)
        })
        rows.append({
            "dataset": label, "model": model_name,
            "fold": "std", "r2": np.std(per_fold_r2)
        })
        # single pooled out-of-fold R² — the headline metric
        rows.append({
            "dataset": label, "model": model_name,
            "fold": "oof", "r2": oof_r2
        })
 
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"{label}_r2_results.csv")
    df.to_csv(csv_path, index=False)
 
    print(f"  Saved: {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────
# Full-pipeline evaluation (R² + additional metrics)
# NOTE: intentionally kept separate from save_r2_results() above so the
# ablation pipeline (which calls save_r2_results) is completely unaffected.
# ─────────────────────────────────────────────────────────────────────────

METRIC_NAMES = ["r2", "mae", "rmse", "pearson_r", "spearman_r"]


def compute_metrics(y_true, y_pred):
    """
    Compute a small suite of regression evaluation metrics for one
    set of true values / predictions.

    r2         : coefficient of determination (existing metric)
    mae        : mean absolute error, same units as TE — robust to outliers,
                 easy to interpret ("predictions are off by X on average")
    rmse       : root mean squared error, same units as TE — penalises large
                 errors more than MAE, standard alongside R² in regression
    pearson_r  : linear correlation between predicted and true TE
    spearman_r : rank correlation between predicted and true TE — useful
                 here because TE measurements are noisy and downstream use
                 (e.g. ranking genes by translation efficiency) often only
                 needs the ORDER to be right, not the exact value

    Parameters
    ----------
    y_true, y_pred : array-like, same length

    Returns
    -------
    dict with keys matching METRIC_NAMES. pearson_r / spearman_r are NaN
    if either array is constant (correlation undefined) or has <2 points.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    try:
        metrics["pearson_r"] = pearsonr(y_true, y_pred)[0]
    except Exception:
        metrics["pearson_r"] = np.nan

    try:
        metrics["spearman_r"] = spearmanr(y_true, y_pred)[0]
    except Exception:
        metrics["spearman_r"] = np.nan

    return metrics


def save_full_results(cv_scores, y, folds, label):
    """
    Save per-fold, out-of-fold (OOF), and summary scores for R², MAE,
    RMSE, Pearson r, and Spearman ρ to a CSV file.

    Mirrors the row layout of save_r2_results() (one row per fold, plus
    "mean" / "std" / "oof" summary rows) but with one column per metric
    instead of a single "r2" column, so existing downstream CSV-reading
    code that expects the old file (<label>_r2_results.csv) still finds it
    untouched — this writes a separate <label>_eval_results.csv.

    Parameters
    ----------
    cv_scores : dict  {model_name: {"oof_r2": float,
                                    "per_fold_r2": np.ndarray,
                                    "oof_predictions": np.ndarray}}
        Returned by fit_models() in models.py.
    y     : np.ndarray  shape (n_genes,)  — true TE values (same y passed
            into fit_models)
    folds : np.ndarray  shape (n_genes,)  — fold assignments (same folds
            passed into fit_models); used to rebuild the per-fold test
            indices so metrics can be computed against y directly.
    label : str
        Dataset label, used in the output filename and 'dataset' column.

    Returns
    -------
    str : path to the saved CSV file
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y = np.asarray(y)
    splits = make_predefined_splits(folds)

    rows = []
    for model_name, result in cv_scores.items():
        oof_preds = result["oof_predictions"]

        # one row per fold, all metrics computed on that fold's held-out genes
        per_fold_metrics = []
        for fold_idx, (_, test_idx) in enumerate(splits):
            m = compute_metrics(y[test_idx], oof_preds[test_idx])
            per_fold_metrics.append(m)
            rows.append({"dataset": label, "model": model_name, "fold": fold_idx, **m})

        # per-fold summary rows (mean / std across folds, per metric)
        mean_row = {"dataset": label, "model": model_name, "fold": "mean"}
        std_row  = {"dataset": label, "model": model_name, "fold": "std"}
        for mn in METRIC_NAMES:
            vals = [m[mn] for m in per_fold_metrics]
            mean_row[mn] = np.nanmean(vals)
            std_row[mn]  = np.nanstd(vals)
        rows.append(mean_row)
        rows.append(std_row)

        # single pooled out-of-fold row — the headline metrics
        oof_row = {"dataset": label, "model": model_name, "fold": "oof"}
        oof_row.update(compute_metrics(y, oof_preds))
        rows.append(oof_row)

    df = pd.DataFrame(rows, columns=["dataset", "model", "fold"] + METRIC_NAMES)
    csv_path = os.path.join(RESULTS_DIR, f"{label}_eval_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"  Saved: {csv_path}")
    return csv_path