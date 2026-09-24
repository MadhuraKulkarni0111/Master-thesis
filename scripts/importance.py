"""
importance.py
=============
Extract feature importances and signed coefficients from fitted models.

Functions
---------
get_importances(fitted_models, feature_names, X, y)
    → dict of {model_name: pd.Series(importance, index=feature_names)}

save_importance_csv(imp_dict, fitted_models, feature_names, label, out_dir)
    → str  (path to saved CSV, written into out_dir)
"""

import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import TOP_N_CSV

# Models whose importance is |coefficient| from a Pipeline
LINEAR_MODELS = ("Lasso", "ElasticNet", "LinearSVM")


def get_importances(fitted_models, feature_names, X=None, y=None):
    """
    Extract a feature importance score for every feature from each model.

    Lasso / ElasticNet / LinearSVM : |coefficient| from Pipeline
    RandomForest                    : MDI (mean decrease in impurity)
    LightGBM / XGBoost              : total gain across all splits
    SVR (RBF)                       : permutation importance — mean R² drop
                                      when each feature is shuffled.
                                      X and y must be provided for this.
    """
    imp = {}

    # ── Linear models: |coefficient| ─────────────────────────────────────
    for mname in LINEAR_MODELS:
        if mname in fitted_models:
            coef = fitted_models[mname].named_steps["model"].coef_
            imp[mname] = pd.Series(
                np.abs(coef.ravel()),
                index=feature_names
            )

    # ── Tree models ───────────────────────────────────────────────────────
    if "RandomForest" in fitted_models:
        imp["RandomForest"] = pd.Series(
            fitted_models["RandomForest"].feature_importances_,
            index=feature_names
        )
    if "LightGBM" in fitted_models:
        imp["LightGBM"] = pd.Series(
            fitted_models["LightGBM"].feature_importances_,
            index=feature_names
        )
    if "XGBoost" in fitted_models:
        imp["XGBoost"] = pd.Series(
            fitted_models["XGBoost"].feature_importances_,
            index=feature_names
        )

    # ── SVR (RBF): permutation importance ────────────────────────────────
    # n_repeats=5 kept at full precision (not reduced) per your call to
    # avoid compromising results — rely on --cpus-per-task for speed
    # instead (permutation_importance parallelises over features).
    if "SVR" in fitted_models:
        if X is None or y is None:
            raise ValueError(
                "X and y must be passed to get_importances() "
                "when SVR is in the model set (needed for permutation importance)."
            )
        print("  Computing permutation importance for SVR "
              "(this may take a while)...")
        perm = permutation_importance(
            fitted_models["SVR"], X, y,
            scoring="r2",
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        imp["SVR"] = pd.Series(
            np.clip(perm.importances_mean, 0, None),
            index=feature_names
        )

    return imp


def save_importance_csv(imp_dict, fitted_models, feature_names,
                        label, out_dir):
    """
    Save the top-N features per model to a CSV file inside out_dir.

    signed_coef is populated for linear models only (Lasso, ElasticNet,
    LinearSVM). Tree models and RBF SVR get np.nan in that column.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_label = label.replace(" ", "_")
    csv_path   = os.path.join(out_dir, f"{safe_label}_top_features.csv")
    rows       = []

    for mname, imp in imp_dict.items():
        for feat, val in imp.nlargest(TOP_N_CSV).items():
            signed = np.nan
            if mname in LINEAR_MODELS:
                coef = fitted_models[mname].named_steps["model"].coef_.ravel()
                signed = coef[feature_names.index(feat)]
            rows.append({
                "dataset":     label,
                "model":       mname,
                "feature":     feat,
                "importance":  val,
                "signed_coef": signed,
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return csv_path