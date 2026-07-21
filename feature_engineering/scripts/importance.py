"""
importance.py
=============
Extract feature importances and signed coefficients from fitted models.
 
Functions
---------
get_importances(fitted_models, feature_names, X, y)
    → dict of {model_name: pd.Series(importance, index=feature_names)}
 
save_importance_csv(imp_dict, fitted_models, feature_names, label, out_prefix)
    → str  (path to saved CSV)
 
Note on SVR importance
----------------------
RBF SVR has no coef_ attribute — the kernel trick maps features into a
high-dimensional space where there is no linear coefficient to read off.
Permutation importance is used instead: each feature column is shuffled
independently, the model predicts on the shuffled data, and the drop in
R² measures how much the model relied on that feature. Features the model
ignores show no R² drop; features the model heavily relies on show a large
drop. This is model-agnostic and works for any kernel or model type.
"""
 
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
 
from config import TOP_N_CSV
 
# Models that expose coef_ (linear models inside a Pipeline)
LINEAR_MODELS = ("Lasso", "ElasticNet")
 
# Models that use kernel tricks or ensembles — no coef_ available
# importance computed via permutation importance
PERMUTATION_MODELS = ("SVR",)
 
 
def get_importances(fitted_models, feature_names, X, y):
    """
    Extract a feature importance score for every feature from each model.
 
    The importance metric differs by model type:
 
    Lasso / ElasticNet
        Absolute value of the learned regression coefficient |β|.
 
    RandomForest
        Mean Decrease in Impurity (MDI).
 
    LightGBM / XGBoost
        Total gain across all splits (importance_type="gain").
 
    SVR (RBF kernel)
        Permutation importance: each feature is shuffled independently
        and the mean drop in R² across 5 repeats is recorded. A large
        drop means the model relied heavily on that feature. Positive
        values only (importance = mean drop; negative means shuffling
        actually improved predictions, i.e. the feature added noise).
        Uses the full training set (X, y) passed in from run_pipeline.py.
 
    Parameters
    ----------
    fitted_models : dict  {model_name: fitted sklearn/lgbm/xgb model}
    feature_names : list of str
    X             : np.ndarray  shape (n_genes, n_features)
                    Required for permutation importance (SVR).
    y             : np.ndarray  shape (n_genes,)
                    Required for permutation importance (SVR).
 
    Returns
    -------
    dict : {model_name: pd.Series}
    """
    imp = {}
 
    # ── Linear models: |coefficient| ─────────────────────────────────────
    for mname in LINEAR_MODELS:
        imp[mname] = pd.Series(
            np.abs(fitted_models[mname].named_steps["model"].coef_),
            index=feature_names
        )
 
    # ── Tree models: built-in importances ────────────────────────────────
    imp["RandomForest"] = pd.Series(
        fitted_models["RandomForest"].feature_importances_,
        index=feature_names
    )
    imp["LightGBM"] = pd.Series(
        fitted_models["LightGBM"].feature_importances_,
        index=feature_names
    )
    imp["XGBoost"] = pd.Series(
        fitted_models["XGBoost"].feature_importances_,
        index=feature_names
    )
 
    # ── SVR (RBF): permutation importance ────────────────────────────────
    # n_repeats=5 shuffles each feature 5 times and averages the R² drop
    # — more repeats = more stable estimate but slower; 5 is a good default
    print("  Computing permutation importance for SVR "
          "(this may take a few minutes)...")
    perm = permutation_importance(
        fitted_models["SVR"], X, y,
        scoring="r2",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    # importances_mean is the mean R² drop per feature across repeats
    # clip at 0 so features that add noise don't appear as negative importance
    imp["SVR"] = pd.Series(
        np.clip(perm.importances_mean, 0, None),
        index=feature_names
    )
 
    return imp
 
def save_importance_csv(imp_dict, fitted_models, feature_names,
                        label, out_prefix):
    """
    Save the top-N features per model to a CSV file.
 
    Columns
    -------
    dataset     : label string identifying the cell line / species
    model       : model name
    feature     : feature name
    importance  : importance score (|coef|, MDI, gain, or permutation R² drop)
    signed_coef : signed coefficient for Lasso and ElasticNet only.
                  np.nan for tree models and SVR (no signed coef available).
                  Positive = higher feature value → higher TE
                  Negative = higher feature value → lower TE
    """
    safe_label = label.replace(" ", "_")
    csv_path   = f"{out_prefix}_{safe_label}_top_features.csv"
    rows       = []
 
    for mname, imp in imp_dict.items():
        for feat, val in imp.nlargest(TOP_N_CSV).items():
            signed = np.nan
            if mname in LINEAR_MODELS:
                signed = fitted_models[mname].named_steps["model"].coef_[
                    feature_names.index(feat)
                ]
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
 