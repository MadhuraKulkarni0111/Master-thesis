"""
importance.py
=============
Extract feature importances and signed coefficients from fitted models.
 
Functions
---------
get_importances(fitted_models, feature_names)
    → dict of {model_name: pd.Series(importance, index=feature_names)}
 
save_importance_csv(imp_dict, fitted_models, feature_names, label, out_prefix)
    → str  (path to saved CSV)
"""
 
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
 
from config import TOP_N_CSV
 
# Models whose importance is |coefficient| from a Pipeline
LINEAR_MODELS = ("Lasso", "ElasticNet", "LinearSVM")
 
# Models with no coef_ — importance via permutation
#PERMUTATION_MODELS = ("SVR")
 
 
def get_importances(fitted_models, feature_names, X=None, y=None):
    """
    Extract a feature importance score for every feature from each model.
 
    Lasso / ElasticNet / LinearSVM : |coefficient| from Pipeline
    RandomForest                    : MDI (mean decrease in impurity)
    LightGBM / XGBoost             : total gain across all splits
    SVR (RBF)                       : permutation importance — mean R² drop
                                      when each feature is shuffled.
                                      X and y must be provided for this.
    """
    imp = {}
 
    # ── Linear models: |coefficient| ─────────────────────────────────────
    for mname in LINEAR_MODELS:
        if mname in fitted_models:
            coef = fitted_models[mname].named_steps["model"].coef_
            # LinearSVR.coef_ is shape (1, n_features) — flatten if needed
            imp[mname] = pd.Series(
                np.abs(coef.ravel()),
                index=feature_names
                )
 
    # ── Tree models ───────────────────────────────────────────────────────
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
    # RBF SVR is evaluated only using OOF R².
    # Feature importance is intentionally omitted because
    # permutation importance is computationally expensive.
    # print("Skipping permutation impirtance calculation for SVR becuase it is computationally complex ")
    if "SVR" in fitted_models:
        if X is None or y is None:
            raise ValueError(
                "X and y must be passed to get_importances() "
                "when SVR is in the model set (needed for permutation importance)."
            )
        print("  Computing permutation importance for SVR "
              "(this may take a few minutes)...")
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
                        label, out_prefix):
    """
    Save the top-N features per model to a CSV file.
    signed_coef is populated for linear models only (Lasso, ElasticNet,
    LinearSVM). Tree models and RBF SVR get np.nan in that column.
    """
    safe_label = label.replace(" ", "_")
    csv_path   = f"{out_prefix}_{safe_label}_top_features.csv"
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