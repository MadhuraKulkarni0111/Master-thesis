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

from config import TOP_N_CSV

def get_importances(fitted_models, feature_names):
    """
    Extract a feature importance score for every feature from each model.

    The importance metric differs by model type:

    Lasso / ElasticNet
        Absolute value of the learned regression coefficient |β|.
        The coefficient tells you exactly how much the model expects TE
        to change for a one-unit increase in that feature (after scaling).
        Taking the absolute value gives magnitude; the sign is preserved
        separately in save_importance_csv() for directional interpretation.

    RandomForest
        Mean Decrease in Impurity (MDI): how much each feature reduces
        the variance of TE values across all splits in all 300 trees,
        averaged and normalised to sum to 1. Higher = more useful for
        splitting genes into high-TE vs low-TE groups.

    LightGBM
        Total gain: the sum of information gain (reduction in squared error)
        attributed to each feature across all splits in all boosting trees.
        This is more reliable than split count (the alternative) because it
        weights each use of a feature by how much it actually helped reduce
        prediction error — a feature used once for a highly informative split
        scores higher than a feature used 100 times for trivial splits.

    Parameters
    ----------
    fitted_models : dict  {model_name: fitted sklearn/lgbm model}
    feature_names : list of str

    Returns
    -------
    dict : {model_name: pd.Series}
        Each Series has feature names as index and importance as values,
        sorted in no particular order (sorting happens in the plot functions).
    """
    imp = {}

    # linear models: |coefficient| as importance
    imp["Lasso"] = pd.Series(
        np.abs(fitted_models["Lasso"].named_steps["model"].coef_),
        index=feature_names
    )
    imp["ElasticNet"] = pd.Series(
        np.abs(fitted_models["ElasticNet"].named_steps["model"].coef_),
        index=feature_names
    )

    # tree models: built-in feature_importances_ attribute
    imp["RandomForest"] = pd.Series(
        fitted_models["RandomForest"].feature_importances_,
        index=feature_names
    )
    # LightGBM: importance_type="gain" was set in build_models()
    imp["LightGBM"] = pd.Series(
        fitted_models["LightGBM"].feature_importances_,
        index=feature_names
    )
    # XGBoost: importance_type="gain" was set in build_models()
    imp["XGBoost"] = pd.Series(
        fitted_models["XGBoost"].feature_importances_,
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
    importance  : importance score (|coef|, MDI, or gain depending on model)
    signed_coef : the raw signed coefficient for Lasso and ElasticNet
                  (np.nan for tree models which have no signed coefficients).
                  Positive = higher feature value → higher TE
                  Negative = higher feature value → lower TE

    Parameters
    ----------
    imp_dict      : dict  {model_name: pd.Series of importances}
    fitted_models : dict  {model_name: fitted model}
    feature_names : list of str
    label         : str   dataset label
    out_prefix    : str   path prefix; CSV saved as <out_prefix>_top_features.csv

    Returns
    -------
    str : path to the saved CSV file
    """
    safe_label = label.replace(" ", "_")
    csv_path = f"{out_prefix}_{safe_label}_top_features.csv"    
    #csv_path = f"{out_prefix}_top_features.csv"
    rows = []

    for mname, imp in imp_dict.items():
        print("Processing:", mname)
        top_features = imp.nlargest(TOP_N_CSV)
        print(top_features.head())
        for feat, val in imp.nlargest(TOP_N_CSV).items():
            signed = np.nan
            if mname in ("Lasso", "ElasticNet"):
                # retrieve the original signed coefficient from the pipeline
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
