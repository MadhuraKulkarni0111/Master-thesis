"""
models.py
=========
Model definitions and cross-validated training.

Functions
---------
build_models()
    → dict of {model_name: unfitted sklearn-compatible model}

fit_models(X, y, folds, feature_names, label)
    → dict of {model_name: fitted model}
"""

import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict  
from sklearn.metrics import r2_score                  
import lightgbm as lgb

from config import (
    LASSO_ALPHA,
    ELASTICNET_ALPHA, ELASTICNET_L1RATIO,
    RF_N_ESTIMATORS, RF_MAX_FEATURES, RF_RANDOM_STATE,
    LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_NUM_LEAVES,
    LGBM_MIN_CHILD_SAMPLES, LGBM_SUBSAMPLE, LGBM_COLSAMPLE,
    LGBM_RANDOM_STATE,
)
from data_loader import make_predefined_splits


def build_models():
    """
    Instantiate all four models with their hyperparameters from config.py.

    Model descriptions
    ------------------

    Lasso
        Linear regression with L1 regularisation. The penalty term is the
        sum of absolute coefficient values multiplied by alpha. L1 drives
        unimportant coefficients to exactly zero, performing automatic
        feature selection. Wrapped in a Pipeline with StandardScaler so
        all 133 features are on a common scale before the penalty is applied
        — without scaling, features measured in large units (e.g. cds_size
        in nucleotides) would be penalised more heavily than unit-less
        frequencies just because of their magnitude.

    ElasticNet
        Combines L1 (Lasso) and L2 (Ridge) penalties. l1_ratio=0.5 means
        equal weighting. The L2 component makes it more stable than pure
        Lasso when features are correlated — which nucleotide frequencies
        always are (if G goes up, at least one of A/T/C must go down).
        Also wrapped in a StandardScaler Pipeline.

    RandomForest
        Ensemble of 300 decision trees, each trained on a bootstrap sample
        of genes and a random subset of sqrt(n_features) ≈ 11 features at
        each split. Aggregating many trees reduces variance and makes the
        model robust to outliers. Feature importance is reported as mean
        decrease in impurity (MDI) — how much each feature reduces the
        variance of TE summed across all splits in all trees. Scale-invariant,
        so no StandardScaler is needed.

    LightGBM
        Gradient-boosted decision trees trained sequentially: each new tree
        corrects the residual errors left by all previous trees. Key
        differences from RandomForest:
          - sequential (boosting) vs parallel (bagging) tree building
          - leaf-wise tree growth (grows the leaf with the highest gain)
            rather than level-wise, giving deeper and more expressive trees
          - importance_type="gain" reports total information gain from each
            feature across all splits, which is more informative than split
            count because it weights each use by how much it reduced error.
        subsample=0.8 and colsample_bytree=0.8 add randomness similar to RF's
        bagging, reducing overfitting. Also scale-invariant.

    Returns
    -------
    dict : {model_name: unfitted model}
    """
    return {
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(
                alpha=LASSO_ALPHA,
                max_iter=10000,
                random_state=42
            ))
        ]),

        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(
                alpha=ELASTICNET_ALPHA,
                l1_ratio=ELASTICNET_L1RATIO,
                max_iter=10000,
                random_state=42
            ))
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_features=RF_MAX_FEATURES,
            n_jobs=-1,
            random_state=RF_RANDOM_STATE
        ),

        "LightGBM": lgb.LGBMRegressor(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            subsample=LGBM_SUBSAMPLE,
            colsample_bytree=LGBM_COLSAMPLE,
            importance_type="gain",
            n_jobs=-1,
            random_state=LGBM_RANDOM_STATE,
            verbose=-1
        ),
    }


def fit_models(X, y, folds, feature_names, label):
    """
    Cross-validate and then fully fit all four models.

    For each model:
      1. Run cross_val_score using the pre-defined dataset splits to get
         one R² per fold. This measures generalisation — how well the model
         predicts TE for genes it has never seen during training.
      2. Print the mean R² ± standard deviation and all per-fold scores.
      3. Refit the model on the FULL dataset so that feature importances
         and coefficients are computed from all available data.

    Why refit on the full dataset after CV?
    The CV scores tell you how well the model generalises. The final fit
    on all data gives you the most information-rich set of coefficients
    and importances for biological interpretation — using only 90% of
    the data for the final model would throw away useful signal.

    Parameters
    ----------
    X            : np.ndarray  shape (n_genes, n_features)
    y            : np.ndarray  shape (n_genes,)
    folds        : np.ndarray  shape (n_genes,)  — fold assignments
    feature_names: list of str
    label        : str  — dataset name for display

    Returns
    -------
    dict : {model_name: fitted model}
    """
    cv_splits = make_predefined_splits(folds)
    n_folds   = len(cv_splits)
    models    = build_models()
    fitted    = {}

    print(f"\n{'='*60}")
    print(f"  Dataset : {label}")
    print(f"  Samples : {X.shape[0]}   Features: {X.shape[1]}")
    print(f"  CV      : {n_folds}-fold  (pre-defined dataset splits)")
    print(f"{'='*60}")

    for name, model in models.items():
        # 1. Generate out-of-fold predictions for the entire dataset
        y_pred = cross_val_predict(
            model, X, y,
            cv=cv_splits,
            n_jobs=-1
        )
        
        # 2. Calculate a singular, comprehensive R² score 
        overall_r2 = r2_score(y, y_pred)
        print(f"  {name:<14}  Overall CV R² = {overall_r2:.4f}")

        # 3. Refit on full dataset for importance extraction
        model.fit(X, y)
        fitted[name] = model

    return fitted
