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
from xgboost import XGBRegressor
from sklearn.svm import SVR, LinearSVR
 
 
from config import (
    LASSO_ALPHA,
    ELASTICNET_ALPHA, ELASTICNET_L1RATIO,
    RF_N_ESTIMATORS, RF_MAX_FEATURES, RF_RANDOM_STATE,
    LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE, LGBM_NUM_LEAVES,
    LGBM_MIN_CHILD_SAMPLES, LGBM_SUBSAMPLE, LGBM_COLSAMPLE,
    LGBM_RANDOM_STATE,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH,
    XGB_SUBSAMPLE, XGB_COLSAMPLE, XGB_RANDOM_STATE,
    SVM_C, SVM_EPSILON, SVM_KERNEL, SVM_GAMMA,
)
 
from data_loader import make_predefined_splits
 
 
def build_models():
    """
    Instantiate all five models with their hyperparameters from config.py.
 
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
 
    XGBoost 
        Gradient-boosted decision trees built sequentially to minimize
        prediction error. Unlike LightGBM's leaf-wise growth strategy,
        XGBoost typically grows trees level-wise, making it more
        conservative and often less prone to overfitting. It includes
        built-in regularization and supports feature importance based on
        gain, cover, or split frequency. Like RandomForest and LightGBM,
        it is scale-invariant and does not require StandardScaler.
 
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
 
        "XGBoost": XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            random_state=XGB_RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1
        ),
 
        # RBF SVR — no coef_ available, importance via permutation in importance.py
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel=SVM_KERNEL,
                C=SVM_C,
                epsilon=SVM_EPSILON,
                gamma=SVM_GAMMA,
            ))
        ]),
 
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVR(
                C=SVM_C,
                epsilon=SVM_EPSILON,
                random_state=42,
                max_iter=10000
            ))
        ]),
    }
 
 
def fit_models(X, y, folds, feature_names, label):
    """
    Cross-validate (out-of-fold) and then fully fit all five models.
 
    Uses cross_val_predict instead of cross_val_score. The difference:
 
      cross_val_score   → returns one R² per fold (10 numbers), then you
                          average them. Each fold is weighted equally
                          regardless of size.
 
      cross_val_predict → returns one PREDICTION per gene (not a score).
                          Every gene gets predicted exactly once, by
                          whichever fold held it out as test data. These
                          predictions are then compared to the true y
                          values ALL AT ONCE to get a single R² — this is
                          the "out-of-fold" (OOF) R², which weights every
                          gene equally rather than every fold equally.
 
    Both per-fold R² (for variance/stability across folds) and the single
    OOF R² (the more standard reported metric) are computed and returned.
 
    Parameters
    ----------
    X            : np.ndarray  shape (n_genes, n_features)
    y            : np.ndarray  shape (n_genes,)
    folds        : np.ndarray  shape (n_genes,)  — fold assignments
    feature_names: list of str
    label        : str  — dataset name for display
 
    Returns
    -------
    fitted     : dict  {model_name: fitted model}
    cv_scores  : dict  {model_name: dict with keys
                         "oof_r2"        → single float, the pooled R²
                         "per_fold_r2"   → np.ndarray, one R² per fold
                         "oof_predictions" → np.ndarray, per-gene predictions
                        }
    """
    cv_splits = make_predefined_splits(folds)
    n_folds   = len(cv_splits)
    models    = build_models()
    fitted    = {}
    cv_scores = {}
 
    print(f"\n{'='*60}")
    print(f"  Dataset : {label}")
    print(f"  Samples : {X.shape[0]}   Features: {X.shape[1]}")
    print(f"  CV      : {n_folds}-fold  (pre-defined dataset splits)")
    print(f"{'='*60}")
 
    for name, model in models.items():
        # one prediction per gene, made by the fold that held it out
        oof_preds = cross_val_predict(
            model, X, y,
            cv=cv_splits,
            n_jobs=-1
        )
 
        # single pooled R² across all genes (out-of-fold R²)
        oof_r2 = r2_score(y, oof_preds)
 
        # also compute per-fold R² for stability/variance reporting
        per_fold_r2 = np.array([
            r2_score(y[test_idx], oof_preds[test_idx])
            for _, test_idx in cv_splits
        ])
 
        print(f"  {name:<14}  OOF R² = {oof_r2:.4f}   "
              f"(per-fold mean = {per_fold_r2.mean():.4f} "
              f"± {per_fold_r2.std():.4f})")
        print(f"  {'':14}  per fold: "
              f"{', '.join(f'{s:.3f}' for s in per_fold_r2)}")
 
        cv_scores[name] = {
            "oof_r2":          oof_r2,
            "per_fold_r2":     per_fold_r2,
            "oof_predictions": oof_preds,
        }
 
        # refit on full dataset for importance extraction
        model.fit(X, y)
        fitted[name] = model
        print(fitted.keys())
 
    return fitted, cv_scores
 