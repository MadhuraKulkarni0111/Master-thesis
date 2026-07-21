"""
run_pipeline.py
===============
Main entry point for the TE feature importance pipeline.
 
This script wires together all the individual modules and runs the
full pipeline for every dataset defined in config.py.
 
Usage
-----
    python run_pipeline.py
 
Outputs (per dataset, saved to the output_prefix defined in config.py)
-------
    <prefix>_feature_importance.png   bar charts: top-30 features + signed coefs
    <prefix>_model_comparison.png     heatmap: top features across all 4 models
    <prefix>_top_features.csv         ranked table: importance + signed coef
 
Pipeline order
--------------
    config.py           → dataset paths, hyperparameters, plot settings
        ↓
    data_loader.py      → load Excel, engineer features, impute, return arrays
        ↓
    models.py           → cross-validate (predefined splits) + fit on full data
        ↓
    results.py          → export R² scores to RESULTS_DIR
    importance.py       → extract importances and signed coefficients
        ↓
    visualise.py        → bar charts and heatmap
    importance.py       → CSV export
"""
 
import warnings
warnings.filterwarnings("ignore")
 
from config         import DATASETS
from data_loader    import load_and_prepare
from models         import fit_models
from model_results  import save_r2_results
from importance     import get_importances, save_importance_csv
from visualise      import plot_top_features, plot_model_comparison
 
 
def run_dataset(path, te_col, label, out_prefix):
    """Run the full pipeline for a single dataset."""
    print(f"\n{'#'*60}")
    print(f"  Running: {label}")
    print(f"{'#'*60}")
 
    # 1. load and engineer features
    X, y, folds, feature_names = load_and_prepare(path, te_col)
 
    # 2. cross-validate and fit all four models
    fitted_models, cv_scores  = fit_models(X, y, folds, feature_names, label)
 
     # 3. export R^2 results to the results folder
    save_r2_results(cv_scores, label)
 
    # 4. extract importances from fitted models
    # X and y are passed through for SVR permutation importance
    importances = get_importances(fitted_models, feature_names, X, y)
 
    # 5. visualise
    plot_top_features(
        importances, fitted_models, feature_names,
        label, out_prefix
    )
    plot_model_comparison(importances, label, out_prefix)
 
    # 6. save CSV table
    save_importance_csv(
        importances, fitted_models, feature_names,
        label, out_prefix
    )
 
    print(f"\n   {label} complete — outputs saved to: {out_prefix}_*")
 
 
def main():
    for path, te_col, label, out_prefix in DATASETS:
        run_dataset(path, te_col, label, out_prefix)
    print("\n\nAll datasets complete.")
 
 
if __name__ == "__main__":
    main()