"""
run_pipeline.py
===============
Main entry point for the TE feature importance pipeline.

Wires together all the individual modules and runs the pipeline for a
given ablation (feature-group subset, see ablations.py) on a given
dataset (see config.py DATASETS).

Usage
-----
    # Run everything: every ablation x every dataset (slow — use the cluster)
    python run_pipeline.py

    # Run a single ablation on a single dataset
    python run_pipeline.py --ablation A7 --dataset human

    # Run a single ablation on both datasets
    python run_pipeline.py --ablation A7

    # Just build (and cache) the full feature matrix for every dataset,
    # then exit. Run this once, serially, before submitting a SLURM array
    # so all array tasks hit a warm cache instead of racing to build it.
    python run_pipeline.py --cache-only

Outputs
-------
Each ablation gets its own directory under RESULTS_DIR:

    <RESULTS_DIR>/A7/
        Human_HCT116_feature_importance.png
        Human_HCT116_model_comparison.png
        Human_HCT116_top_features.csv
        Human_HCT116_r2_results.csv
        Mouse_4T1_feature_importance.png
        Mouse_4T1_model_comparison.png
        Mouse_4T1_top_features.csv
        Mouse_4T1_r2_results.csv

So `results/A7/` has everything for that ablation, across both datasets,
and nothing from any other ablation collides with it.

Pipeline order
--------------
    config.py + ablations.py → dataset paths, hyperparameters, feature groups
        v
    data_loader.py      → load Excel, engineer/cache features, slice groups,
                          impute, return arrays
        v
    models.py           → cross-validate (predefined splits) + fit on full data
        v
    model_results.py    → export R² scores to <out_dir>
    importance.py       → extract importances and signed coefficients
        v
    visualise.py        → bar charts and heatmap, saved to <out_dir>
    importance.py       → CSV export, saved to <out_dir>
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from config         import DATASETS, RESULTS_DIR
from ablations      import ABLATIONS
from data_loader    import load_and_prepare
from models         import fit_models
from model_results  import save_r2_results
from importance     import get_importances, save_importance_csv
from visualise      import plot_top_features, plot_model_comparison


def run_dataset(path, te_col, label, species, ablation_id, feature_groups):
    """Run the full pipeline for a single (dataset, ablation) combination."""
    out_dir = str(Path(RESULTS_DIR) / ablation_id)

    print(f"\n{'#'*60}")
    print(f"  Running: {label}  |  Ablation: {ablation_id}  |  groups: {feature_groups}")
    print(f"  Output dir: {out_dir}")
    print(f"{'#'*60}")

    # 1. load cached features, sliced to this ablation's feature groups
    X, y, folds, feature_names = load_and_prepare(path, te_col, species, feature_groups)

    # 2. cross-validate and fit all models
    fitted_models, cv_scores = fit_models(X, y, folds, label)

    # 3. export R² results
    save_r2_results(cv_scores, label, out_dir)

    # 4. extract importances from fitted models
    importances = get_importances(fitted_models, feature_names, X, y)

    # 5. visualise
    plot_top_features(importances, fitted_models, feature_names, label, out_dir)
    plot_model_comparison(importances, label, out_dir)

    # 6. save CSV table
    save_importance_csv(importances, fitted_models, feature_names, label, out_dir)

    print(f"\n   {label} / {ablation_id} complete — outputs saved to: {out_dir}/")


def cache_only():
    """Build (and cache) the full feature matrix for every dataset, then exit."""
    for path, te_col, label, _, species in DATASETS:
        print(f"\nBuilding full feature cache for {label} ({species})...")
        load_and_prepare(path, te_col, species, feature_groups=None)
    print("\nAll feature caches built.")


def main():
    parser = argparse.ArgumentParser(description="Run the TE ablation pipeline.")
    parser.add_argument(
        "--ablation", choices=list(ABLATIONS.keys()), default=None,
        help="Which ablation to run (e.g. A7). Default: run all ablations."
    )
    parser.add_argument(
        "--dataset", choices=["human", "mouse"], default=None,
        help="Which dataset to run (matches the species field in config.DATASETS). "
             "Default: run all datasets."
    )
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Only build and cache the full feature matrix for each dataset, then exit. "
             "Run this once before a SLURM array job so tasks don't race to build the cache."
    )
    args = parser.parse_args()

    if args.cache_only:
        cache_only()
        return

    ablation_ids = [args.ablation] if args.ablation else list(ABLATIONS.keys())
    datasets = [d for d in DATASETS if args.dataset is None or d[4] == args.dataset]

    for path, te_col, label, _, species in datasets:
        for aid in ablation_ids:
            run_dataset(path, te_col, label, species, aid, ABLATIONS[aid])

    print("\n\nAll requested runs complete.")


if __name__ == "__main__":
    main()
