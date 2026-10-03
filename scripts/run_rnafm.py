"""
run_rnafm.py
============
Run the normal (non-ablation) pipeline with RNA-FM embeddings as features.

Two feature sets
----------------
    rnafm     : RNA-FM embedding only (640 columns)
    combined  : full hand-crafted feature set (same as A13, 326 columns)
                + RNA-FM embedding (640 columns)

Each uses the same models, CV folds, metrics, importances and plots as
run_pipeline.py — only the feature matrix differs.

Usage
-----
    # 1. Laptop: compute and cache the embeddings only (no model training,
    #    does not need ViennaRNA / sklearn models to be importable)
    python run_rnafm.py --embed-only

    # 2. Run the models (laptop or cluster)
    python run_rnafm.py                               # both datasets, both feature sets
    python run_rnafm.py --dataset human               # one dataset
    python run_rnafm.py --features rnafm              # embedding only

Outputs
-------
    <RESULTS_DIR>/RNAFM/              ← rnafm feature set
    <RESULTS_DIR>/RNAFM_plus_full/    ← combined feature set
each containing, per dataset, the same files as an ablation directory:
    <label>_<set>_r2_results.csv, _eval_results.csv, _top_features.csv,
    _feature_importance.png, _model_comparison.png

Caching
-------
    results/rnafm_cache/<hash>.npy   one vector per sequence (resumable)
    results/feature_cache/<stem>_<species>_<n>_rnafm_<pool>_L<len>.parquet
                                     the assembled per-dataset matrix
To move work from laptop to cluster, copy the whole results/rnafm_cache/
and results/feature_cache/ folders across.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (DATASETS, RESULTS_DIR, MAX_SAMPLES,
                    RNAFM_POOLING, RNAFM_MAX_LEN)

FEATURE_CACHE_DIR = Path(RESULTS_DIR) / "feature_cache"
OUT_DIRS = {"rnafm": "RNAFM", "combined": "RNAFM_plus_full"}


def load_df(path, te_col):
    """Load rows exactly as data_loader.load_and_prepare does, so the row
    order matches the hand-crafted feature matrix for the combined set."""
    df = pd.read_excel(path)
    df = df.dropna(subset=[te_col, "tx_sequence"])
    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)
    return df


def _emb_cache_path(path, species):
    stem = Path(path).stem
    suffix = f"n{MAX_SAMPLES}" if MAX_SAMPLES is not None else "full"
    return FEATURE_CACHE_DIR / (
        f"{stem}_{species}_{suffix}_rnafm_{RNAFM_POOLING}_L{RNAFM_MAX_LEN}.parquet"
    )


def get_embeddings(df, path, species):
    """Return the RNA-FM matrix for df, from the parquet cache if valid."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _emb_cache_path(path, species)

    if cache.exists():
        X_fm = pd.read_parquet(cache)
        if X_fm.index.equals(df.index):
            print(f"  Loaded cached RNA-FM matrix: {cache}")
            return X_fm
        print("  RNA-FM cache index does not match current data — rebuilding.")

    # imported here so the hand-crafted-only parts never need torch
    from rna_fm_features import build_rna_fm_features
    X_fm = build_rna_fm_features(df)
    X_fm.to_parquet(cache)
    print(f"  Cached RNA-FM matrix: {cache}")
    return X_fm


def run_one(path, te_col, label, species, feature_set):
    # imported here so --embed-only works without ViennaRNA installed
    from data_loader   import load_and_prepare
    from models        import fit_models
    from model_results import save_r2_results, save_full_results
    from importance    import get_importances, save_importance_csv
    from visualise     import plot_top_features, plot_model_comparison

    out_dir = str(Path(RESULTS_DIR) / OUT_DIRS[feature_set])
    run_label = f"{label}_{feature_set}"

    print(f"\n{'#'*60}")
    print(f"  Running: {label}  |  Features: {feature_set}")
    print(f"  Output dir: {out_dir}")
    print(f"{'#'*60}")

    df = load_df(path, te_col)
    X_fm = get_embeddings(df, path, species)

    if feature_set == "rnafm":
        X = X_fm.values
        y = df[te_col].values
        folds = df["fold"].values
        feature_names = list(X_fm.columns)
    else:
        X_hc, y, folds, hc_names = load_and_prepare(path, te_col, species,
                                                    ablation_spec=None)
        if X_hc.shape[0] != X_fm.shape[0]:
            raise RuntimeError(
                f"Row mismatch: hand-crafted {X_hc.shape[0]} vs "
                f"RNA-FM {X_fm.shape[0]} — delete both caches and rebuild."
            )
        X = np.hstack([X_hc, X_fm.values])
        feature_names = hc_names + list(X_fm.columns)

    fitted_models, cv_scores = fit_models(X, y, folds, run_label)

    save_r2_results(cv_scores, run_label, out_dir)
    save_full_results(cv_scores, y, folds, run_label, out_dir)

    importances = get_importances(fitted_models, feature_names, X, y)
    plot_top_features(importances, fitted_models, feature_names, run_label, out_dir)
    plot_model_comparison(importances, run_label, out_dir)
    save_importance_csv(importances, fitted_models, feature_names, run_label, out_dir)

    print(f"\n   {run_label} complete — outputs saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run the TE pipeline with RNA-FM features.")
    parser.add_argument("--dataset", choices=["human", "mouse"], default=None,
                        help="Default: both datasets.")
    parser.add_argument("--features", choices=["rnafm", "combined", "both"],
                        default="both",
                        help="rnafm = embedding only; combined = hand-crafted + embedding.")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only compute and cache embeddings, then exit.")
    args = parser.parse_args()

    datasets = [d for d in DATASETS if args.dataset is None or d[4] == args.dataset]

    if args.embed_only:
        for path, te_col, label, _, species in datasets:
            print(f"\nEmbedding {label} ({species})...")
            get_embeddings(load_df(path, te_col), path, species)
        print("\nAll RNA-FM embeddings cached.")
        return

    feature_sets = ["rnafm", "combined"] if args.features == "both" else [args.features]
    for path, te_col, label, _, species in datasets:
        for fs in feature_sets:
            run_one(path, te_col, label, species, fs)

    print("\n\nAll RNA-FM runs complete.")


if __name__ == "__main__":
    main()