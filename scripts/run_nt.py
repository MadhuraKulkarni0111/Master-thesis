"""
run_nt.py
=========
Run the normal (non-ablation) pipeline with Nucleotide Transformer v2
embeddings as features. Same structure as run_rnafm.py.

Two feature sets
----------------
    nt        : NT-v2 embedding only
    combined  : full hand-crafted feature set (same as A13) + NT-v2 embedding

Same models, CV folds, metrics, importances and plots as run_pipeline.py;
only the feature matrix differs.

Usage
-----
    # once, on the login node (internet needed):
    python download_nt_model.py

    # on a GPU: compute and cache the embeddings only
    python run_nt.py --embed-only [--dataset human]

    # models (CPU is enough once embeddings are cached)
    python run_nt.py [--dataset human] [--features nt|combined|both]

Outputs
-------
    <RESULTS_DIR>/NT/               <- nt feature set
    <RESULTS_DIR>/NT_plus_full/     <- combined feature set
Files per dataset (<label> = e.g. Human_HCT116_nt):
    <label>_r2_results.csv, _eval_results.csv, _top_features.csv,
    _feature_importance.png, _model_comparison.png
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (DATASETS, RESULTS_DIR, MAX_SAMPLES,
                    NT_MODEL_NAME, NT_POOLING, NT_MAX_TOKENS)

FEATURE_CACHE_DIR = Path(RESULTS_DIR) / "feature_cache"
OUT_DIRS = {"nt": "NT", "combined": "NT_plus_full"}
RUN_SUFFIX = {"nt": "nt", "combined": "nt_plus_full"}


def load_df(path, te_col):
    """Load rows exactly as data_loader.load_and_prepare does, so row order
    matches the hand-crafted feature matrix for the combined set."""
    df = pd.read_excel(path)
    df = df.dropna(subset=[te_col, "tx_sequence"])
    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)
    return df


def _emb_cache_path(path, species):
    stem = Path(path).stem
    suffix = f"n{MAX_SAMPLES}" if MAX_SAMPLES is not None else "full"
    model_tag = NT_MODEL_NAME.split("/")[-1]
    return FEATURE_CACHE_DIR / (
        f"{stem}_{species}_{suffix}_nt_{model_tag}_{NT_POOLING}_T{NT_MAX_TOKENS}.parquet"
    )


def get_embeddings(df, path, species):
    """Return the NT-v2 matrix for df, from the parquet cache if valid."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _emb_cache_path(path, species)

    if cache.exists():
        X_nt = pd.read_parquet(cache)
        if X_nt.index.equals(df.index):
            print(f"  Loaded cached NT-v2 matrix: {cache}")
            return X_nt
        print("  NT-v2 cache index does not match current data — rebuilding.")

    # imported here so the hand-crafted-only parts never need transformers
    from nt_features import build_nt_features
    X_nt = build_nt_features(df)
    X_nt.to_parquet(cache)
    print(f"  Cached NT-v2 matrix: {cache}")
    return X_nt


def run_one(path, te_col, label, species, feature_set):
    from data_loader   import load_and_prepare
    from models        import fit_models
    from model_results import save_r2_results, save_full_results
    from importance    import get_importances, save_importance_csv
    from visualise     import plot_top_features, plot_model_comparison

    out_dir = str(Path(RESULTS_DIR) / OUT_DIRS[feature_set])
    run_label = f"{label}_{RUN_SUFFIX[feature_set]}"

    print(f"\n{'#'*60}")
    print(f"  Running: {label}  |  Features: {feature_set}  |  {NT_MODEL_NAME}")
    print(f"  Output dir: {out_dir}")
    print(f"{'#'*60}")

    df = load_df(path, te_col)
    X_nt = get_embeddings(df, path, species)

    if feature_set == "nt":
        X = X_nt.values
        y = df[te_col].values
        folds = df["fold"].values
        feature_names = list(X_nt.columns)
    else:
        X_hc, y, folds, hc_names = load_and_prepare(path, te_col, species,
                                                    ablation_spec=None)
        if X_hc.shape[0] != X_nt.shape[0]:
            raise RuntimeError(
                f"Row mismatch: hand-crafted {X_hc.shape[0]} vs "
                f"NT {X_nt.shape[0]} — delete both caches and rebuild."
            )
        X = np.hstack([X_hc, X_nt.values])
        feature_names = hc_names + list(X_nt.columns)

    fitted_models, cv_scores = fit_models(X, y, folds, run_label)

    save_r2_results(cv_scores, run_label, out_dir)
    save_full_results(cv_scores, y, folds, run_label, out_dir)

    importances = get_importances(fitted_models, feature_names, X, y)
    plot_top_features(importances, fitted_models, feature_names, run_label, out_dir)
    plot_model_comparison(importances, run_label, out_dir)
    save_importance_csv(importances, fitted_models, feature_names, run_label, out_dir)

    print(f"\n   {run_label} complete — outputs saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run the TE pipeline with NT-v2 features.")
    parser.add_argument("--dataset", choices=["human", "mouse"], default=None,
                        help="Default: both datasets.")
    parser.add_argument("--features", choices=["nt", "combined", "both"], default="both",
                        help="nt = embedding only; combined = hand-crafted + embedding.")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only compute and cache embeddings, then exit.")
    args = parser.parse_args()

    datasets = [d for d in DATASETS if args.dataset is None or d[4] == args.dataset]

    if args.embed_only:
        for path, te_col, label, _, species in datasets:
            print(f"\nEmbedding {label} ({species}) with {NT_MODEL_NAME}...")
            get_embeddings(load_df(path, te_col), path, species)
        print("\nAll NT-v2 embeddings cached.")
        return

    feature_sets = ["nt", "combined"] if args.features == "both" else [args.features]
    for path, te_col, label, _, species in datasets:
        for fs in feature_sets:
            run_one(path, te_col, label, species, fs)

    print("\n\nAll NT-v2 runs complete.")


if __name__ == "__main__":
    main()
