"""
Translation Efficiency (TE) Feature Importance Pipeline
========================================================
Models: Lasso, ElasticNet, Random Forest, LightGBM
Targets: TE_HCT116 (human) | TE_4T1 (mouse)

Features engineered per transcript:
  - Region sizes (utr5, cds, utr3) + log-transformed
  - GC content per region (utr5, cds, utr3, full tx)
  - Mononucleotide frequencies (A/T/G/C) per region  -> 12 features
  - Dinucleotide frequencies (16) per region          -> 48 features
  - Codon usage frequencies (61 sense codons)            from CDS
  - uAUG count in 5'UTR

"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# ── Codon table ───────────────────────────────────────────────────────────────
STOP_CODONS = {"TAA", "TAG", "TGA"}
ALL_CODONS  = [
    a + b + c
    for a, b, c in product("ACGT", repeat=3)
    if (a + b + c) not in STOP_CODONS
]  # 61 sense codons


# ── Sequence helpers ──────────────────────────────────────────────────────────

def extract_regions(row):
    """Slice full tx_sequence into (utr5, cds, utr3) strings."""
    seq = str(row["tx_sequence"]).upper().replace("U", "T")
    u5  = int(row["utr5_size"]) if pd.notna(row["utr5_size"]) else 0
    cds = int(row["cds_size"])  if pd.notna(row["cds_size"])  else 0
    return seq[:u5], seq[u5 : u5 + cds], seq[u5 + cds:]


def gc_content(seq):
    if not seq:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def mono_freq(seq, label):
    """Fraction of each nucleotide (A/T/G/C) in a region."""
    n = len(seq)
    if n == 0:
        return {f"{label}_{nt}": np.nan for nt in "ATGC"}
    return {f"{label}_{nt}": seq.count(nt) / n for nt in "ATGC"}


def di_freq(seq, label):
    """Fraction of each of the 16 dinucleotides in a region."""
    dinucs = [a + b for a, b in product("ATGC", repeat=2)]
    n = len(seq) - 1
    if n <= 0:
        return {f"{label}_{d}": np.nan for d in dinucs}
    return {f"{label}_{d}": seq.count(d) / n for d in dinucs}


def codon_freq(cds_seq):
    """Relative frequency of each of the 61 sense codons in the CDS."""
    result = {f"codon_{c}": 0.0 for c in ALL_CODONS}
    codons_found = []
    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i : i + 3]
        if len(codon) == 3 and codon not in STOP_CODONS and "N" not in codon:
            codons_found.append(codon)
    total = len(codons_found)
    if total == 0:
        return {k: np.nan for k in result}
    for c in codons_found:
        result[f"codon_{c}"] += 1
    return {k: v / total for k, v in result.items()}


def uaug_count(utr5_seq):
    """Number of upstream ATG codons in the 5'UTR."""
    return utr5_seq.count("ATG")


def build_features(df):
    """Engineer all features for every row; returns a feature DataFrame."""
    records = []
    for _, row in df.iterrows():
        utr5, cds, utr3 = extract_regions(row)
        full = utr5 + cds + utr3

        feat = {}

        # sizes — raw and log-transformed
        feat["log_utr5"]  = np.log1p(len(utr5))
        feat["log_cds"]   = np.log1p(len(cds))
        feat["log_utr3"]  = np.log1p(len(utr3))

        # GC content per region
        feat["gc_utr5"] = gc_content(utr5)
        feat["gc_cds"]  = gc_content(cds)
        feat["gc_utr3"] = gc_content(utr3)
        feat["gc_full"] = gc_content(full)

        # upstream AUG count
        feat["uAUG_count"] = uaug_count(utr5) # maybe remove this as it willbe codedin the codon-freq

        # mononucleotide frequencies (4 × 3 regions = 12)
        feat.update(mono_freq(utr5, "utr5"))
        feat.update(mono_freq(cds,  "cds"))
        feat.update(mono_freq(utr3, "utr3"))

        # dinucleotide frequencies (16 × 3 regions = 48)
        feat.update(di_freq(utr5, "utr5"))
        feat.update(di_freq(cds,  "cds"))
        feat.update(di_freq(utr3, "utr3"))

        # codon usage (61 sense codons from CDS)
        feat.update(codon_freq(cds))

        records.append(feat)

    return pd.DataFrame(records, index=df.index)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_prepare(path, te_col):
    print(f"\nLoading {path}  →  target: {te_col}")
    df = pd.read_excel(path)
    df = df.dropna(subset=[te_col, "tx_sequence"])
    print(f"  Rows after dropping NA in target/sequence: {len(df)}")

    folds = df["fold"].values          # pre-defined 10-fold splits

    X             = build_features(df)
    y             = df[te_col].values
    feature_names = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)
    return X_arr, y, folds, feature_names


# ── Predefined CV splits ──────────────────────────────────────────────────────

def make_predefined_splits(folds):
    """
    Convert the dataset fold column into (train_idx, test_idx) tuples.
    Each unique fold is held out once; all others form the training set.
    """
    splits = []
    for held_out in sorted(np.unique(folds)):
        train_idx = np.where(folds != held_out)[0]
        test_idx  = np.where(folds == held_out)[0]
        splits.append((train_idx, test_idx))
    return splits


# ── Model definitions ─────────────────────────────────────────────────────────

def build_models():
    """
    Return a dict of model name → unfitted model.

    LightGBM notes
    ──────────────
    LightGBM builds an ensemble of gradient-boosted decision trees, but
    unlike Random Forest it trains them *sequentially*: each new tree
    corrects the residual errors left by all previous trees.

    Key hyperparameters chosen here:
      n_estimators   = 1000   – maximum number of trees; early stopping
                                will halt before this if validation loss
                                stops improving (not used in CV here, but
                                good practice to set high).
      learning_rate  = 0.05   – how much each tree contributes; lower
                                values need more trees but generalise
                                better.
      num_leaves     = 63     – maximum leaves per tree; controls model
                                complexity. Rule of thumb: 2^(max_depth)-1.
      min_child_samples = 20  – minimum genes per leaf; prevents overfitting
                                on rare codon/dinucleotide combinations.
      subsample      = 0.8    – fraction of training genes sampled per tree
                                (row subsampling, like RF's bagging).
      colsample_bytree=0.8    – fraction of features sampled per tree
                                (column subsampling).
      importance_type="gain"  – use total information gain rather than
                                split count as the feature importance
                                metric; gain is more informative because
                                it weights splits by how much they reduce
                                prediction error, not just how often a
                                feature is used.
    """
    return {
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.001, max_iter=10000, random_state=42))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(alpha=0.001, l1_ratio=0.5,
                                  max_iter=10000, random_state=42))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            importance_type="gain",   # gain > split for feature importance
            n_jobs=-1,
            random_state=42,
            verbose=-1                # suppress LightGBM training logs
        ),
    }


# ── Model fitting & CV ────────────────────────────────────────────────────────

def fit_models(X, y, folds, feature_names, label):
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
        scores = cross_val_score(
            model, X, y,
            cv=cv_splits,
            scoring="r2",
            n_jobs=-1
        )
        print(f"  {name:<14}  R² = {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  {'':14}  per fold: "
              f"{', '.join(f'{s:.3f}' for s in scores)}")
        model.fit(X, y)
        fitted[name] = model

    return fitted


# ── Feature importance extraction ─────────────────────────────────────────────

def get_importances(fitted_models, feature_names):
    """
    Extract feature importances for all four models.
    """
    imp = {}

    imp["Lasso"] = pd.Series(
        np.abs(fitted_models["Lasso"].named_steps["model"].coef_),
        index=feature_names
    )
    imp["ElasticNet"] = pd.Series(
        np.abs(fitted_models["ElasticNet"].named_steps["model"].coef_),
        index=feature_names
    )
    imp["RandomForest"] = pd.Series(
        fitted_models["RandomForest"].feature_importances_,
        index=feature_names
    )
    imp["LightGBM"] = pd.Series(
        fitted_models["LightGBM"].feature_importances_,   
        index=feature_names
    )

    return imp


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "Lasso":        "#4E79A7",
    "ElasticNet":   "#F28E2B",
    "RandomForest": "#59A14F",
    "LightGBM":     "#B07AA1",   
}


def plot_top_features(imp_dict, fitted_models, feature_names,
                      label, out_prefix, top_n=30):
    """
    Two-panel figure per model:
      Left  – horizontal bar chart of top-N feature importances (magnitude)
      Right – signed coefficients for linear models; for tree models the
              same importance bars are shown (trees have no signed coefs)
    """
    fig_path = f"{out_prefix}_feature_importance.png"
    n_models  = len(imp_dict)

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(22, 7 * n_models),
        constrained_layout=True
    )
    fig.suptitle(
        f"Feature Importance & Coefficients — {label}",
        fontsize=16, fontweight="bold", y=1.01
    )

    for row_idx, (mname, imp_series) in enumerate(imp_dict.items()):
        color = COLORS[mname]
        top   = imp_series.nlargest(top_n)

        # ── Left panel: importance magnitude ──────────────────────────────
        ax_imp = axes[row_idx, 0]
        ax_imp.barh(range(top_n), top.values[::-1],
                    color=color, alpha=0.85, edgecolor="white")
        ax_imp.set_yticks(range(top_n))
        ax_imp.set_yticklabels(top.index[::-1], fontsize=8)
        ax_imp.set_xlabel(
            "|Coefficient|" if mname in ("Lasso", "ElasticNet")
            else "Gain Importance" if mname == "LightGBM"
            else "MDI Importance",
            fontsize=10
        )
        ax_imp.set_title(f"{mname} — Top {top_n} features",
                         fontsize=12, fontweight="bold", color=color)
        ax_imp.spines[["top", "right"]].set_visible(False)

        # ── Right panel: signed coefficients or repeated importance ───────
        ax_coef = axes[row_idx, 1]

        if mname in ("Lasso", "ElasticNet"):
            # linear models: show signed coefficients with direction colours
            signed = pd.Series(
                fitted_models[mname].named_steps["model"].coef_,
                index=feature_names
            ).reindex(top.index)
            bar_colors = [
                "#D95F02" if v > 0 else "#1B9E77"
                for v in signed.values[::-1]
            ]
            ax_coef.barh(range(top_n), signed.values[::-1],
                         color=bar_colors, alpha=0.85, edgecolor="white")
            ax_coef.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax_coef.set_xlabel("Signed Coefficient", fontsize=10)
            ax_coef.set_title(f"{mname} — Signed Coefficients (top {top_n})",
                              fontsize=12, fontweight="bold", color=color)
            from matplotlib.patches import Patch
            ax_coef.legend(handles=[
                Patch(facecolor="#D95F02", label="Positive (↑ TE)"),
                Patch(facecolor="#1B9E77", label="Negative (↓ TE)"),
            ], fontsize=9, loc="lower right")

        else:
            # tree models: no signed coefficients — repeat importance bar
            ax_coef.barh(range(top_n), top.values[::-1],
                         color=color, alpha=0.85, edgecolor="white")
            xlabel = ("Gain Importance" if mname == "LightGBM"
                      else "MDI Importance")
            ax_coef.set_xlabel(xlabel, fontsize=10)
            note = ("gain-weighted" if mname == "LightGBM"
                    else "mean decrease in impurity")
            ax_coef.set_title(
                f"{mname} — Importance ({note})\n"
                f"[tree models have no signed coefficients]",
                fontsize=11, fontweight="bold", color=color
            )

        ax_coef.set_yticks(range(top_n))
        ax_coef.set_yticklabels(top.index[::-1], fontsize=8)
        ax_coef.spines[["top", "right"]].set_visible(False)

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")
    return fig_path


def plot_model_comparison(imp_dict, label, out_prefix, top_n=25):
    """
    Heatmap: union of top-N features from each model × all models.
    Each column is normalised 0–1 so models with different importance
    scales can be compared side by side.
    """
    fig_path = f"{out_prefix}_model_comparison.png"

    top_features = set()
    for s in imp_dict.values():
        top_features.update(s.nlargest(top_n).index.tolist())
    top_features = sorted(top_features)

    df_heat = pd.DataFrame(
        {m: s.reindex(top_features).fillna(0) for m, s in imp_dict.items()}
    )
    df_heat = df_heat / df_heat.max()   # normalise each model to [0, 1]

    fig, ax = plt.subplots(
        figsize=(11, max(8, len(top_features) * 0.35)),
        constrained_layout=True
    )
    im = ax.imshow(df_heat.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels(df_heat.columns, fontsize=11)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    plt.colorbar(im, ax=ax, label="Normalised Importance (0–1 per model)")
    ax.set_title(
        f"Model Comparison Heatmap — {label}\n"
        f"(features ranked in top {top_n} of at least one model)",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")
    return fig_path


def save_importance_csv(imp_dict, fitted_models, feature_names,
                        label, out_prefix):
    """Save top-50 features per model with importance + signed coef to CSV."""
    csv_path = f"{out_prefix}_top_features.csv"
    rows = []
    for mname, imp in imp_dict.items():
        for feat, val in imp.nlargest(50).items():
            signed = np.nan
            if mname in ("Lasso", "ElasticNet"):
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


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = [
    (
        "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/Human_data.xlsx",
        "TE_HCT116", "Human_HCT116",
        "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/human_hct116"
    ),
    (
        "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/Mouse_data.xlsx",
        "TE_4T1", "Mouse_4T1",
        "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/mouse_4t1"
    ),
]

for path, te_col, label, prefix in DATASETS:
    X, y, folds, feature_names = load_and_prepare(path, te_col)
    fitted_models               = fit_models(X, y, folds, feature_names, label)
    importances                 = get_importances(fitted_models, feature_names)

    plot_top_features(importances, fitted_models, feature_names,
                      label, prefix, top_n=30)
    plot_model_comparison(importances, label, prefix, top_n=25)
    save_importance_csv(importances, fitted_models, feature_names,
                        label, prefix)

print("\nAll done.")