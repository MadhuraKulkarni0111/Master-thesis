"""
Translation Efficiency (TE) Feature Importance Pipeline
========================================================
Models: Lasso, ElasticNet, Random Forest
Targets: TE_HCT116 (human) | TE_4T1 (mouse)

Features engineered per transcript:
  - Region sizes (utr5, cds, utr3)
  - GC content per region (utr5, cds, utr3, full tx)
  - Mononucleotide frequencies (A/T/G/C) per region  -> 12 features (4*3)
  - Dinucleotide frequencies (16) per region          -> 48 features (4*2*3)
  - Codon usage frequencies (61 sense codons)         ->  from CDS 
  - uAUG count in 5'UTR
  - log-transformed sizes
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# ── Codon table ──────────────────────────────────────────────────────────────
STOP_CODONS = {"TAA", "TAG", "TGA"}
ALL_CODONS = [
    a + b + c
    for a, b, c in product("ACGT", repeat=3)
    if (a + b + c) not in STOP_CODONS
]  # 61 sense codons

# ── Sequence helpers ──────────────────────────────────────────────────────────

def extract_regions(row):
    """Return (utr5, cds, utr3) subsequences from full tx_sequence."""
    seq = str(row["tx_sequence"]).upper().replace("U", "T")
    u5  = int(row["utr5_size"]) if pd.notna(row["utr5_size"]) else 0
    cds = int(row["cds_size"])  if pd.notna(row["cds_size"])  else 0
    return seq[:u5], seq[u5 : u5 + cds], seq[u5 + cds :]


def gc_content(seq):
    if not seq:
        return np.nan
    return (seq.count("G") + seq.count("C")) / len(seq)


def mono_freq(seq, label):
    """Return dict of {label_A: freq, ...} for A/T/G/C."""
    n = len(seq)
    if n == 0:
        return {f"{label}_{nt}": np.nan for nt in "ATGC"}
    return {f"{label}_{nt}": seq.count(nt) / n for nt in "ATGC"}


def di_freq(seq, label):
    """Return dict of 16 dinucleotide frequencies."""
    dinucs = [a + b for a, b in product("ATGC", repeat=2)]
    n = len(seq) - 1
    if n <= 0:
        return {f"{label}_{d}": np.nan for d in dinucs}
    return {f"{label}_{d}": seq.count(d) / n for d in dinucs}


def codon_freq(cds_seq):
    """Return dict of 61 sense-codon frequencies within CDS."""
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
        result[f"codon_{c}"] = result.get(f"codon_{c}", 0) + 1
    return {k: v / total for k, v in result.items()}


def uaug_count(utr5_seq):
    """Count upstream AUG codons in 5'UTR."""
    return utr5_seq.count("ATG")


def build_features(df):
    """Engineer all features for a dataframe, return feature DataFrame."""
    records = []
    for _, row in df.iterrows():
        utr5, cds, utr3 = extract_regions(row)
        full = utr5 + cds + utr3

        feat = {}
        # --- sizes (raw + log) ---
        feat["utr5_size"]    = len(utr5)
        feat["cds_size"]     = len(cds)
        feat["utr3_size"]    = len(utr3)
        feat["log_utr5"]     = np.log1p(len(utr5))
        feat["log_cds"]      = np.log1p(len(cds))
        feat["log_utr3"]     = np.log1p(len(utr3))
        feat["log_tx"]       = np.log1p(len(full))

        # --- GC content ---
        feat["gc_utr5"] = gc_content(utr5)
        feat["gc_cds"]  = gc_content(cds)
        feat["gc_utr3"] = gc_content(utr3)
        feat["gc_full"] = gc_content(full)

        # --- uAUG ---
        feat["uAUG_count"] = uaug_count(utr5)

        # --- mononucleotide frequencies ---
        feat.update(mono_freq(utr5, "utr5"))
        feat.update(mono_freq(cds,  "cds"))
        feat.update(mono_freq(utr3, "utr3"))

        # --- dinucleotide frequencies ---
        feat.update(di_freq(utr5, "utr5"))
        feat.update(di_freq(cds,  "cds"))
        feat.update(di_freq(utr3, "utr3"))

        # --- codon usage ---
        feat.update(codon_freq(cds))

        records.append(feat)

    return pd.DataFrame(records, index=df.index)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_prepare(path, te_col):
    print(f"\nLoading {path}  →  target: {te_col}")
    df = pd.read_excel(path)
    df = df.dropna(subset=[te_col, "tx_sequence"])
    print(f"  Rows after dropping NA in target/sequence: {len(df)}")
    X = build_features(df)
    y = df[te_col].values
    feature_names = X.columns.tolist()

    # impute any remaining NaNs (e.g. empty regions → nan freq)
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X)
    return X_arr, y, feature_names


# ── Model fitting & CV ────────────────────────────────────────────────────────

def fit_models(X, y, feature_names, label, out_prefix):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()

    models = {
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
            n_estimators=300, max_features="sqrt",
            n_jobs=-1, random_state=42
        ),
    }

    results = {}
    print(f"\n{'='*55}")
    print(f"  Dataset: {label}")
    print(f"  Features: {X.shape[1]}   Samples: {X.shape[0]}")
    print(f"{'='*55}")

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="r2", n_jobs=-1)
        print(f"  {name:<14}  R² = {scores.mean():.4f} ± {scores.std():.4f}")
        model.fit(X, y)
        results[name] = model

    return results


# ── Feature importance extraction ────────────────────────────────────────────

def get_importances(models, X, y, feature_names):
    imp = {}

    # Lasso coefficients
    lasso_coef = models["Lasso"].named_steps["model"].coef_
    imp["Lasso"] = pd.Series(np.abs(lasso_coef), index=feature_names)

    # ElasticNet coefficients
    en_coef = models["ElasticNet"].named_steps["model"].coef_
    imp["ElasticNet"] = pd.Series(np.abs(en_coef), index=feature_names)

    # RandomForest feature importance
    rf_imp = models["RandomForest"].feature_importances_
    imp["RandomForest"] = pd.Series(rf_imp, index=feature_names)

    return imp


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "Lasso":        "#4E79A7",
    "ElasticNet":   "#F28E2B",
    "RandomForest": "#59A14F",
}

def plot_top_features(imp_dict, models, feature_names, label,
                      out_prefix, top_n=30):
    """
    Two-panel plot per model:
      Left  – top-N feature importances (|coef| or MDI)
      Right – raw signed coefficients for linear models (RF: same as left)
    """
    fig_path = f"{out_prefix}_feature_importance.png"
    n_models = len(imp_dict)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(22, 7 * n_models),
                             constrained_layout=True)

    fig.suptitle(f"Feature Importance & Coefficients — {label}",
                 fontsize=16, fontweight="bold", y=1.01)

    for row_idx, (mname, imp_series) in enumerate(imp_dict.items()):
        color = COLORS[mname]
        top = imp_series.nlargest(top_n)

        # ---- Left: importance magnitude ----
        ax_imp = axes[row_idx, 0]
        bars = ax_imp.barh(range(top_n), top.values[::-1],
                           color=color, alpha=0.85, edgecolor="white")
        ax_imp.set_yticks(range(top_n))
        ax_imp.set_yticklabels(top.index[::-1], fontsize=8)
        ax_imp.set_xlabel("|Coefficient|" if mname != "RandomForest"
                          else "MDI Importance", fontsize=10)
        ax_imp.set_title(f"{mname} — Top {top_n} features", fontsize=12,
                         fontweight="bold", color=color)
        ax_imp.spines[["top", "right"]].set_visible(False)

        # ---- Right: signed coefficients / RF importance ----
        ax_coef = axes[row_idx, 1]

        if mname in ("Lasso", "ElasticNet"):
            signed = pd.Series(
                models[mname].named_steps["model"].coef_,
                index=feature_names
            ).reindex(top.index)
            bar_colors = [
                "#D95F02" if v > 0 else "#1B9E77" for v in signed.values[::-1]
            ]
            ax_coef.barh(range(top_n), signed.values[::-1],
                         color=bar_colors, alpha=0.85, edgecolor="white")
            ax_coef.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax_coef.set_xlabel("Signed Coefficient", fontsize=10)
            ax_coef.set_title(f"{mname} — Signed Coefficients (top {top_n})",
                              fontsize=12, fontweight="bold", color=color)

            # legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#D95F02", label="Positive (↑ TE)"),
                Patch(facecolor="#1B9E77", label="Negative (↓ TE)"),
            ]
            ax_coef.legend(handles=legend_elements, fontsize=9,
                           loc="lower right")
        else:
            # RF: repeat bar chart for the right panel
            ax_coef.barh(range(top_n), top.values[::-1],
                         color=color, alpha=0.85, edgecolor="white")
            ax_coef.set_xlabel("MDI Importance", fontsize=10)
            ax_coef.set_title(
                f"{mname} — Importance (RF has no signed coefs)",
                fontsize=12, fontweight="bold", color=color
            )

        ax_coef.set_yticks(range(top_n))
        ax_coef.set_yticklabels(top.index[::-1], fontsize=8)
        ax_coef.spines[["top", "right"]].set_visible(False)

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")
    return fig_path


def plot_model_comparison(imp_dict, label, out_prefix, top_n=20):
    """Heatmap: top features × models (normalised 0–1)."""
    fig_path = f"{out_prefix}_model_comparison.png"

    # Union of top-N from each model
    top_features = set()
    for s in imp_dict.values():
        top_features.update(s.nlargest(top_n).index.tolist())
    top_features = sorted(top_features)

    df_heat = pd.DataFrame(
        {m: s.reindex(top_features).fillna(0) for m, s in imp_dict.items()}
    )
    # normalise each model column to [0, 1]
    df_heat = df_heat / df_heat.max()

    fig, ax = plt.subplots(figsize=(10, max(8, len(top_features) * 0.35)),
                           constrained_layout=True)
    im = ax.imshow(df_heat.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels(df_heat.columns, fontsize=11)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    plt.colorbar(im, ax=ax, label="Normalised Importance")
    ax.set_title(f"Model Comparison Heatmap — {label}", fontsize=13,
                 fontweight="bold")

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")
    return fig_path


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = [
    ("/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/Human_data.xlsx", "TE_HCT116", "Human_HCT116",
     "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/human_hct116"),
    ("/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/Mouse_data.xlsx", "TE_4T1",    "Mouse_4T1",
     "/Users/madhurakulkarni/Desktop/master_thesis/feature_engineering/mouse_4t1"),
]

all_output_files = []

for path, te_col, label, prefix in DATASETS:
    X, y, feature_names = load_and_prepare(path, te_col)
    fitted_models        = fit_models(X, y, feature_names, label, prefix)
    importances          = get_importances(fitted_models, X, y, feature_names)

    f1 = plot_top_features(importances, fitted_models, feature_names,
                           label, prefix, top_n=30)
    f2 = plot_model_comparison(importances, label, prefix, top_n=25)
    all_output_files += [f1, f2]

    # --- Save top-feature CSV ---
    csv_path = f"{prefix}_top_features.csv"
    rows = []
    for mname, imp in importances.items():
        for feat, val in imp.nlargest(50).items():
            signed = np.nan
            if mname in ("Lasso", "ElasticNet"):
                signed = fitted_models[mname].named_steps["model"].coef_[
                    feature_names.index(feat)
                ]
            rows.append({
                "dataset": label,
                "model": mname,
                "feature": feat,
                "importance": val,
                "signed_coef": signed,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    all_output_files.append(csv_path)

print("\nAll done.")
