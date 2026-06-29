"""
visualise.py
============
All plotting functions for the TE feature importance pipeline.

Functions
---------
plot_top_features(imp_dict, fitted_models, feature_names, label,
                  out_prefix, top_n)
    Two-panel bar charts per model: importance magnitude + signed coefficients.
    Saved as <out_prefix>_feature_importance.png

plot_model_comparison(imp_dict, label, out_prefix, top_n)
    Heatmap comparing top features across all models side by side.
    Saved as <out_prefix>_model_comparison.png
"""
print("visualising)")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap

from config import MODEL_COLORS, TOP_N_BAR, TOP_N_HEATMAP


def plot_top_features(imp_dict, fitted_models, feature_names,
                      label, out_prefix, top_n=TOP_N_BAR):
    """
    Create a two-panel horizontal bar chart figure for every model.

    Left panel — importance magnitude
        Shows the top-N features ranked by their importance score.
        For Lasso/ElasticNet this is |coefficient|.
        For RandomForest this is MDI (mean decrease in impurity).
        For LightGBM anf XGboost this is total gain across all splits.

    Right panel — direction
        For Lasso and ElasticNet: signed coefficients coloured by direction.
            Orange bar = positive coefficient → higher feature value raises TE
            Green bar  = negative coefficient → higher feature value lowers TE
        For RandomForest and LightGBM: tree models do not produce signed
            coefficients, so the importance bar chart is repeated with a
            note explaining this.

    One row of panels per model → figure height scales with number of models.

    Parameters
    ----------
    imp_dict      : dict  {model_name: pd.Series of importances}
    fitted_models : dict  {model_name: fitted model}
    feature_names : list of str
    label         : str   dataset label for the figure title
    out_prefix    : str   path prefix; PNG saved as <out_prefix>_feature_importance.png
    top_n         : int   number of features to show (default from config)

    Returns
    -------
    str : path to the saved PNG
    """
    safe_label = label.replace(" ", "_")
    fig_path = f"{out_prefix}_{safe_label}_feature_importance.png"
    #fig_path = f"{out_prefix}_feature_importance.png"
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
        color = MODEL_COLORS[mname]
        top   = imp_series.nlargest(top_n)

        # ── Left panel: importance magnitude ──────────────────────────────
        ax_imp = axes[row_idx, 0]
        ax_imp.barh(
            range(top_n), top.values[::-1],
            color=color, alpha=0.85, edgecolor="white"
        )
        ax_imp.set_yticks(range(top_n))
        ax_imp.set_yticklabels(top.index[::-1], fontsize=8)
        ax_imp.set_xlabel(
            "|Coefficient|" if mname in ("Lasso", "ElasticNet") else
            "Gain Importance" if mname in ("LightGBM", "XGBoost") else
            "MDI Importance", 
            fontsize=10
            )
        ax_imp.set_title(
            f"{mname} — Top {top_n} features",
            fontsize=12, fontweight="bold", color=color
        )
        ax_imp.spines[["top", "right"]].set_visible(False)

        # ── Right panel: signed coefficients or repeated importance ───────
        ax_coef = axes[row_idx, 1]

        if mname in ("Lasso", "ElasticNet"):
            # retrieve raw signed coefficients from inside the Pipeline
            signed = pd.Series(
                fitted_models[mname].named_steps["model"].coef_,
                index=feature_names
            ).reindex(top.index)

            bar_colors = [
                "#D95F02" if v > 0 else "#1B9E77"
                for v in signed.values[::-1]
            ]
            ax_coef.barh(
                range(top_n), signed.values[::-1],
                color=bar_colors, alpha=0.85, edgecolor="white"
            )
            ax_coef.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax_coef.set_xlabel("Signed Coefficient", fontsize=10)
            ax_coef.set_title(
                f"{mname} — Signed Coefficients (top {top_n})",
                fontsize=12, fontweight="bold", color=color
            )
            ax_coef.legend(handles=[
                Patch(facecolor="#D95F02", label="Positive (↑ TE)"),
                Patch(facecolor="#1B9E77", label="Negative (↓ TE)"),
            ], fontsize=9, loc="lower right")

        else:
            # tree models: no signed coefficients — repeat importance bar
            ax_coef.barh(
                range(top_n), top.values[::-1],
                color=color, alpha=0.85, edgecolor="white"
            )
            if mname in ("LightGBM", "XGBoost"):
                xlabel = "Gain Importance"
                note = "gain-weighted"
            else:
                xlabel = "MDI Importance"
                note = "mean decrease in impurity"
            ax_coef.set_xlabel(xlabel, fontsize=10)
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


def plot_model_comparison(imp_dict, label, out_prefix, top_n=TOP_N_HEATMAP):
    """
    Heatmap comparing feature importances across all five models.

    Takes the union of the top-N features from each model, then builds a
    matrix of (features × models) where each model's column is normalised
    to [0, 1]. This lets you compare models with very different importance
    scales (e.g. gain values in the thousands vs MDI values near zero) on
    the same colour axis.

    Features that are dark red in all four columns are the most reliable
    biological signals — every model independently decided they were
    important regardless of its learning algorithm.

    Parameters
    ----------
    imp_dict   : dict  {model_name: pd.Series of importances}
    label      : str   dataset label for the figure title
    out_prefix : str   path prefix; PNG saved as <out_prefix>_model_comparison.png
    top_n      : int   top features per model to include (default from config)

    Returns
    -------
    str : path to the saved PNG
    """
    safe_label = label.replace(" ", "_")
    fig_path = f"{out_prefix}_{safe_label}_model_comparison.png"
    #fig_path = f"{out_prefix}_model_comparison.png"

    # union of top-N features from every model
    top_features = set()
    for s in imp_dict.values():
        top_features.update(s.nlargest(top_n).index.tolist())
    top_features = sorted(top_features)

    # build matrix and normalise each model column to [0, 1]
    df_heat = pd.DataFrame(
        {m: s.reindex(top_features).fillna(0) for m, s in imp_dict.items()}
    )
    df_heat = df_heat / df_heat.max()

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

