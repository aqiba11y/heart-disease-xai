"""All plots produced by the pipeline."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import FIGURES_DIR

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts().sort_index()
    sns.barplot(x=["No Disease", "Disease"], y=counts.values, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    _save(fig, "01_class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "02_correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame, numeric_features):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        if i >= len(axes):
            break
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=axes[i], palette=["#2ecc71", "#e74c3c"])
        axes[i].set_title(f"Distribution of {col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Feature Distributions by Target Class", fontsize=14, fontweight="bold")
    _save(fig, "03_feature_distributions.png")


def plot_confusion_matrices(tuned_results, predictions, y_test):
    n = len(tuned_results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (name, info) in enumerate(tuned_results.items()):
        cm = confusion_matrix(y_test, predictions[name]["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
        axes[i].set_title(name)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Confusion Matrices Across All Models", fontsize=14, fontweight="bold")
    _save(fig, "04_confusion_matrices.png")


def plot_roc_curves(tuned_results, predictions, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, info in tuned_results.items():
        y_proba = predictions[name]["y_proba"]
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right")
    _save(fig, "05_roc_curves.png")


def plot_pr_curves(tuned_results, predictions, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, info in tuned_results.items():
        y_proba = predictions[name]["y_proba"]
        if y_proba is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax.plot(recall, precision, lw=2, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - All Models")
    ax.legend(loc="lower left")
    _save(fig, "06_pr_curves.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame):
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(13, 6))
    melted = metrics_df.melt(id_vars="Model", value_vars=metrics_cols, var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax, palette="viridis")
    ax.set_title("Model Comparison Across All Metrics", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=5)
    plt.xticks(rotation=20, ha="right")
    _save(fig, "07_metrics_comparison.png")


def plot_best_model_summary(metrics_df: pd.DataFrame):
    best = metrics_df.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [best[m] for m in metrics_cols]
    bars = ax.barh(metrics_cols, values, color=sns.color_palette("crest", len(metrics_cols)))
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Best Model: {best['Model']}", fontweight="bold")
    _save(fig, "08_best_model_summary.png")


def run_all_plots(df, numeric_features, tuned_results, predictions, y_test, metrics_df):
    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df, numeric_features)
    plot_confusion_matrices(tuned_results, predictions, y_test)
    plot_roc_curves(tuned_results, predictions, y_test)
    plot_pr_curves(tuned_results, predictions, y_test)
    plot_metrics_comparison(metrics_df)
    plot_best_model_summary(metrics_df)
    print(f"\nAll plots saved to: {FIGURES_DIR}")
