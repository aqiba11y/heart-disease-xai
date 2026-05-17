"""Render the system-architecture block diagram as a PNG figure."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import FIGURES_DIR


def draw_block_diagram():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    blocks = [
        # (y-center, title, subtitle, color)
        (13.2, "INPUT LAYER", "UCI Cleveland Heart Disease Dataset\n303 patients x 14 columns", "#3498db"),
        (11.7, "DATA PREPROCESSING", "Load CSV - Binary target - Impute missing\n(ca, thal via mode)", "#2ecc71"),
        (10.2, "FEATURE ENGINEERING", "Numeric: z = (x - mean) / std\nCategorical: One-Hot Encoding", "#2ecc71"),
        (8.7,  "DATA SPLITTING + BALANCING", "Stratified 80/20 split\nSMOTE on training set only", "#f39c12"),
        (7.0,  "MODEL LAYER (9 Models)",
               "Classifiers: LR, DT, RF, SVC, KNN, NB\nRegressors: DTR, RFR, SVR (thresholded)", "#9b59b6"),
        (5.2,  "HYPERPARAMETER TUNING", "GridSearchCV + 10-Fold Stratified CV\nScoring: ROC-AUC", "#e67e22"),
        (3.7,  "EVALUATION",
               "Accuracy, Precision, Recall, F1, ROC-AUC\nConfusion Matrix, ROC, PR curves", "#1abc9c"),
        (2.2,  "EXPLAINABLE AI (SHAP)", "Global feature importance\nPer-patient explanations", "#e74c3c"),
        (0.7,  "DEPLOYMENT LAYER", "Streamlit Web App\nLive risk gauge + SHAP explanation", "#34495e"),
    ]

    block_w = 7.5
    block_h = 1.1
    x_center = 5.0

    for y, title, subtitle, color in blocks:
        box = FancyBboxPatch(
            (x_center - block_w / 2, y - block_h / 2),
            block_w, block_h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=2,
            edgecolor="black",
            facecolor=color,
            alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(x_center, y + 0.22, title,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x_center, y - 0.22, subtitle,
                ha="center", va="center",
                fontsize=9, color="white")

    # Arrows between consecutive blocks
    for i in range(len(blocks) - 1):
        y_top = blocks[i][0] - block_h / 2
        y_bot = blocks[i + 1][0] + block_h / 2
        arrow = FancyArrowPatch(
            (x_center, y_top), (x_center, y_bot),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2,
            color="black",
        )
        ax.add_patch(arrow)

    ax.set_title("System Architecture: Heart Disease Prediction with Explainable AI",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    out = FIGURES_DIR / "00_system_block_diagram.png"
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    draw_block_diagram()
