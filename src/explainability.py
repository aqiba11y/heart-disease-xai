"""SHAP-based explainability for tree-based and other models."""
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import FIGURES_DIR


def explain_with_shap(model, X_train, X_test, feature_names, model_name: str = "model"):
    """Generate SHAP summary and bar plots for the given model.

    Uses TreeExplainer for tree models when possible, falls back to KernelExplainer otherwise.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # If the model is one of our regressor wrappers, point SHAP at the inner
    # regressor so TreeExplainer can work on tree-based regressors directly.
    underlying = getattr(model, "regressor_", model)

    try:
        explainer = shap.TreeExplainer(underlying)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        background = shap.sample(X_train, min(50, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:50])
        X_test = X_test[:50]

    # XGBoost / RF return either a 2-D array (binary positive class) or a list of 2 arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    safe_name = model_name.lower().replace(" ", "_")

    # Bar plot - global feature importance
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"09_shap_bar_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot - feature impact distribution
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"10_shap_summary_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return explainer, shap_values


def explain_single_prediction(explainer, instance, feature_names, model_name: str = "model"):
    """Explain a single patient prediction using SHAP force plot."""
    shap_values = explainer.shap_values(instance.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values[0]
