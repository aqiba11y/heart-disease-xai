"""Evaluation metrics and model comparison."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import RESULTS_DIR


def evaluate_model(name, model, X_test, y_test) -> dict:
    """Compute the full classification metric set for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    return metrics, y_pred, y_proba


def evaluate_all_models(tuned_results, X_test, y_test) -> pd.DataFrame:
    """Evaluate every tuned model and build a comparison table."""
    rows = []
    predictions = {}

    for name, info in tuned_results.items():
        metrics, y_pred, y_proba = evaluate_model(name, info["model"], X_test, y_test)
        metrics["CV-AUC"] = info["best_cv_score"]
        rows.append(metrics)
        predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    return df, predictions
