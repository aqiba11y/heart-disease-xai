"""End-to-end training orchestrator.

Run this script (or `python -m src.train`) to execute the full pipeline:
    load -> clean -> preprocess -> tune -> evaluate -> visualize -> explain.
"""
import sys
import time

import joblib
import pandas as pd

from src.config import MODELS_DIR, NUMERIC_FEATURES, RESULTS_DIR
from src.data_loader import get_clean_dataset
from src.evaluation import evaluate_all_models
from src.explainability import explain_with_shap
from src.preprocessing import prepare_data
from src.tuning import tune_all_models
from src.visualize import run_all_plots


def main():
    t0 = time.time()

    print("=" * 70)
    print("HEART DISEASE PREDICTION - EXPLAINABLE AI PIPELINE")
    print("=" * 70)

    print("\n[1/5] Loading and cleaning the UCI Cleveland dataset ...")
    df = get_clean_dataset()
    print(f"  -> {len(df)} patients, {df.shape[1] - 1} features")
    print(f"  -> class balance: {df['target'].value_counts().to_dict()}")

    print("\n[2/5] Preprocessing (scale + one-hot + SMOTE) ...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    print(f"  -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n[3/5] Hyperparameter tuning with 10-fold CV (this may take a few minutes) ...")
    tuned_results = tune_all_models(X_train, y_train, verbose=1)

    print("\n[4/5] Evaluating all models on the test set ...")
    metrics_df, predictions = evaluate_all_models(tuned_results, X_test, y_test)
    print("\nFinal comparison table:")
    print(metrics_df.to_string(index=False))

    print("\n[5/5] Generating visualizations and SHAP explanations ...")
    run_all_plots(df, NUMERIC_FEATURES, tuned_results, predictions, y_test, metrics_df)

    best_name = metrics_df.iloc[0]["Model"]
    best_model = tuned_results[best_name]["model"]
    print(f"\n  Best model: {best_name}")
    print(f"  Generating SHAP plots for {best_name} ...")
    try:
        explain_with_shap(best_model, X_train, X_test, feature_names, best_name)
    except Exception as exc:
        print(f"  SHAP failed for {best_name}: {exc}")
        # Fall back to a tree-friendly model
        for fallback in ["XGBoost", "Random Forest", "Decision Tree"]:
            if fallback in tuned_results and fallback != best_name:
                print(f"  Trying fallback: {fallback}")
                try:
                    explain_with_shap(
                        tuned_results[fallback]["model"], X_train, X_test, feature_names, fallback
                    )
                    break
                except Exception as e2:
                    print(f"    {fallback} also failed: {e2}")

    joblib.dump(
        {
            "best_model_name": best_name,
            "best_model": best_model,
            "preprocessor": preprocessor,
            "feature_names": feature_names,
        },
        MODELS_DIR / "best_pipeline.joblib",
    )
    print(f"\n  Saved best pipeline -> {MODELS_DIR / 'best_pipeline.joblib'}")

    print(f"\nDone in {time.time() - t0:.1f} seconds.")
    print(f"All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
