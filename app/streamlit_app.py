"""Streamlit web app: Heart Disease Risk Prediction with Explainable AI."""
import sys
from pathlib import Path

# Ensure project root is on the import path so we can import from src.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RESULTS_DIR,
)

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="HD",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline():
    """Load the trained best-pipeline artifact (cached across reruns)."""
    bundle = joblib.load(MODELS_DIR / "best_pipeline.joblib")
    return bundle


@st.cache_data
def load_metrics():
    return pd.read_csv(RESULTS_DIR / "metrics.csv")


def patient_input_form():
    """Render the patient input sidebar and return a single-row DataFrame."""
    st.sidebar.header("Patient Information")

    age = st.sidebar.slider("Age", 25, 80, 55, help=FEATURE_DESCRIPTIONS["age"])
    sex = st.sidebar.selectbox("Sex", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        [("Typical Angina", 1), ("Atypical Angina", 2), ("Non-anginal Pain", 3), ("Asymptomatic", 4)],
        format_func=lambda x: x[0],
    )[1]
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
    )[1]
    restecg = st.sidebar.selectbox(
        "Resting ECG",
        [("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)],
        format_func=lambda x: x[0],
    )[1]
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.sidebar.selectbox(
        "Exercise-Induced Angina?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
    )[1]
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST",
        [("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)],
        format_func=lambda x: x[0],
    )[1]
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox(
        "Thalassemia",
        [("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)],
        format_func=lambda x: x[0],
    )[1]

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    return pd.DataFrame([data])


def risk_gauge(probability: float):
    """Display the predicted risk as a colored progress bar plus a verdict."""
    pct = probability * 100
    if pct < 30:
        color, level = "green", "LOW RISK"
    elif pct < 60:
        color, level = "orange", "MODERATE RISK"
    else:
        color, level = "red", "HIGH RISK"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; border-radius: 12px;
                    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);">
            <h1 style="color: {color}; margin: 0;">{pct:.1f}%</h1>
            <h3 style="color: {color}; margin: 0;">{level}</h3>
            <p>Predicted probability of heart disease</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def explain_prediction(model, preprocessor, X_transformed, feature_names):
    """Generate a SHAP waterfall-style bar chart for a single prediction."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    contributions = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_values[0]})
    contributions["abs"] = contributions["SHAP Value"].abs()
    contributions = contributions.sort_values("abs", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contributions["SHAP Value"]]
    ax.barh(contributions["Feature"], contributions["SHAP Value"], color=colors)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Top 10 Features Driving This Prediction")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig, contributions


def main():
    st.title("Heart Disease Risk Predictor")
    st.markdown(
        "**An Explainable AI system for cardiovascular disease risk assessment** | "
        "Built with scikit-learn, XGBoost, and SHAP"
    )

    bundle = load_pipeline()
    model = bundle["best_model"]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    best_name = bundle["best_model_name"]

    st.sidebar.info(f"**Active model:** {best_name}")
    input_df = patient_input_form()

    tab_pred, tab_compare, tab_eda, tab_about = st.tabs(
        ["Prediction", "Model Comparison", "Data Insights", "About"]
    )

    with tab_pred:
        st.subheader("Patient Risk Assessment")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Input summary**")
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        with col2:
            X_t = preprocessor.transform(input_df)
            proba = model.predict_proba(X_t)[0, 1]
            risk_gauge(proba)

            st.write("")
            if proba >= 0.5:
                st.error("Recommendation: Consult a cardiologist for further evaluation.")
            else:
                st.success("Recommendation: Maintain a healthy lifestyle and routine checkups.")

        st.markdown("---")
        st.subheader("Why this prediction? (Explainable AI)")
        try:
            fig, contributions = explain_prediction(model, preprocessor, X_t, feature_names)
            colA, colB = st.columns([2, 1])
            with colA:
                st.pyplot(fig)
            with colB:
                st.write("**Top contributing features**")
                st.dataframe(
                    contributions[["Feature", "SHAP Value"]].round(3).reset_index(drop=True),
                    use_container_width=True,
                )
                st.caption("Red bars = pushes risk UP, Green bars = pushes risk DOWN.")
        except Exception as exc:
            st.warning(f"SHAP explanation unavailable for the active model: {exc}")

    with tab_compare:
        st.subheader("Comparing 7 AI Algorithms")
        metrics_df = load_metrics()
        st.dataframe(metrics_df.style.format("{:.4f}", subset=metrics_df.columns[1:]), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if (FIGURES_DIR / "07_metrics_comparison.png").exists():
                st.image(str(FIGURES_DIR / "07_metrics_comparison.png"))
        with col2:
            if (FIGURES_DIR / "05_roc_curves.png").exists():
                st.image(str(FIGURES_DIR / "05_roc_curves.png"))

        col3, col4 = st.columns(2)
        with col3:
            if (FIGURES_DIR / "04_confusion_matrices.png").exists():
                st.image(str(FIGURES_DIR / "04_confusion_matrices.png"))
        with col4:
            if (FIGURES_DIR / "06_pr_curves.png").exists():
                st.image(str(FIGURES_DIR / "06_pr_curves.png"))

    with tab_eda:
        st.subheader("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if (FIGURES_DIR / "01_class_distribution.png").exists():
                st.image(str(FIGURES_DIR / "01_class_distribution.png"))
        with col2:
            if (FIGURES_DIR / "08_best_model_summary.png").exists():
                st.image(str(FIGURES_DIR / "08_best_model_summary.png"))
        if (FIGURES_DIR / "02_correlation_heatmap.png").exists():
            st.image(str(FIGURES_DIR / "02_correlation_heatmap.png"))
        if (FIGURES_DIR / "03_feature_distributions.png").exists():
            st.image(str(FIGURES_DIR / "03_feature_distributions.png"))

    with tab_about:
        st.subheader("About this Project")
        st.markdown(
            """
            **Course:** CSC 412 - Artificial Intelligence
            **Institution:** Bahria University, H-11 Campus, Islamabad
            **Domain:** Healthcare & Bioinformatics (SDG 3)

            ### Methodology
            1. **Dataset:** UCI Cleveland Heart Disease (303 patients, 13 clinical features)
            2. **Preprocessing:** Missing-value imputation, standard scaling, one-hot encoding, SMOTE
            3. **9 Models compared** (6 classifiers + 3 regressors):
               - **Classifiers:** Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Classifier, K-Nearest Neighbors, Gaussian Naive Bayes
               - **Regressors (thresholded at 0.5):** Decision Tree Regressor, Random Forest Regressor, Support Vector Regressor
            4. **Hyperparameter tuning:** GridSearchCV with 10-fold stratified cross-validation
            5. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, PR curves, confusion matrices
            6. **Explainability:** SHAP values for both global and per-patient explanations

            ### Why XAI matters in healthcare
            Black-box models are insufficient for medical decisions; clinicians need to understand
            *why* a model flagged a patient as high-risk. SHAP exposes per-feature contributions to
            every prediction, making AI accountable and usable in clinical practice.
            """
        )


if __name__ == "__main__":
    main()
