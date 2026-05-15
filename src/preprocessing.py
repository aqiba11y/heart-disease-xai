"""Preprocessing pipeline: scaling, encoding, splitting, and class balancing."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the column transformer that scales numerics and one-hot encodes categoricals."""
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame):
    """Train/test split with stratification on the target."""
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to balance the training set."""
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def prepare_data(df: pd.DataFrame, use_smote: bool = True):
    """Complete preprocessing pipeline.

    Returns transformed train/test arrays plus the fitted preprocessor for reuse.
    """
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    if use_smote:
        X_train_t, y_train = apply_smote(X_train_t, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    feature_names = (
        NUMERIC_FEATURES
        + list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
    )

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    from src.data_loader import get_clean_dataset

    df = get_clean_dataset()
    X_train, X_test, y_train, y_test, pre, names = prepare_data(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"After SMOTE - class balance: {np.bincount(y_train)}")
    print(f"Feature names ({len(names)}): {names}")
