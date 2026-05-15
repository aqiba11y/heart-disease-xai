"""Load and clean the UCI Cleveland Heart Disease dataset."""
import pandas as pd
import numpy as np
from src.config import RAW_DATASET, COLUMN_NAMES, DATA_PROCESSED


def load_raw_data() -> pd.DataFrame:
    """Load the raw UCI Cleveland dataset and assign column names."""
    df = pd.read_csv(RAW_DATASET, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and convert target to binary."""
    df = df.copy()

    # Original target is 0 (no disease) and 1-4 (varying disease severity).
    # We collapse it into a binary classification problem: 0 = healthy, 1 = disease.
    df["target"] = (df["target"] > 0).astype(int)

    # Impute missing values in `ca` and `thal` using the mode (categorical features)
    for col in ["ca", "thal"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Cast known-categorical columns to int
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]:
        df[col] = df[col].astype(int)

    return df


def get_clean_dataset() -> pd.DataFrame:
    """Public entry point: load and clean the dataset."""
    df = load_raw_data()
    df = clean_data(df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "cleveland_clean.csv", index=False)
    return df


if __name__ == "__main__":
    df = get_clean_dataset()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"Class distribution:\n{df['target'].value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
