"""Central configuration for the Heart Disease XAI project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "trained_models"
REPORT_DIR = ROOT / "report"

RAW_DATASET = DATA_RAW / "cleveland.csv"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1: typical angina, 2: atypical, 3: non-anginal, 4: asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (1: up, 2: flat, 3: down)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia (3: normal, 6: fixed defect, 7: reversible defect)",
}

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
