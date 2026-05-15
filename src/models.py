"""Model definitions and hyperparameter search spaces for all 6 algorithms."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def get_models() -> dict:
    """Return all candidate models with their default hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    }


def get_param_grids() -> dict:
    """Hyperparameter search spaces for GridSearchCV."""
    return {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        "Neural Network (MLP)": {
            "hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001, 0.01],
        },
    }
