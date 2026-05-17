"""Model definitions and hyperparameter search spaces.

We compare nine models drawn from the topics covered in the course:
    Classifiers:    Logistic Regression, Decision Tree, Random Forest,
                    Support Vector Classifier, K-Nearest Neighbours, Naive Bayes
    Regressors:     Decision Tree Regressor, Random Forest Regressor,
                    Support Vector Regressor
Regressors are wrapped so they can produce class labels (threshold at 0.5)
and probability-like scores for the standard classification metrics.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import RANDOM_STATE


class RegressorAsClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a regressor so it works inside our classification pipeline.

    The regressor learns to predict the binary target as a continuous value
    in roughly [0, 1]. We expose predict() (threshold at 0.5) and
    predict_proba() (raw output clipped to [0, 1]) so all sklearn metrics
    and SHAP machinery keep working unchanged.

    Parameter access (including nested params via the ``regressor__`` prefix)
    is handled automatically by ``BaseEstimator`` because we store the wrapped
    estimator as the ``regressor`` attribute exactly as sklearn expects.
    """

    # Explicitly tell sklearn this is a classifier (the inner regressor
    # attribute would otherwise confuse newer versions of sklearn that
    # introspect estimator type via tags).
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        return tags

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, np.asarray(y, dtype=float))
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        raw = self.regressor_.predict(X)
        return (raw >= 0.5).astype(int)

    def predict_proba(self, X):
        raw = np.clip(self.regressor_.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - raw, raw])


def get_models() -> dict:
    """Return all nine candidate models."""
    return {
        # ---- Classifiers ----
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree (Classification)": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest (Classification)": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Classifier": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        # ---- Regressors (wrapped) ----
        "Decision Tree (Regression)": RegressorAsClassifier(
            DecisionTreeRegressor(random_state=RANDOM_STATE)
        ),
        "Random Forest (Regression)": RegressorAsClassifier(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        ),
        "Support Vector Regression": RegressorAsClassifier(SVR()),
    }


def get_param_grids() -> dict:
    """Hyperparameter search spaces. Regressor params are accessed via
    the `regressor__` prefix exposed by RegressorAsClassifier.get_params."""
    return {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "Decision Tree (Classification)": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest (Classification)": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
        "Support Vector Classifier": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Naive Bayes": {
            "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        },
        "Decision Tree (Regression)": {
            "regressor__max_depth": [3, 5, 7, 10, None],
            "regressor__min_samples_split": [2, 5, 10],
        },
        "Random Forest (Regression)": {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [5, 10, None],
        },
        "Support Vector Regression": {
            "regressor__C": [0.1, 1, 10],
            "regressor__kernel": ["rbf", "linear"],
            "regressor__gamma": ["scale", "auto"],
        },
    }
