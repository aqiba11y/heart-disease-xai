"""Hyperparameter tuning with GridSearchCV and 10-fold cross-validation."""
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE
from src.models import get_models, get_param_grids


def tune_model(name, model, param_grid, X_train, y_train, verbose: int = 0):
    """Run GridSearchCV for a single model and return the best estimator."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_all_models(X_train, y_train, verbose: int = 1) -> dict:
    """Tune every model and return a dict of best estimators with metadata."""
    models = get_models()
    grids = get_param_grids()
    results = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if verbose:
            print(f"\n[Tuning] {name} ...")
        best_model, best_params, best_cv_score = tune_model(
            name, model, grids[name], X_train, y_train
        )
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
        }
        if verbose:
            print(f"  best CV AUC: {best_cv_score:.4f}")
            print(f"  best params: {best_params}")

        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(best_model, MODELS_DIR / f"{filename}.joblib")

    return results
