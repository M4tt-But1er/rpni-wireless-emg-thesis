"""Nested hyperparameter search for RandomForest and SVM, wrapped around the
existing (unchanged) Stage-3 feature selection.

Design: feature selection and hyperparameter tuning are done sequentially,
not jointly -- a full joint search (every feature subset x every
hyperparameter combo) would multiply an already-large cost further for
limited extra benefit. For each outer training fold: Stage 3 picks features
using the FIXED default hyperparameters (unchanged from every other result
in this project), then a grid search over hyperparameters is run (its own
inner, grouped, stratified CV) using THAT feature set. This keeps the two
concerns separable and the cost bounded, while still answering the actual
question asked: does properly cross-validated hyperparameter selection
change these models' accuracy, retrained per condition.

Grids are intentionally modest (8 RF combos, 12 SVM combos) -- this is a
pilot to check whether tuning is worth extending to the full ~160-condition
sweep, not an exhaustive search.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0
N_INNER_SPLITS = 3

RF_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 15],
    "min_samples_leaf": [1, 3],
}
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1],
}


def make_tunable_base(model_name: str):
    """Same fixed choices as models.py's make_model_pipeline (class_weight,
    kernel, random_state) EXCEPT the hyperparameters being tuned here."""
    if model_name == "RandomForest":
        return False, RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1), RF_PARAM_GRID
    if model_name == "SVM":
        return True, SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE), SVM_PARAM_GRID
    raise ValueError(model_name)


def tune_hyperparameters(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Grouped-stratified inner CV grid search. Returns (best_params, best_inner_cv_score)."""
    needs_scaling, base_est, param_grid = make_tunable_base(model_name)
    Xs = StandardScaler().fit_transform(X) if needs_scaling else X

    n_groups_smallest_class = min(np.unique(groups[y == c]).size for c in np.unique(y))
    n_splits = min(N_INNER_SPLITS, max(2, n_groups_smallest_class))
    inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(base_est, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1, refit=False)
    search.fit(Xs, y, groups=groups)
    return search.best_params_, search.best_score_


def fit_tuned_model(model_name: str, X: np.ndarray, y: np.ndarray, best_params: dict):
    needs_scaling, base_est, _ = make_tunable_base(model_name)
    base_est.set_params(**best_params)
    scaler = None
    Xf = X
    if needs_scaling:
        scaler = StandardScaler().fit(X)
        Xf = scaler.transform(X)
    base_est.fit(Xf, y)
    return base_est, scaler
