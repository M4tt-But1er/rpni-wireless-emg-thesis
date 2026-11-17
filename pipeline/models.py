"""Model definitions, stage-3 wrapper feature selection, and the final
grouped-stratified 5-fold CV evaluation shared across all models.

Rest is ~53-55% of windows in both participants' combined datasets, so
class_weight='balanced' (or inverse-frequency loss weighting for the LSTM)
is used everywhere it's supported, to avoid majority-class collapse.

All models for a given participant are evaluated on the *exact same* 5 folds
(precomputed once via StratifiedGroupKFold grouped by segment_id) so
cross-model accuracy comparisons are apples-to-apples.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 0
N_SPLITS = 5


class RBFNetClassifier(BaseEstimator, ClassifierMixin):
    """Classic Broomhead-Lowe RBF network: fixed Gaussian basis functions
    centered at k-means cluster centers (widths from the mean nearest-center
    distance, Haykin's heuristic), followed by a trained linear (softmax)
    readout. Standing in for "RBFNN" using only sklearn primitives (no
    dedicated RBFNN library available) -- this is a genuine RBF network,
    not an MLP substitution.
    """

    def __init__(self, n_centers=64, C=1.0, random_state=RANDOM_STATE, sigma_scale=1.0):
        self.n_centers = n_centers
        self.C = C
        self.random_state = random_state
        # multiplier on the heuristic width; 1.0 (default) is the original behaviour,
        # other values are only used by run_rbfnn_tuning.py
        self.sigma_scale = sigma_scale

    def _rbf_transform(self, X):
        d2 = cdist(X, self.centers_, metric="sqeuclidean")
        return np.exp(-d2 / (2 * self.sigma_ ** 2))

    def fit(self, X, y):
        n_centers = min(self.n_centers, X.shape[0])
        km = KMeans(n_clusters=n_centers, random_state=self.random_state, n_init=10).fit(X)
        self.centers_ = km.cluster_centers_
        dists = cdist(self.centers_, self.centers_)
        np.fill_diagonal(dists, np.inf)
        self.sigma_ = dists.min(axis=1).mean()
        if self.sigma_ <= 0:
            self.sigma_ = 1.0
        self.sigma_ = self.sigma_ * self.sigma_scale
        Phi = self._rbf_transform(X)
        self.clf_ = LogisticRegression(max_iter=2000, C=self.C, class_weight="balanced")
        self.clf_.fit(Phi, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self._rbf_transform(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._rbf_transform(X))


def make_model_pipeline(model_name: str):
    """Returns (needs_scaling: bool, estimator)"""
    if model_name == "LDA":
        return True, LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if model_name == "RandomForest":
        return False, RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    if model_name == "SVM":
        return True, SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                          random_state=RANDOM_STATE)
    if model_name == "RBFNN":
        return True, RBFNetClassifier(n_centers=64, C=1.0)
    raise ValueError(model_name)


def compute_cv_folds(df) -> np.ndarray:
    """Fold id (0..N_SPLITS-1) per row, grouped by segment_id, stratified by
    cue. Same partition reused for every model on this dataset."""
    y = df["cue"].to_numpy()
    groups = df["segment_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(df), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(cv.split(np.zeros(len(df)), y, groups)):
        fold_ids[test_idx] = fold
    assert (fold_ids >= 0).all()
    return fold_ids


def evaluate_with_folds(model_name: str, X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray):
    accs = []
    for fold in range(fold_ids.max() + 1):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold
        needs_scaling, est = make_model_pipeline(model_name)
        Xtr, Xte = X[train_mask], X[test_mask]
        if needs_scaling:
            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        est.fit(Xtr, y[train_mask])
        preds = est.predict(Xte)
        accs.append((preds == y[test_mask]).mean())
    return np.array(accs)


# ---------------- Stage 3: per-model wrapper feature selection ----------------

def rfecv_select(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  feature_names: list[str], min_features=3):
    """RFECV for models with native coef_/feature_importances_ (LDA, RandomForest)."""
    needs_scaling, est = make_model_pipeline(model_name)
    Xs = StandardScaler().fit_transform(X) if needs_scaling else X
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(est, step=1, cv=cv.split(Xs, y, groups), scoring="accuracy",
                      min_features_to_select=min_features, n_jobs=-1)
    selector.fit(Xs, y)
    selected = [f for f, keep in zip(feature_names, selector.support_) if keep]
    return selected


def permutation_forward_select(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                feature_names: list[str], max_features=None):
    """For models without native importances (SVM-RBF, RBFNN): rank features
    by permutation importance (fit once on a train/val split), then forward-
    select by adding features in that rank order and tracking grouped-CV
    accuracy; keep the prefix that maximizes accuracy."""
    n = len(feature_names)
    needs_scaling, est = make_model_pipeline(model_name)

    cv_split = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(cv_split.split(X, y, groups))
    Xtr, Xval = X[train_idx], X[val_idx]
    if needs_scaling:
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xval = scaler.transform(Xtr), scaler.transform(Xval)
    est.fit(Xtr, y[train_idx])
    perm = permutation_importance(est, Xval, y[val_idx], n_repeats=10,
                                   random_state=RANDOM_STATE, scoring="accuracy")
    order = np.argsort(perm.importances_mean)[::-1]
    ranked_names = [feature_names[i] for i in order]

    if max_features is None:
        max_features = n
    prefix_lengths = list(range(1, max_features + 1))
    accs = []
    fold_ids_local = np.full(len(y), -1, dtype=int)
    cv2 = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, test_idx) in enumerate(cv2.split(X, y, groups)):
        fold_ids_local[test_idx] = fold

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    for k in prefix_lengths:
        cols = [name_to_idx[nm] for nm in ranked_names[:k]]
        Xk = X[:, cols]
        fold_accs = evaluate_with_folds(model_name, Xk, y, fold_ids_local)
        accs.append(fold_accs.mean())

    best_k = prefix_lengths[int(np.argmax(accs))]
    selected = ranked_names[:best_k]
    return selected, ranked_names, np.array(accs)
