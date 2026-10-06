"""Match winner classification — 11 models."""
from __future__ import annotations
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                      GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report, roc_curve)
from xgboost import XGBClassifier

# Scikit-learn 1.6+ compatibility for XGBoost
if hasattr(XGBClassifier, "__sklearn_tags__"):
    def _xgb_clf_tags(self):
        tags = super(XGBClassifier, self).__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
    XGBClassifier.__sklearn_tags__ = _xgb_clf_tags

try:
    import shap
except Exception:
    shap = None
import warnings

warnings.filterwarnings('ignore')

CLASSIF_NAMES = ['lr', 'knn', 'nb', 'svm', 'dt', 'bag', 'rf', 'ada', 'gb', 'xgb', 'stack']

CLASSIF_FEATURES = [
    'team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_bat_first',
    'toss_winner_is_team1', 'venue_enc', 'season_idx',
    'team1_h2h_wins', 'venue_team1_wins', 'team1_elo', 'team2_elo'
]

# Models that require feature scaling
_SCALE_MODELS = {'lr', 'knn', 'svm'}


def build_models(cfg: dict) -> dict:
    """Return a dict of untrained model instances keyed by short name."""
    # Base estimators used inside StackingClassifier
    _lr = LogisticRegression(max_iter=1000, random_state=42)
    _rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_split=5, random_state=42)
    _gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)

    models = {
        'lr': LogisticRegression(max_iter=1000, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=7),
        'nb': GaussianNB(),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'dt': DecisionTreeClassifier(
            max_depth=5, min_samples_split=20, min_samples_leaf=10,
            random_state=42, criterion='gini'
        ),
        'bag': BaggingClassifier(n_estimators=200, random_state=42),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
        ),
        'ada': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
        'xgb': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        ),
        'stack': StackingClassifier(
            estimators=[('lr', _lr), ('rf', _rf), ('gb', _gb)],
            final_estimator=XGBClassifier(n_estimators=100, random_state=42,
                                          use_label_encoder=False,
                                          eval_metric='logloss'),
            cv=5,
            stack_method='predict_proba'
        ),
    }
    return models


def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> tuple[dict, StandardScaler]:
    """
    Fit all 11 classifiers.

    Returns
    -------
    fitted_models : dict  {name -> fitted estimator}
    scaler        : StandardScaler fitted on X_train
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    models = build_models(cfg)
    fitted: dict = {}

    for name, model in models.items():
        X_fit = X_scaled if name in _SCALE_MODELS else X_train
        model.fit(X_fit, y_train)
        fitted[name] = model

    return fitted, scaler


def evaluate_all(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Evaluate all models on the test set.

    Returns DataFrame with columns:
        model, accuracy, auc, precision, recall, f1
    """
    X_scaled = scaler.transform(X_test)
    rows = []

    for name, model in models.items():
        X_ev = X_scaled if name in _SCALE_MODELS else X_test
        y_pred = model.predict(X_ev)

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_ev)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float('nan')

        rows.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': auc,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        })

    return pd.DataFrame(rows)


def save_all(
    models: dict,
    models_dir: str,
    scaler: StandardScaler,
    feat_cols: list,
    label_encoders: dict,
) -> None:
    """
    Persist all artefacts to *models_dir*.

    Files written
    -------------
    match_winner_{name}.joblib  — one per model
    classif_scaler.joblib
    classif_features.joblib
    label_encoders.joblib
    """
    mdir = Path(models_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, mdir / f'match_winner_{name}.joblib')

    joblib.dump(scaler, mdir / 'classif_scaler.joblib')
    joblib.dump(feat_cols, mdir / 'classif_features.joblib')
    joblib.dump(label_encoders, mdir / 'label_encoders.joblib')


def load_all(models_dir: str) -> dict:
    """
    Load all artefacts from *models_dir*.

    Returns
    -------
    {
        'models': {name: estimator},
        'scaler': StandardScaler,
        'feat_cols': list,
        'label_encoders': dict,
    }
    """
    mdir = Path(models_dir)
    result: dict = {'models': {}, 'scaler': None, 'feat_cols': [], 'label_encoders': {}}

    for name in CLASSIF_NAMES:
        p = mdir / f'match_winner_{name}.joblib'
        if p.exists():
            result['models'][name] = joblib.load(p)

    scaler_path = mdir / 'classif_scaler.joblib'
    if scaler_path.exists():
        result['scaler'] = joblib.load(scaler_path)

    feat_path = mdir / 'classif_features.joblib'
    if feat_path.exists():
        result['feat_cols'] = joblib.load(feat_path)

    le_path = mdir / 'label_encoders.joblib'
    if le_path.exists():
        result['label_encoders'] = joblib.load(le_path)

    return result


def predict_win_proba(
    models: dict,
    X: np.ndarray,
    scaler: StandardScaler,
) -> dict:
    """
    Return P(team1 wins) for each model.

    Parameters
    ----------
    X : shape (1, n_features) or (n_samples, n_features)

    Returns
    -------
    {name: float}  — probability of team1 winning
    """
    X_scaled = scaler.transform(X)
    probs: dict = {}

    for name, model in models.items():
        X_in = X_scaled if name in _SCALE_MODELS else X
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X_in)[:, 1]
        else:
            p = model.predict(X_in).astype(float)
        probs[name] = float(np.mean(p))

    return probs


def run_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    scaler: StandardScaler,
) -> dict:
    """
    Grid search for lr, knn, svm, dt, rf.

    Returns
    -------
    {model_name: {'best_params': dict, 'best_score': float}}
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_scaled = scaler.transform(X)

    search_configs = {
        'lr': (
            LogisticRegression(max_iter=1000, random_state=42),
            {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']},
            X_scaled,
        ),
        'knn': (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]},
            X_scaled,
        ),
        'svm': (
            SVC(kernel='rbf', probability=True, random_state=42),
            {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']},
            X_scaled,
        ),
        'dt': (
            DecisionTreeClassifier(random_state=42),
            {'max_depth': [3, 5, 7, 10, None], 'criterion': ['gini', 'entropy']},
            X,
        ),
        'rf': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
            X,
        ),
    }

    results: dict = {}
    for name, (estimator, param_grid, X_use) in search_configs.items():
        gs = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        gs.fit(X_use, y)
        results[name] = {
            'best_params': gs.best_params_,
            'best_score': float(gs.best_score_),
        }

    return results


def run_random_search(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    scaler: StandardScaler,
) -> dict:
    """
    Randomised search for xgb.

    Returns
    -------
    {model_name: {'best_params': dict, 'best_score': float}}
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
    }

    xgb = XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )
    rs = RandomizedSearchCV(
        xgb, param_dist, n_iter=50, cv=cv,
        scoring='accuracy', random_state=42, n_jobs=-1
    )
    rs.fit(X, y)

    return {
        'xgb': {
            'best_params': rs.best_params_,
            'best_score': float(rs.best_score_),
        }
    }


def get_shap_values(
    models: dict,
    X: np.ndarray,
    feat_cols: list,
) -> tuple:
    """
    Compute SHAP values using the 'xgb' model (TreeExplainer).

    Returns
    -------
    (shap_values: np.ndarray, expected_value: float)

    Raises
    ------
    ValueError  if 'xgb' is not in *models*
    """
    if 'xgb' not in models:
        raise ValueError("'xgb' model not found in models dict. Train or load it first.")

    xgb_model = models['xgb']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    expected_value = float(explainer.expected_value)

    return shap_values, expected_value


def get_cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    models: dict,
    scaler: StandardScaler,
    k: int = 5,
) -> dict:
    """
    k-fold cross-validation accuracy for every model.

    Returns
    -------
    {name: np.ndarray of shape (k,)}
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    X_scaled = scaler.transform(X)
    cv_scores: dict = {}

    for name, model in models.items():
        X_use = X_scaled if name in _SCALE_MODELS else X
        scores = cross_val_score(model, X_use, y, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_scores[name] = scores

    return cv_scores
