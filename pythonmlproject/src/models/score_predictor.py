"""Score prediction — regression models."""
from __future__ import annotations
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Scikit-learn 1.6+ compatibility for XGBoost
if hasattr(XGBRegressor, "__sklearn_tags__"):
    def _xgb_reg_tags(self):
        tags = super(XGBRegressor, self).__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags
    XGBRegressor.__sklearn_tags__ = _xgb_reg_tags

import warnings
warnings.filterwarnings('ignore')

REGRESS_NAMES = ['linear', 'ridge', 'lasso', 'xgb']
SCORE_FEATURES = [
    'over', 'cum_runs', 'cum_wickets', 'run_rate',
    'venue_enc', 'season_idx', 'overs_limit'
]


def build_models(cfg: dict) -> dict:
    """Return a dict of untrained regression model instances."""
    models = {
        'linear': LinearRegression(),
        'ridge': RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5),
        'lasso': LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=5000),
        'xgb': XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }
    return models


def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Fit all 4 regression models.

    Returns
    -------
    {name: fitted_estimator}
    """
    models = build_models(cfg)
    fitted: dict = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def evaluate_all(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate all regression models on the test set.

    Returns DataFrame with columns:
        model, mae, rmse, r2, adj_r2
    """
    n = len(y_test)
    p = X_test.shape[1]
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        # Adjusted R²: 1 - (1 - R²) * (n - 1) / (n - p - 1)
        denom = n - p - 1
        if denom > 0:
            adj_r2 = 1 - (1 - r2) * (n - 1) / denom
        else:
            adj_r2 = float('nan')

        rows.append({
            'model': name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'adj_r2': adj_r2,
        })

    return pd.DataFrame(rows)


def save_all(
    models: dict,
    models_dir: str,
    feat_cols: list,
) -> None:
    """
    Persist regression models and feature list to *models_dir*.

    Files written
    -------------
    score_{name}.joblib   — one per model
    score_features.joblib
    """
    mdir = Path(models_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, mdir / f'score_{name}.joblib')

    joblib.dump(feat_cols, mdir / 'score_features.joblib')


def load_all(models_dir: str) -> dict:
    """
    Load all regression artefacts from *models_dir*.

    Returns
    -------
    {'models': {name: estimator}, 'feat_cols': list}
    """
    mdir = Path(models_dir)
    result: dict = {'models': {}, 'feat_cols': []}

    for name in REGRESS_NAMES:
        p = mdir / f'score_{name}.joblib'
        if p.exists():
            result['models'][name] = joblib.load(p)

    feat_path = mdir / 'score_features.joblib'
    if feat_path.exists():
        result['feat_cols'] = joblib.load(feat_path)

    return result


def predict_score(
    models: dict,
    X: np.ndarray,
) -> dict:
    """
    Predict final score for each model.

    Parameters
    ----------
    X : shape (1, n_features) or (n_samples, n_features)

    Returns
    -------
    {name: float}  — predicted score (mean over rows if multiple)
    """
    preds: dict = {}
    for name, model in models.items():
        y_hat = model.predict(X)
        preds[name] = float(np.mean(y_hat))
    return preds


def get_coefficients(
    models: dict,
    feat_cols: list,
) -> dict:
    """
    Extract learned coefficients for linear, ridge, lasso.

    Returns
    -------
    {name: pd.Series(coef, index=feat_cols)}
    """
    coef_dict: dict = {}
    linear_names = {'linear', 'ridge', 'lasso'}

    for name, model in models.items():
        if name not in linear_names:
            continue
        if hasattr(model, 'coef_'):
            coef = model.coef_
            # RidgeCV / LassoCV may return 1-D or 2-D
            coef = np.array(coef).ravel()
            coef_dict[name] = pd.Series(coef, index=feat_cols)

    return coef_dict


def get_learning_curves(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """
    Compute learning curves for all models.

    Returns
    -------
    {name: (train_sizes, train_scores, val_scores)}
        train_sizes  : np.ndarray of sample counts
        train_scores : np.ndarray of shape (n_sizes, cv)
        val_scores   : np.ndarray of shape (n_sizes, cv)
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    curves: dict = {}

    for name, model in models.items():
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            cv=kf,
            scoring='neg_mean_absolute_error',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        curves[name] = (train_sizes, train_scores, val_scores)

    return curves
