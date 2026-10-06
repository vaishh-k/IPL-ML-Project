# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: ensemble_service.py
# Path: backend/services/ensemble_service.py
# Description: Service layer executing ensemble pipelines (regression and classification targets) and metrics extraction.
# ==============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

# Storage for trained models
_TRAINED_ENSEMBLE_REG = {}
_TRAINED_ENSEMBLE_CLS = {}

# Ensembles require scaling for certain baseline steps, standard scalers are fit
NEEDS_SCALING = {'xgb', 'gb', 'ada', 'rf'}  # Standardize features for consistent performance

class EnsembleRegPipeline:
    def __init__(self):
        # [Function]: __init__
        # [Description]: Implements and executes the   Init   logic within this module pipeline.
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.target_col = ""
        self.categorical_cols = []
        self.numerical_cols = []
        self.models = {}
        self.category_options = {}
        self.best_model_key = None

    def fit_transform_features(self, df):
        # [Function]: fit_transform_features
        # [Description]: Implements and executes the Fit Transform Features logic within this module pipeline.
        self.feature_cols = df.columns.tolist()
        X = df.copy()
        for col in self.feature_cols:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)
                unique_vals = sorted(X[col].astype(str).unique().tolist())
                self.category_options[col] = unique_vals
                
                le = LabelEncoder()
                le.fit(unique_vals + ['<unknown>'])
                self.encoders[col] = le
                X[col] = le.transform(X[col].astype(str))
        return X

    def transform_input(self, input_dict):
        # [Function]: transform_input
        # [Description]: Implements and executes the Transform Input logic within this module pipeline.
        row_df = pd.DataFrame([input_dict])
        for col in self.feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0
                
        for col in self.feature_cols:
            if col in self.categorical_cols:
                le = self.encoders[col]
                val = str(input_dict.get(col, '<unknown>'))
                if val not in le.classes_:
                    val = '<unknown>'
                row_df[col] = le.transform([val])[0]
            else:
                try:
                    row_df[col] = float(input_dict.get(col, 0.0))
                except (ValueError, TypeError):
                    row_df[col] = 0.0
        return row_df[self.feature_cols]

class EnsembleClsPipeline:
    def __init__(self):
        # [Function]: __init__
        # [Description]: Implements and executes the   Init   logic within this module pipeline.
        self.encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.target_col = ""
        self.categorical_cols = []
        self.numerical_cols = []
        self.models = {}
        self.category_options = {}
        self.best_model_key = None
        self.label_names = []

    def fit_transform_features(self, df):
        # [Function]: fit_transform_features
        # [Description]: Implements and executes the Fit Transform Features logic within this module pipeline.
        self.feature_cols = df.columns.tolist()
        X = df.copy()
        for col in self.feature_cols:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)
                unique_vals = sorted(X[col].astype(str).unique().tolist())
                self.category_options[col] = unique_vals
                le = LabelEncoder()
                le.fit(unique_vals + ['<unknown>'])
                self.encoders[col] = le
                X[col] = le.transform(X[col].astype(str))
        return X

    def transform_input(self, input_dict):
        # [Function]: transform_input
        # [Description]: Implements and executes the Transform Input logic within this module pipeline.
        row_df = pd.DataFrame([{c: None for c in self.feature_cols}])
        for col in self.feature_cols:
            if col in self.categorical_cols:
                le = self.encoders[col]
                val = str(input_dict.get(col, '<unknown>'))
                if val not in le.classes_:
                    val = '<unknown>'
                row_df[col] = le.transform([val])[0]
            else:
                try:
                    row_df[col] = float(input_dict.get(col, 0.0))
                except Exception:
                    row_df[col] = 0.0
        return row_df[self.feature_cols]


#  DATA PREPARATION 

def _prepare_first_inning_score(session_id=None):
    # [Function]: _prepare_first_inning_score
    # [Description]: Implements and executes the  Prepare First Inning Score logic within this module pipeline.
    if session_id:
        from utils.helpers import load_dataframe, has_stage_dataframe
        try:
            if has_stage_dataframe(session_id, "cleaned"):
                df = load_dataframe(session_id, "cleaned")
            else:
                df = load_dataframe(session_id, "original")
            
            is_pca = 'PC1' in df.columns and len(df.columns) <= 10
            
            target = None
            for c in ['target_runs', 'runs_scored', 'first_ings_runs', 'score', 'first_inning_total', 'team1_runs']:
                if c in df.columns:
                    target = c
                    break
            
            is_bbb = any(c in df.columns for c in ['runs_off_bat', 'total_runs', 'extra_runs'])
            
            if is_pca or (target is None and not is_bbb):
                matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
                if os.path.exists(matches_path):
                    df = pd.read_csv(matches_path, low_memory=False)
                    target = next((c for c in ['target_runs', 'runs_scored', 'first_ings_runs', 'score', 'team1_runs'] if c in df.columns), None)
                    is_bbb = False
            
            if is_bbb:
                if 'innings' in df.columns:
                    df = df[df['innings'] == 1]
                group_cols = [c for c in ['match_id', 'batting_team', 'bowling_team', 'venue'] if c in df.columns]
                runs_col = next((c for c in ['total_runs', 'runs_off_bat', 'runs', 'runs_scored', 'batter_runs', 'runs_total'] if c in df.columns), None)
                if runs_col is None:
                    raise ValueError('Cannot identify a runs column.')
                agg_df = df.groupby(group_cols)[runs_col].sum().reset_index()
                agg_df.rename(columns={runs_col: 'first_inning_total'}, inplace=True)
                target = 'first_inning_total'
                keep = [c for c in group_cols if c != 'match_id']
                df_clean = agg_df[keep + [target]].dropna()
                return df_clean[keep], df_clean[target], target, keep
            
            if target is None:
                for c in df.columns:
                    if 'run' in c.lower() or 'score' in c.lower():
                        target = c
                        break
            if target is None:
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    target = num_cols[-1]
            if target is None:
                raise ValueError('No target column found in active dataset')
            keep = [c for c in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision'] if c in df.columns]
            if not keep:
                keep = [c for c in df.columns if c != target]
            df_clean = df[keep + [target]].dropna()
            return df_clean[keep], df_clean[target], target, keep
        except Exception:
            pass
    matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
    bbb_path = os.path.join(DATA_DIR, 'processed_ipl_ball_by_ball.csv')
    if os.path.exists(matches_path):
        df = pd.read_csv(matches_path, low_memory=False)
        target = None
        for c in ['target_runs', 'runs_scored', 'first_ings_runs', 'score', 'team1_runs']:
            if c in df.columns:
                target = c
                break
        if target is not None:
            # Exclude season and city based on user's previous preference
            keep = [c for c in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision'] if c in df.columns]
            df = df[keep + [target]].dropna()
            return df[keep], df[target], target, keep

    # Fallback to ball-by-ball aggregation
    # Try processed_ipl_ball_by_ball.csv first, then fall back to raw ipl_ball_by_ball.csv
    for path in [bbb_path, os.path.join(DATA_DIR, 'ipl_ball_by_ball.csv')]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            if 'innings' in df.columns:
                df = df[df['innings'] == 1]
            group_cols = [c for c in ['match_id', 'batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision'] if c in df.columns]
            runs_col = next((c for c in ['total_runs', 'runs_off_bat', 'runs', 'runs_scored', 'runs_total'] if c in df.columns), None)
            if runs_col is not None:
                agg_df = df.groupby(group_cols)[runs_col].sum().reset_index()
                agg_df.rename(columns={runs_col: 'first_inning_total'}, inplace=True)
                target = 'first_inning_total'
                keep = [c for c in group_cols if c != 'match_id']
                agg_df = agg_df[keep + [target]].dropna()
                return agg_df[keep], agg_df[target], target, keep

    raise FileNotFoundError('No IPL matches or ball-by-ball dataset containing runs columns found.')

def _prepare_match_winner(session_id=None):
    # [Function]: _prepare_match_winner
    # [Description]: Implements and executes the  Prepare Match Winner logic within this module pipeline.
    if session_id:
        from utils.helpers import load_dataframe, has_stage_dataframe
        try:
            if has_stage_dataframe(session_id, "cleaned"):
                df = load_dataframe(session_id, "cleaned")
            else:
                df = load_dataframe(session_id, "original")
            
            is_pca = 'PC1' in df.columns and len(df.columns) <= 10
            
            target = 'winner'
            if is_pca or target not in df.columns:
                matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
                if os.path.exists(matches_path):
                    df = pd.read_csv(matches_path, low_memory=False)
            
            if target not in df.columns:
                non_num = df.select_dtypes(exclude=['number']).columns.tolist()
                if non_num:
                    target = non_num[-1]
            if target not in df.columns:
                raise ValueError("Target column 'winner' not found in active dataset")
            keep = [c for c in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision'] if c in df.columns]
            if not keep:
                keep = [c for c in df.columns if c != target]
            df_clean = df[keep + [target]].dropna()
            return df_clean[keep], df_clean[target], target, keep
        except Exception:
            pass
    matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
    if not os.path.exists(matches_path):
        raise FileNotFoundError('ipl_matches.csv not found')
    df = pd.read_csv(matches_path, low_memory=False)
    if 'winner' not in df.columns:
        raise ValueError("'winner' column not found in ipl_matches.csv")
    
    # Exclude season and city based on user's previous preference
    keep = [c for c in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision'] if c in df.columns]
    df = df[keep + ['winner']].dropna()
    return df[keep], df['winner'], 'winner', keep

#  ENSEMBLE REGRESSION 

REG_ALGO_MAP = {
    'rf':  lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    'gb':  lambda: GradientBoostingRegressor(random_state=42),
    'xgb': lambda: XGBRegressor(n_estimators=100, random_state=42),
    'ada': lambda: AdaBoostRegressor(random_state=42),
}

REG_LABELS = {
    'rf':  'Random Forest Regressor',
    'gb':  'Gradient Boosting Regressor',
    'xgb': 'XGBoost Regressor',
    'ada': 'AdaBoost Regressor',
}

def train_and_evaluate_ensemble_reg(algorithms: list, session_id=None) -> dict:
    # [Function]: train_and_evaluate_ensemble_reg
    # [Description]: Implements and executes the Train And Evaluate Ensemble Reg logic within this module pipeline.
    X_raw, y, target_col, feature_cols = _prepare_first_inning_score(session_id)
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()].copy()
    pipeline = EnsembleRegPipeline()
    X_enc = pipeline.fit_transform_features(X_raw)
    pipeline.target_col = target_col

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)
    pipeline.scaler.fit(X_train)
    X_train_sc = pipeline.scaler.transform(X_train)
    X_test_sc  = pipeline.scaler.transform(X_test)

    results = {}
    for algo_key in algorithms:
        if algo_key not in REG_ALGO_MAP:
            continue
        model = REG_ALGO_MAP[algo_key]()
        model.fit(X_train_sc, y_train)
        
        y_pred = model.predict(X_test_sc)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        try:
            cv_scores = cross_val_score(model, X_train_sc, y_train, cv=3, scoring='r2')
            cv_r2 = float(np.mean(cv_scores))
        except Exception:
            cv_r2 = r2

        results[algo_key] = {
            'label': REG_LABELS[algo_key],
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 4),
            'cv_r2': round(cv_r2, 4),
        }
        pipeline.models[algo_key] = model

    best_key = max(results, key=lambda k: results[k]['r2']) if results else None
    pipeline.best_model_key = best_key
    _TRAINED_ENSEMBLE_REG['first_inning_score'] = pipeline

    features_meta = []
    for col in feature_cols:
        if col in pipeline.categorical_cols:
            features_meta.append({'name': col, 'type': 'categorical', 'options': pipeline.category_options[col]})
        else:
            features_meta.append({'name': col, 'type': 'numerical', 
                                   'min': float(X_raw[col].min()), 'max': float(X_raw[col].max()),
                                   'default': float(X_raw[col].mean())})

    return {
        'target_column': target_col,
        'feature_columns': feature_cols,
        'features_metadata': features_meta,
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'results': results,
        'best_model': best_key,
    }

def make_ensemble_reg_prediction(algorithm: str, inputs: dict) -> dict:
    # [Function]: make_ensemble_reg_prediction
    # [Description]: Implements and executes the Make Ensemble Reg Prediction logic within this module pipeline.
    if 'first_inning_score' not in _TRAINED_ENSEMBLE_REG:
        raise ValueError("Ensemble Regression models not trained. Please train first.")
    pipeline = _TRAINED_ENSEMBLE_REG['first_inning_score']
    if algorithm not in pipeline.models:
        raise ValueError(f"Algorithm '{algorithm}' not trained.")
    model = pipeline.models[algorithm]
    row_df = pipeline.transform_input(inputs)
    row_sc = pipeline.scaler.transform(row_df)
    pred = float(model.predict(row_sc)[0])
    return {
        'prediction': round(pred, 2)
    }


#  ENSEMBLE CLASSIFICATION 

CLS_ALGO_MAP = {
    'rf':  lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'gb':  lambda: GradientBoostingClassifier(random_state=42),
    'xgb': lambda: XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'ada': lambda: AdaBoostClassifier(random_state=42),
}

CLS_LABELS = {
    'rf':  'Random Forest Classifier',
    'gb':  'Gradient Boosting Classifier',
    'xgb': 'XGBoost Classifier',
    'ada': 'AdaBoost Classifier',
}

def train_and_evaluate_ensemble_cls(algorithms: list, session_id=None) -> dict:
    # [Function]: train_and_evaluate_ensemble_cls
    # [Description]: Implements and executes the Train And Evaluate Ensemble Cls logic within this module pipeline.
    X_raw, y_raw, target_col, feature_cols = _prepare_match_winner(session_id)
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()].copy()
    pipeline = EnsembleClsPipeline()
    
    # Fit target encoder for multiclass team names
    y_encoded = pipeline.target_encoder.fit_transform(y_raw.astype(str))
    label_names = pipeline.target_encoder.classes_.tolist()
    pipeline.label_names = label_names

    X_enc = pipeline.fit_transform_features(X_raw)
    pipeline.target_col = target_col

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    pipeline.scaler.fit(X_train)
    X_train_sc = pipeline.scaler.transform(X_train)
    X_test_sc  = pipeline.scaler.transform(X_test)

    results = {}
    conf_matrices = {}

    for algo_key in algorithms:
        if algo_key not in CLS_ALGO_MAP:
            continue
        model = CLS_ALGO_MAP[algo_key]()
        model.fit(X_train_sc, y_train)

        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc) if hasattr(model, 'predict_proba') else None

        acc = float(accuracy_score(y_test, y_pred))
        # Use macro average for multiclass metrics
        prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
        # Multiclass ROC AUC requires probability inputs
        try:
            if y_prob is not None:
                roc = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
            else:
                roc = None
        except Exception:
            roc = None

        try:
            cv_scores = cross_val_score(model, X_train_sc, y_train, cv=3, scoring='accuracy')
            cv_acc = float(np.mean(cv_scores))
        except Exception:
            cv_acc = acc

        results[algo_key] = {
            'label': CLS_LABELS[algo_key],
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'roc_auc': round(roc, 4) if roc is not None else None,
            'cv_accuracy': round(cv_acc, 4),
        }
        conf_matrices[algo_key] = confusion_matrix(y_test, y_pred).tolist()
        pipeline.models[algo_key] = model

    best_key = max(results, key=lambda k: results[k]['accuracy']) if results else None
    pipeline.best_model_key = best_key
    _TRAINED_ENSEMBLE_CLS['match_winner'] = pipeline

    features_meta = []
    for col in feature_cols:
        if col in pipeline.categorical_cols:
            features_meta.append({'name': col, 'type': 'categorical', 'options': pipeline.category_options[col]})
        else:
            features_meta.append({'name': col, 'type': 'numerical',
                                   'min': float(X_raw[col].min()), 'max': float(X_raw[col].max()),
                                   'default': float(X_raw[col].mean())})

    return {
        'target_column': target_col,
        'label_names': label_names,
        'feature_columns': feature_cols,
        'features_metadata': features_meta,
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'results': results,
        'confusion_matrices': conf_matrices,
        'best_model': best_key,
    }

def make_ensemble_cls_prediction(algorithm: str, inputs: dict) -> dict:
    # [Function]: make_ensemble_cls_prediction
    # [Description]: Implements and executes the Make Ensemble Cls Prediction logic within this module pipeline.
    if 'match_winner' not in _TRAINED_ENSEMBLE_CLS:
        raise ValueError("Ensemble Classification models not trained. Please train first.")
    pipeline = _TRAINED_ENSEMBLE_CLS['match_winner']
    if algorithm not in pipeline.models:
        raise ValueError(f"Algorithm '{algorithm}' not trained.")
    model = pipeline.models[algorithm]
    row_df = pipeline.transform_input(inputs)
    row_sc = pipeline.scaler.transform(row_df)
    
    pred_encoded = int(model.predict(row_sc)[0])
    pred_label = str(pipeline.target_encoder.inverse_transform([pred_encoded])[0])
    
    proba = model.predict_proba(row_sc)[0].tolist() if hasattr(model, 'predict_proba') else None
    
    return {
        'prediction': pred_encoded,
        'prediction_label': pred_label,
        'probabilities': proba,
        'label_names': pipeline.label_names
    }
