# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: regression_service.py
# Path: backend/services/regression_service.py
# Description: Service layer executing regression pipeline steps: preprocessing, scaling, training, evaluation, and inference.
# ==============================================================================

import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

ALGORITHM_MAP = {
    'linear':        lambda: LinearRegression(),
    'ridge':         lambda: Ridge(alpha=1.0),
    'lasso':         lambda: Lasso(alpha=0.1, max_iter=5000),
    'decision_tree': lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
}

ALGORITHM_LABELS = {
    'linear':        'Linear Regression',
    'ridge':         'Ridge Regression',
    'lasso':         'Lasso Regression',
    'decision_tree': 'Decision Tree',
}

# In-memory storage for trained models, encoders, and scalers
_TRAINED_MODELS = {}

class FitPipeline:
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

def _first_inning_from_bbb_raw(bbb_path):
    # [Function]: _first_inning_from_bbb_raw
    # [Description]: Implements and executes the  First Inning From Bbb Raw logic within this module pipeline.
    df = pd.read_csv(bbb_path, low_memory=False)
    if 'innings' in df.columns:
        df = df[df['innings'] == 1]
    group_cols = [c for c in ['match_id', 'batting_team', 'bowling_team', 'venue'] if c in df.columns]
    runs_col = next((c for c in ['total_runs', 'runs_off_bat', 'runs'] if c in df.columns), None)
    if runs_col is None:
        raise ValueError('Cannot identify a runs column.')
    agg_df = df.groupby(group_cols)[runs_col].sum().reset_index()
    agg_df.rename(columns={runs_col: 'first_inning_total'}, inplace=True)
    target = 'first_inning_total'
    keep = [c for c in group_cols if c != 'match_id']
    agg_df = agg_df[keep + [target]].dropna()
    return agg_df.drop(columns=[target]), agg_df[target], target, keep

def _prepare_first_inning_score_raw():
    # [Function]: _prepare_first_inning_score_raw
    # [Description]: Implements and executes the  Prepare First Inning Score Raw logic within this module pipeline.
    matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
    bbb_path = os.path.join(DATA_DIR, 'processed_ipl_ball_by_ball.csv')
    if os.path.exists(matches_path):
        df = pd.read_csv(matches_path, low_memory=False)
        target = None
        for c in ['target_runs', 'runs_scored', 'first_ings_runs', 'score', 'team1_runs']:
            if c in df.columns:
                target = c
                break
        if target is None:
            if os.path.exists(bbb_path):
                return _first_inning_from_bbb_raw(bbb_path)
            raise ValueError('No target column found in ipl_matches.csv')
        keep = [c for c in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision'] if c in df.columns]
        df = df[keep + [target]].dropna()
        return df.drop(columns=[target]), df[target], target, keep
    elif os.path.exists(bbb_path):
        return _first_inning_from_bbb_raw(bbb_path)
    raise FileNotFoundError('No IPL dataset found in data/')

def _prepare_batsman_score_raw():
    # [Function]: _prepare_batsman_score_raw
    # [Description]: Implements and executes the  Prepare Batsman Score Raw logic within this module pipeline.
    bbb_path = os.path.join(DATA_DIR, 'processed_ipl_ball_by_ball.csv')
    if not os.path.exists(bbb_path):
        raise FileNotFoundError('processed_ipl_ball_by_ball.csv not found')
    df = pd.read_csv(bbb_path, low_memory=False)
    target = None
    for c in ['batter_runs', 'batsman_runs', 'runs_off_bat']:
        if c in df.columns:
            target = c
            break
    if target is None:
        raise ValueError('Cannot find batsman runs column.')
    keep = [c for c in ['innings', 'over', 'ball_in_over', 'ball_number', 'batting_team', 'bowling_team', 'batter', 'bowler', 'striker', 'non_striker'] if c in df.columns]
    df = df[keep + [target]].dropna()
    if len(df) > 60000:
        df = df.sample(60000, random_state=42)
    return df.drop(columns=[target]), df[target], target, keep

def prepare_data_raw(problem_type, session_id=None):
    # [Function]: prepare_data_raw
    # [Description]: Implements and executes the Prepare Data Raw logic within this module pipeline.
    if session_id:
        from utils.helpers import load_dataframe, has_stage_dataframe
        try:
            if has_stage_dataframe(session_id, "cleaned"):
                df = load_dataframe(session_id, "cleaned")
            else:
                df = load_dataframe(session_id, "original")
            
            # Check if df is a PCA-reduced dataset (which doesn't contain actual features/target)
            is_pca = 'PC1' in df.columns and len(df.columns) <= 10
            
            if problem_type == 'first_inning_score':
                target = None
                for c in ['target_runs', 'runs_scored', 'first_ings_runs', 'score', 'first_inning_total', 'team1_runs']:
                    if c in df.columns:
                        target = c
                        break
                
                # Check if it is ball-by-ball raw or selected table
                is_bbb = any(c in df.columns for c in ['runs_off_bat', 'total_runs', 'extra_runs'])
                
                if is_pca or (target is None and not is_bbb):
                    # fallback to raw ipl_matches.csv
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
                    return df_clean.drop(columns=[target]), df_clean[target], target, keep
                
                if target is None:
                    # check case-insensitively for run/score
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
                return df_clean.drop(columns=[target]), df_clean[target], target, keep
                
            elif problem_type == 'batsman_score':
                target = None
                for c in ['batter_runs', 'batsman_runs', 'runs_off_bat']:
                    if c in df.columns:
                        target = c
                        break
                if is_pca or target is None:
                    bbb_raw = os.path.join(DATA_DIR, 'ipl_ball_by_ball.csv')
                    if os.path.exists(bbb_raw):
                        df = pd.read_csv(bbb_raw, low_memory=False)
                        target = next((c for c in ['batter_runs', 'batsman_runs', 'runs_off_bat'] if c in df.columns), None)
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
                keep = [c for c in ['innings', 'over', 'ball_in_over', 'ball_number', 'batting_team', 'bowling_team', 'batter', 'bowler', 'striker', 'non_striker'] if c in df.columns]
                if not keep:
                    keep = [c for c in df.columns if c != target]
                df_clean = df[keep + [target]].dropna()
                if len(df_clean) > 60000:
                    df_clean = df_clean.sample(60000, random_state=42)
                return df_clean.drop(columns=[target]), df_clean[target], target, keep
        except Exception:
            pass

    if problem_type == 'first_inning_score':
        return _prepare_first_inning_score_raw()
    elif problem_type == 'batsman_score':
        return _prepare_batsman_score_raw()
    raise ValueError(f'Unknown problem_type: {problem_type}')

def train_and_evaluate(problem_type, algorithms, session_id=None):
    # [Function]: train_and_evaluate
    # [Description]: Implements and executes the Train And Evaluate logic within this module pipeline.
    X_raw, y, target_col, feature_cols = prepare_data_raw(problem_type, session_id)
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()].copy()
    
    pipeline = FitPipeline()
    X_encoded = pipeline.fit_transform_features(X_raw)
    pipeline.target_col = target_col
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Fit scaling parameters on entire encoded space
    pipeline.scaler.fit(X_train)
    X_train_sc = pipeline.scaler.transform(X_train)
    X_test_sc  = pipeline.scaler.transform(X_test)
    
    results = {}
    predictions_map = {}
    for algo_key in algorithms:
        if algo_key not in ALGORITHM_MAP:
            continue
        model = ALGORITHM_MAP[algo_key]()
        label = ALGORITHM_LABELS[algo_key]
        if algo_key in ('linear', 'ridge', 'lasso'):
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        pipeline.models[algo_key] = model
        
        mae  = float(mean_absolute_error(y_test, y_pred))
        mse  = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_test, y_pred))
        try:
            cv_scores = cross_val_score(model, X_train_sc if algo_key in ('linear','ridge','lasso') else X_train, y_train, cv=3, scoring='r2')
            cv_r2 = float(np.mean(cv_scores))
        except Exception:
            cv_r2 = r2
            
        results[algo_key] = {
            'label': label,
            'mae': round(mae,4),
            'mse': round(mse,4),
            'rmse': round(rmse,4),
            'r2': round(r2,4),
            'cv_r2': round(cv_r2,4)
        }
        n_s = min(500, len(y_test))
        idx = np.random.choice(len(y_test), n_s, replace=False)
        predictions_map[algo_key] = {
            'actual': list(np.array(y_test)[idx].astype(float)),
            'predicted': list(y_pred[idx].astype(float))
        }
        
    best_key = max(results, key=lambda k: results[k]['r2']) if results else None
    pipeline.best_model_key = best_key
    
    _TRAINED_MODELS[problem_type] = pipeline
    
    # Construct metadata details for realtime user interface
    features_meta = []
    for col in feature_cols:
        if col in pipeline.categorical_cols:
            features_meta.append({
                'name': col,
                'type': 'categorical',
                'options': pipeline.category_options[col]
            })
        else:
            features_meta.append({
                'name': col,
                'type': 'numerical',
                'min': float(X_raw[col].min()),
                'max': float(X_raw[col].max()),
                'default': float(X_raw[col].mean())
            })
            
    return {
        'problem_type': problem_type,
        'target_column': target_col,
        'feature_columns': list(feature_cols),
        'features_metadata': features_meta,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'results': results,
        'predictions': predictions_map,
        'best_model': best_key
    }

def make_prediction(problem_type: str, algorithm: str, inputs: dict) -> float:
    # [Function]: make_prediction
    # [Description]: Implements and executes the Make Prediction logic within this module pipeline.
    if problem_type not in _TRAINED_MODELS:
        raise ValueError("Model is not trained. Please train the regression model first.")
    pipeline = _TRAINED_MODELS[problem_type]
    if algorithm not in pipeline.models:
        raise ValueError(f"Algorithm '{algorithm}' has not been trained.")
    model = pipeline.models[algorithm]
    row_df = pipeline.transform_input(inputs)
    if algorithm in ('linear', 'ridge', 'lasso'):
        row_scaled = pipeline.scaler.transform(row_df)
        pred = model.predict(row_scaled)[0]
    else:
        pred = model.predict(row_df)[0]
    return float(pred)
