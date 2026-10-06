# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: classification_service.py
# Path: backend/services/classification_service.py
# Description: Service layer executing classification pipeline steps: preprocessing, scaling, training, evaluation, and inference.
# ==============================================================================

import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

ALGORITHM_MAP = {
    'logistic':      lambda: LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree': lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
    'svm':           lambda: SVC(kernel='rbf', probability=True, random_state=42),
    'naive_bayes':   lambda: GaussianNB(),
}
ALGORITHM_LABELS = {
    'logistic':      'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'svm':           'SVM (RBF Kernel)',
    'naive_bayes':   'Naive Bayes',
}
NEEDS_SCALING = {'logistic', 'svm', 'naive_bayes'}

_TRAINED_CLASSIFIERS = {}

class ClassificationPipeline:
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

# -- Problem data preparers ----------------------------------------------------

def _prepare_defend_target():
    # [Function]: _prepare_defend_target
    # [Description]: Implements and executes the  Prepare Defend Target logic within this module pipeline.
    matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
    if not os.path.exists(matches_path):
        raise FileNotFoundError('ipl_matches.csv not found')
    df = pd.read_csv(matches_path, low_memory=False)
    if 'winner' not in df.columns:
        raise ValueError("'winner' column not found in ipl_matches.csv")
    feature_cols = [c for c in ['venue', 'team1', 'team2',
                                 'toss_winner', 'toss_decision'] if c in df.columns]
    if 'toss_decision' not in df.columns:
        raise ValueError("'toss_decision' column required")
    df = df[feature_cols + ['winner']].dropna()

    def is_defended(row):
        # [Function]: is_defended
        # [Description]: Implements and executes the Is Defended logic within this module pipeline.
        if row['toss_winner'] == row['team1']:
            batting_first = row['team1'] if row['toss_decision'] in ['bat', 'batting'] else row['team2']
        else:
            batting_first = row['team2'] if row['toss_decision'] in ['bat', 'batting'] else row['team1']
        return 1 if row['winner'] == batting_first else 0

    df['defended'] = df.apply(is_defended, axis=1)
    return df[feature_cols], df['defended'], 'defended', feature_cols, ['Chased', 'Defended']


def _prepare_batsman_fifty():
    # [Function]: _prepare_batsman_fifty
    # [Description]: Implements and executes the  Prepare Batsman Fifty logic within this module pipeline.
    bbb_path = os.path.join(DATA_DIR, 'processed_ipl_ball_by_ball.csv')
    if not os.path.exists(bbb_path):
        raise FileNotFoundError('processed_ipl_ball_by_ball.csv not found')
    df = pd.read_csv(bbb_path, low_memory=False)
    batter_col = next((c for c in ['batter', 'batsman', 'striker'] if c in df.columns), None)
    match_col  = next((c for c in ['match_id', 'id'] if c in df.columns), None)
    runs_col   = next((c for c in ['batter_runs', 'batsman_runs', 'runs_off_bat'] if c in df.columns), None)
    if not all([batter_col, match_col, runs_col]):
        raise ValueError("Cannot find required columns for batsman 50+ problem")
    group_cols = [match_col, batter_col]
    agg = df.groupby(group_cols)[runs_col].sum().reset_index()
    agg.rename(columns={runs_col: 'total_runs'}, inplace=True)
    agg['scored_fifty'] = (agg['total_runs'] >= 50).astype(int)
    extra_cols = [c for c in ['batting_team', 'bowling_team', 'venue', 'innings'] if c in df.columns]
    if extra_cols:
        extra = df.groupby(group_cols)[extra_cols].first().reset_index()
        agg = agg.merge(extra, on=group_cols, how='left')
    feature_cols = [batter_col] + [c for c in extra_cols if c in agg.columns]
    agg = agg[feature_cols + ['scored_fifty']].dropna()
    if len(agg) > 50000:
        agg = agg.sample(50000, random_state=42)
    return agg[feature_cols], agg['scored_fifty'], 'scored_fifty', feature_cols, ['Below 50', '50+']


def prepare_data_cls(problem_type, session_id=None):
    # [Function]: prepare_data_cls
    # [Description]: Implements and executes the Prepare Data Cls logic within this module pipeline.
    if session_id:
        from utils.helpers import load_dataframe, has_stage_dataframe
        try:
            if has_stage_dataframe(session_id, "cleaned"):
                df = load_dataframe(session_id, "cleaned")
            else:
                df = load_dataframe(session_id, "original")
            
            is_pca = 'PC1' in df.columns and len(df.columns) <= 10
            
            if problem_type == 'defend_target':
                target = 'winner'
                if is_pca or target not in df.columns:
                    matches_path = os.path.join(DATA_DIR, 'ipl_matches.csv')
                    if os.path.exists(matches_path):
                        df = pd.read_csv(matches_path, low_memory=False)
                
                if target not in df.columns:
                    non_num = df.select_dtypes(exclude=['number']).columns.tolist()
                    if non_num:
                        target = non_num[-1]
                feature_cols = [c for c in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision'] if c in df.columns]
                if not feature_cols:
                    feature_cols = [c for c in df.columns if c != target]
                df_clean = df[feature_cols + [target]].dropna()
                
                def is_defended(row):
                    t_win = row.get('toss_winner')
                    t_dec = row.get('toss_decision', 'field')
                    t1 = row.get('team1')
                    t2 = row.get('team2')
                    if t_win == t1:
                        batting_first = t1 if t_dec in ['bat', 'batting'] else t2
                    else:
                        batting_first = t2 if t_dec in ['bat', 'batting'] else t1
                    return 1 if row[target] == batting_first else 0

                df_clean['defended'] = df_clean.apply(is_defended, axis=1)
                return df_clean[feature_cols], df_clean['defended'], 'defended', feature_cols, ['Chased', 'Defended']
                
            elif problem_type == 'batsman_fifty':
                batter_col = next((c for c in ['batter', 'batsman', 'striker'] if c in df.columns), None)
                match_col  = next((c for c in ['match_id', 'id'] if c in df.columns), None)
                runs_col   = next((c for c in ['batter_runs', 'batsman_runs', 'runs_off_bat'] if c in df.columns), None)
                
                if is_pca or not all([batter_col, match_col, runs_col]):
                    bbb_raw = os.path.join(DATA_DIR, 'ipl_ball_by_ball.csv')
                    if os.path.exists(bbb_raw):
                        df = pd.read_csv(bbb_raw, low_memory=False)
                        batter_col = next((c for c in ['batter', 'batsman', 'striker'] if c in df.columns), None)
                        match_col  = next((c for c in ['match_id', 'id'] if c in df.columns), None)
                        runs_col   = next((c for c in ['batter_runs', 'batsman_runs', 'runs_off_bat'] if c in df.columns), None)
                
                if not all([batter_col, match_col, runs_col]):
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    runs_col = num_cols[-1] if num_cols else None
                    batter_col = df.columns[0]
                    match_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                group_cols = [match_col, batter_col]
                agg = df.groupby(group_cols)[runs_col].sum().reset_index()
                agg.rename(columns={runs_col: 'total_runs'}, inplace=True)
                agg['scored_fifty'] = (agg['total_runs'] >= 50).astype(int)
                extra_cols = [c for c in ['batting_team', 'bowling_team', 'venue', 'innings'] if c in df.columns]
                if extra_cols:
                    extra = df.groupby(group_cols)[extra_cols].first().reset_index()
                    agg = agg.merge(extra, on=group_cols, how='left')
                feature_cols = [batter_col] + [c for c in extra_cols if c in agg.columns]
                agg = agg[feature_cols + ['scored_fifty']].dropna()
                if len(agg) > 50000:
                    agg = agg.sample(50000, random_state=42)
                return agg[feature_cols], agg['scored_fifty'], 'scored_fifty', feature_cols, ['Below 50', '50+']
        except Exception:
            pass

    if problem_type == 'defend_target':
        return _prepare_defend_target()
    elif problem_type == 'batsman_fifty':
        return _prepare_batsman_fifty()
    raise ValueError(f'Unknown problem_type: {problem_type}')


def train_and_evaluate_cls(problem_type, algorithms, session_id=None):
    # [Function]: train_and_evaluate_cls
    # [Description]: Implements and executes the Train And Evaluate Cls logic within this module pipeline.
    X_raw, y, target_col, feature_cols, label_names = prepare_data_cls(problem_type, session_id)
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()].copy()
    pipeline = ClassificationPipeline()
    X_enc = pipeline.fit_transform_features(X_raw)
    pipeline.target_col = target_col
    pipeline.label_names = label_names

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.scaler.fit(X_train)
    X_train_sc = pipeline.scaler.transform(X_train)
    X_test_sc  = pipeline.scaler.transform(X_test)

    results = {}
    conf_matrices = {}

    for algo_key in algorithms:
        if algo_key not in ALGORITHM_MAP:
            continue
        model = ALGORITHM_MAP[algo_key]()
        label = ALGORITHM_LABELS[algo_key]
        uses_sc = algo_key in NEEDS_SCALING

        if uses_sc:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        pipeline.models[algo_key] = model

        acc  = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec  = float(recall_score(y_test, y_pred, zero_division=0))
        f1   = float(f1_score(y_test, y_pred, zero_division=0))
        roc  = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
        try:
            cv_scores = cross_val_score(model, X_train_sc if uses_sc else X_train, y_train, cv=3, scoring='accuracy')
            cv_acc = float(np.mean(cv_scores))
        except Exception:
            cv_acc = acc

        results[algo_key] = {
            'label': label,
            'accuracy':    round(acc, 4),
            'precision':   round(prec, 4),
            'recall':      round(rec, 4),
            'f1':          round(f1, 4),
            'roc_auc':     round(roc, 4) if roc is not None else None,
            'cv_accuracy': round(cv_acc, 4),
        }
        conf_matrices[algo_key] = confusion_matrix(y_test, y_pred).tolist()

    best_key = max(results, key=lambda k: results[k]['accuracy']) if results else None
    pipeline.best_model_key = best_key
    _TRAINED_CLASSIFIERS[problem_type] = pipeline

    features_meta = []
    for col in feature_cols:
        if col in pipeline.categorical_cols:
            features_meta.append({'name': col, 'type': 'categorical',
                                   'options': pipeline.category_options[col]})
        else:
            features_meta.append({'name': col, 'type': 'numerical',
                                   'min': float(X_raw[col].min()),
                                   'max': float(X_raw[col].max()),
                                   'default': float(X_raw[col].mean())})

    return {
        'problem_type': problem_type,
        'target_column': target_col,
        'label_names': label_names,
        'feature_columns': list(feature_cols),
        'features_metadata': features_meta,
        'train_size': int(len(X_train)),
        'test_size':  int(len(X_test)),
        'results': results,
        'confusion_matrices': conf_matrices,
        'best_model': best_key,
    }


def make_classification_prediction(problem_type: str, algorithm: str, inputs: dict):
    # [Function]: make_classification_prediction
    # [Description]: Implements and executes the Make Classification Prediction logic within this module pipeline.
    if problem_type not in _TRAINED_CLASSIFIERS:
        raise ValueError("Model not trained. Please train first.")
    pipeline = _TRAINED_CLASSIFIERS[problem_type]
    if algorithm not in pipeline.models:
        raise ValueError(f"Algorithm '{algorithm}' not trained.")
    model = pipeline.models[algorithm]
    row_df = pipeline.transform_input(inputs)
    uses_sc = algorithm in NEEDS_SCALING
    if uses_sc:
        row_sc = pipeline.scaler.transform(row_df)
        pred  = int(model.predict(row_sc)[0])
        proba = model.predict_proba(row_sc)[0].tolist() if hasattr(model, 'predict_proba') else None
    else:
        pred  = int(model.predict(row_df)[0])
        proba = model.predict_proba(row_df)[0].tolist() if hasattr(model, 'predict_proba') else None
    label_names = pipeline.label_names
    pred_label  = label_names[pred] if pred < len(label_names) else str(pred)
    return {'prediction': pred, 'prediction_label': pred_label,
            'probabilities': proba, 'label_names': label_names}
