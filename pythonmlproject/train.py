"""
IPL Prediction Model — End-to-End Training Pipeline.

Steps:
  STEP 0: Load + validate CSV
  STEP 1: Elo ratings → elo_ratings.csv
  STEP 2: Feature engineering → features.parquet
  STEP 3: Score prediction features → score_features.parquet
  STEP 4: Train 4 regression models → score_*.joblib
  STEP 5: Train 11 classification models → match_winner_*.joblib
  STEP 6: K-Means(k=6) + DBSCAN + PCA clustering → clusters.csv
  STEP 7: Compute SHAP values → shap_values.npy + print summary

Usage:
  py train.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Allow imports from project root (src/ lives here)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import shap

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def _step(n: int, msg: str) -> float:
    log.info("=" * 60)
    log.info(f"STEP {n}: {msg}")
    log.info("=" * 60)
    return time.time()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IPL Training Pipeline")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    t_total = time.time()

    # Resolve paths
    data_raw  = PROJECT_ROOT / "data" / "raw"
    data_proc = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    logs_dir   = PROJECT_ROOT / "logs"
    data_proc.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import src modules so TF env var is set first
    from src.data.loader import load_config, load_raw, make_match_df
    from src.data.features import (
        build_features,
        build_score_features,
        build_sequence_features,
        build_player_features,
        CLASSIF_FEATURES,
        SCORE_FEATURES,
        SEQ_LEN,
    )
    from src.data.elo import compute_elo, save_elo
    from src.models import match_winner, score_predictor

    # -----------------------------------------------------------------------
    # STEP 0: Load + validate CSV
    # -----------------------------------------------------------------------
    t = _step(0, "Load + validate CSV")
    cfg = load_config()
    raw = load_raw(cfg)
    log.info(f"Raw shape: {raw.shape}")
    assert len(raw) > 0, "CSV is empty!"
    log.info(f"Columns present: {list(raw.columns[:10])} ...")
    log.info(f"STEP 0 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 1: Elo ratings → elo_ratings.csv  (before features so we can join)
    # -----------------------------------------------------------------------
    t = _step(1, "Elo ratings → elo_ratings.csv")
    match_df = make_match_df(raw, cfg)
    log.info(f"match_df shape: {match_df.shape}  (after filtering no-results/ties)")
    elo_df = compute_elo(match_df, cfg)
    elo_path = data_proc / "elo_ratings.csv"
    save_elo(elo_df, str(elo_path))
    log.info(f"elo_ratings.csv → {elo_path}  rows={len(elo_df)}")
    log.info(f"STEP 1 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 2: Feature engineering → features.parquet
    # -----------------------------------------------------------------------
    t = _step(2, "Feature engineering → features.parquet")
    features_df, label_encoders = build_features(match_df, elo_df, cfg)
    feat_path = data_proc / "features.parquet"
    features_df.to_parquet(feat_path, index=False)
    log.info(f"features.parquet → {feat_path}  shape={features_df.shape}")
    log.info(f"STEP 2 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 3: Score prediction features → score_features.parquet
    # -----------------------------------------------------------------------
    t = _step(3, "Score prediction features → score_features.parquet")
    score_df = build_score_features(raw, match_df, cfg)
    score_feat_path = data_proc / "score_features.parquet"
    score_df.to_parquet(score_feat_path, index=False)
    log.info(f"score_features.parquet → {score_feat_path}  shape={score_df.shape}")
    log.info(f"STEP 3 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 4: Train 4 regression models → score_*.joblib
    # -----------------------------------------------------------------------
    t = _step(4, "Train 4 regression models → score_*.joblib")
    X_score = score_df[SCORE_FEATURES].values
    y_score = score_df["final_score"].values
    X_str, X_ste, y_str, y_ste = train_test_split(
        X_score, y_score, test_size=0.2, random_state=42
    )
    reg_models = score_predictor.train_all(X_str, y_str, cfg)
    reg_metrics_df = score_predictor.evaluate_all(reg_models, X_ste, y_ste)
    score_predictor.save_all(reg_models, str(models_dir), SCORE_FEATURES)
    log.info("\nRegression metrics:\n" + reg_metrics_df.to_string(index=False))
    log.info(f"STEP 4 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 5: Train 11 classification models → match_winner_*.joblib
    # -----------------------------------------------------------------------
    t = _step(5, "Train 11 classification models → match_winner_*.joblib")
    X_clf = features_df[CLASSIF_FEATURES].values
    y_clf = features_df["team1_won"].values.astype(int)
    X_ctr, X_cte, y_ctr, y_cte = train_test_split(
        X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
    )
    clf_models, clf_scaler = match_winner.train_all(X_ctr, y_ctr, cfg)
    clf_metrics = match_winner.evaluate_all(clf_models, X_cte, y_cte, clf_scaler)
    match_winner.save_all(
        clf_models, str(models_dir), clf_scaler, CLASSIF_FEATURES, label_encoders
    )
    log.info("\nClassification metrics:\n" + clf_metrics.to_string(index=False))
    log.info(f"STEP 5 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 6: K-Means(k=6) + DBSCAN + PCA → clusters.csv
    # -----------------------------------------------------------------------
    t = _step(6, "KMeans(k=6) + DBSCAN + PCA → clusters.csv")
    X_player, player_names = build_player_features(elo_df)

    # StandardScaler for clustering (K-Means is distance-based)
    p_scaler = StandardScaler()
    Xp_scaled = p_scaler.fit_transform(X_player)

    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    km_labels = km.fit_predict(Xp_scaled)

    db = DBSCAN(eps=0.5, min_samples=5)
    db_labels = db.fit_predict(Xp_scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(Xp_scaled)

    clusters_df = elo_df[["player_name", "elo_rating", "match_count", "pom_count", "win_rate"]].copy()
    clusters_df["kmeans_cluster"] = km_labels
    clusters_df["dbscan_cluster"] = db_labels
    clusters_df["pca1"]           = pca_coords[:, 0]
    clusters_df["pca2"]           = pca_coords[:, 1]
    # Backward-compatible aliases used by earlier versions of the app
    clusters_df["cluster"]        = km_labels
    clusters_df["dbscan_label"]   = db_labels
    clusters_df["pc1"]            = pca_coords[:, 0]
    clusters_df["pc2"]            = pca_coords[:, 1]
    clusters_path = data_proc / "clusters.csv"
    clusters_df.to_csv(clusters_path, index=False)
    log.info(f"clusters.csv → {clusters_path}  rows={len(clusters_df)}")
    log.info(f"KMeans cluster sizes: {pd.Series(km_labels).value_counts().sort_index().to_dict()}")
    log.info(f"DBSCAN outliers (label=-1): {(db_labels == -1).sum()}")
    log.info(f"STEP 6 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # STEP 7: SHAP values → shap_values.npy
    # -----------------------------------------------------------------------
    t = _step(7, "Compute SHAP values → shap_values.npy")
    try:
        shap_vals, expected_val = match_winner.get_shap_values(
            clf_models, X_cte, CLASSIF_FEATURES
        )
        shap_path = models_dir / "shap_values.npy"
        np.save(str(shap_path), shap_vals)
        log.info(f"shap_values.npy saved → {shap_path}")

        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_summary = pd.DataFrame({
            "feature": CLASSIF_FEATURES,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        log.info("\nSHAP Feature Importance (XGBoost):\n" + shap_summary.to_string(index=False))
    except Exception as exc:
        log.warning(f"SHAP step skipped: {exc}")
        log.info(f"STEP 7 done in {_elapsed(t)}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)

    log.info("\nClassification Summary:")
    log.info(clf_metrics[["model", "accuracy", "auc", "f1"]].to_string(index=False))

    log.info("\nRegression Summary:")
    log.info(reg_metrics_df[["model", "mae", "r2"]].to_string(index=False))

    log.info(f"\nTotal training time: {_elapsed(t_total)}")
    log.info("\nRun: streamlit run app.py")


if __name__ == "__main__":
    main()
