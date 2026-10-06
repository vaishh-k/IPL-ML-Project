"""IPL Prediction Model — Streamlit Entry Point."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


import streamlit as st
st.set_page_config(
    page_title="IPL Project",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.helpers import apply_theme, load_dataset_stats


st.markdown(apply_theme(), unsafe_allow_html=True)

stats = load_dataset_stats()
n_matches = stats["n_matches"]
n_seasons = stats["n_seasons"]

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #1F3864 0%, #0d1f3c 60%, #7C3AED 100%);
        border-radius: 16px;
        padding: 48px 40px 40px 40px;
        margin-bottom: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    ">
        <h1 style="color:#FF6B00; font-size:3rem; margin:0 0 8px 0; letter-spacing:2px;">
            🏏 IPL Prediction Model
        </h1>
        <p style="color:#e0e0e0; font-size:1.3rem; margin:0; letter-spacing:1px;">
            2008–2026 &nbsp;·&nbsp; {n_matches:,} Matches &nbsp;·&nbsp; 25 Algorithms
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Metric columns
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🗓️ Seasons", str(n_seasons), help="IPL seasons covered in dataset")
with c2:
    st.metric("🏟️ Matches", f"{n_matches:,}", help="Total IPL matches in processed dataset")
with c3:
    st.metric("🤖 Algorithms", "25", help="ML, clustering & analytics algorithms")
with c4:
    st.metric("🧠 Analytics", "ML + Elo", help="Prediction, Elo ratings & clustering workflows")

st.divider()

# ---------------------------------------------------------------------------
# Page directory table
# ---------------------------------------------------------------------------
st.subheader("📑 App Pages")

pages = [
    ("🏠", "Home",             "Dataset overview — seasons, venues, toss analysis, champions"),
    ("📜", "History & EDA",    "Filter matches by season · H2H · score distributions · heatmaps"),
    ("🎯", "Match Prediction", "11 classifiers · win probability · SHAP · ROC-AUC · confusion matrix"),
    ("📈", "Score Prediction", "4 regressors (Linear · Ridge · Lasso · XGB) · over-by-over forecast"),
    ("👤", "Player Analytics", "Elo ratings · KMeans/DBSCAN clusters · PCA scatter plot"),
    ("🏆", "Team Optimizer",   "Best XI selector · simulate tournament brackets"),
    ("🔬", "Algo Lab",         "Live hyperparameter tuning · GridSearch · RandomSearch · cross-val"),
]

header_cols = st.columns([1, 3, 7])
header_cols[0].markdown("**Icon**")
header_cols[1].markdown("**Page**")
header_cols[2].markdown("**Description**")
st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

for icon, name, desc in pages:
    cols = st.columns([1, 3, 7])
    cols[0].markdown(icon)
    cols[1].markdown(f"**{name}**")
    cols[2].markdown(desc)

st.divider()

# ---------------------------------------------------------------------------
# Algorithm coverage
# ---------------------------------------------------------------------------
st.subheader("📋 Algorithm Coverage (25 Total)")

algo_cols = st.columns(3)

with algo_cols[0]:
    st.markdown(
        """
        **Regression (4)**
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - XGBoost Regressor

        **Classification (11)**
        - Logistic Regression
        - K-Nearest Neighbours
        - Naïve Bayes
        - Support Vector Machine
        - Decision Tree
        - Bagging Classifier
        - Random Forest
        - AdaBoost
        - Gradient Boosting
        - XGBoost Classifier
        - Stacking Ensemble
        """
    )

with algo_cols[1]:
    st.markdown(
        """
        **Clustering (3)**
        - K-Means (k=6)
        - DBSCAN
        - PCA (2-component)
        """
    )

with algo_cols[2]:
    st.markdown(
        """
        **Analytics & Optimization (7)**
        - SHAP Explainability
        - Stratified K-Fold CV
        - Feature Importance
        - Elo Rating System
        - Matchup Score Engine
        - Team XI Optimizer
        - Tournament Simulator
        """
    )

st.divider()
st.caption(
    "Built with Streamlit · scikit-learn · XGBoost · Plotly · SHAP  "
    "| Data: IPL 2008–2026"
)
