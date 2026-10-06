"""🎯 Match Prediction — Win Probability (11 Classifiers)."""
from __future__ import annotations

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import streamlit as st
st.set_page_config(
    page_title="IPL Project",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.utils.helpers import apply_theme, get_project_root, load_model_safe, get_team_color

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.markdown(apply_theme(), unsafe_allow_html=True)

IPL_THEME = {"navy": "#1F3864", "orange": "#FF6B00", "purple": "#7C3AED"}

PROJ = get_project_root()
MODELS_DIR = PROJ / "models"
PROC_DIR = PROJ / "data" / "processed"
FEAT_PATH = PROC_DIR / "features.parquet"
ELO_PATH = PROC_DIR / "elo_ratings.csv"

CLASSIF_FEATURES = [
    "team1_enc", "team2_enc", "toss_winner_enc", "toss_bat_first",
    "toss_winner_is_team1", "venue_enc", "season_idx",
    "team1_h2h_wins", "venue_team1_wins", "team1_elo", "team2_elo",
]

CLASSIF_NAMES = ["lr", "knn", "nb", "svm", "dt", "bag", "rf", "ada", "gb", "xgb", "stack"]
SCALE_MODELS = {"lr", "knn", "svm"}

# Full names for display
MODEL_LABELS = {
    "lr": "Logistic Regression",
    "knn": "K-Nearest Neighbours",
    "nb": "Naïve Bayes",
    "svm": "Support Vector Machine",
    "dt": "Decision Tree",
    "bag": "Bagging Classifier",
    "rf": "Random Forest",
    "ada": "AdaBoost",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost Classifier",
    "stack": "Stacking Ensemble",
}

ALGO_FORMULAS = {
    "lr": (
        r"\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b),\quad \sigma(z)=\frac{1}{1+e^{-z}}",
        "C=1.0, solver=lbfgs, max_iter=1000",
    ),
    "knn": (
        r"d(\mathbf{x},\mathbf{x}') = \sqrt{\sum_{i=1}^n (x_i - x'_i)^2}",
        "n_neighbors=7, metric=euclidean",
    ),
    "nb": (
        r"P(C|\mathbf{x}) \propto P(C)\prod_{i=1}^n P(x_i|C)",
        "var_smoothing=1e-9",
    ),
    "svm": (
        r"\min_{\mathbf{w},b}\frac{1}{2}\|\mathbf{w}\|^2\;s.t.\;y_i(\mathbf{w}^\top\mathbf{x}_i+b)\geq 1",
        "kernel=rbf, C=1.0, gamma=scale",
    ),
    "dt": (
        r"H(S) = -\sum_c p_c \log_2 p_c \quad (\text{Gini impurity})",
        "max_depth=5, criterion=gini, min_samples_split=20",
    ),
    "bag": (
        r"\hat{f}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B f_b(\mathbf{x})",
        "n_estimators=200, base=DecisionTree",
    ),
    "rf": (
        r"\hat{y} = \text{mode}\{f_b(\mathbf{x})\}_{b=1}^B,\; f_b \text{ trained on bootstrap sample}",
        "n_estimators=200, max_depth=10, min_samples_split=5",
    ),
    "ada": (
        r"F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \alpha_m h_m(\mathbf{x})",
        "n_estimators=100, learning_rate=0.1",
    ),
    "gb": (
        r"F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) - \eta\nabla_F L(y,F_{m-1}(\mathbf{x}))",
        "n_estimators=200, max_depth=4, learning_rate=0.05",
    ),
    "xgb": (
        r"\mathcal{L}(\phi)=\sum_i l(\hat{y}_i,y_i)+\sum_k\Omega(f_k),\; \Omega(f)=\gamma T+\frac{1}{2}\lambda\|w\|^2",
        "n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8",
    ),
    "stack": (
        r"\hat{y} = g\bigl([f_1(\mathbf{x}),\;f_2(\mathbf{x}),\;f_3(\mathbf{x})]\bigr)",
        "base=[LR, RF, GB], meta=XGBoost, cv=5",
    ),
}

# ---------------------------------------------------------------------------
# Resource loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifiers():
    """Load all 11 classification models + scaler + label encoders."""
    result = {
        "models": {},
        "scaler": None,
        "label_encoders": {},
        "feat_cols": CLASSIF_FEATURES,
    }
    for name in CLASSIF_NAMES:
        p = MODELS_DIR / f"match_winner_{name}.joblib"
        m = load_model_safe(str(p))
        if m is not None:
            result["models"][name] = m

    scaler_p = MODELS_DIR / "classif_scaler.joblib"
    if scaler_p.exists():
        result["scaler"] = joblib.load(scaler_p)

    le_p = MODELS_DIR / "label_encoders.joblib"
    if le_p.exists():
        result["label_encoders"] = joblib.load(le_p)

    return result


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    if FEAT_PATH.exists():
        return pd.read_parquet(FEAT_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_elo() -> pd.DataFrame:
    if ELO_PATH.exists():
        return pd.read_csv(ELO_PATH)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def encode_value(le_dict: dict, col: str, val: str) -> int:
    """Encode a category label using the fitted LabelEncoder safely."""
    le = le_dict.get(col) or le_dict.get("team")
    if le is not None:
        classes = list(le.classes_)
        if val in classes:
            return int(le.transform([val])[0])
    return 0


def get_h2h_wins(df: pd.DataFrame, team1: str, team2: str) -> int:
    """Count historical wins of team1 vs team2."""
    if df.empty or "winner" not in df.columns:
        return 0
    mask = (
        ((df["team1"] == team1) & (df.get("team2", pd.Series(dtype=str)) == team2))
        | ((df.get("team2", pd.Series(dtype=str)) == team1) & (df["team1"] == team2))
    )
    h2h_matches = df[mask]
    return int((h2h_matches["winner"] == team1).sum())


def get_venue_wins(df: pd.DataFrame, team1: str, venue: str) -> int:
    """Count historical wins of team1 at a given venue."""
    if df.empty or "winner" not in df.columns or "venue" not in df.columns:
        return 0
    mask = (df["venue"] == venue) & (
        (df["team1"] == team1) | (df.get("team2", pd.Series(dtype=str)) == team1)
    )
    return int((df[mask]["winner"] == team1).sum())


def get_team_elo(feat_df: pd.DataFrame, team: str) -> float:
    """Retrieve the latest cumulative Elo rating for a team from features dataset."""
    if feat_df.empty:
        return 1500.0
    t1_matches = feat_df[feat_df["team1"] == team]
    if not t1_matches.empty and "team1_elo" in t1_matches.columns:
        return float(t1_matches["team1_elo"].iloc[-1])
    t2_matches = feat_df[feat_df["team2"] == team]
    if not t2_matches.empty and "team2_elo" in t2_matches.columns:
        return float(t2_matches["team2_elo"].iloc[-1])
    return 1500.0


def get_season_idx(season: str | int) -> int:
    """Return season index relative to 2008 (season_year - 2008)."""
    try:
        s_str = str(season).split("/")[0]
        return max(0, int(s_str) - 2008)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🎯 Match Prediction — Win Probability")

models_ready = any((MODELS_DIR / f"match_winner_{n}.joblib").exists() for n in CLASSIF_NAMES)
if not models_ready:
    st.warning("⚠️ No trained classifier models found. Run `py train.py` first.")

with st.spinner("Loading models and data…"):
    clf_bundle = load_classifiers()
    feat_df = load_features()
    elo_df = load_elo()

le = clf_bundle.get("label_encoders", {})
scaler = clf_bundle.get("scaler")
clf_models = clf_bundle.get("models", {})


def _le_classes(col: str, fallback_col: str | None = None) -> list[str]:
    if col in le:
        return sorted(le[col].classes_.tolist())
    if fallback_col and not feat_df.empty and fallback_col in feat_df.columns:
        return sorted(feat_df[fallback_col].dropna().unique().tolist())
    return []


teams = _le_classes("team1", "team1") or _le_classes("team2", "team2") or ["Chennai Super Kings", "Mumbai Indians"]
venues = _le_classes("venue", "venue") or ["M Chinnaswamy Stadium", "Wankhede Stadium"]
seasons = sorted(feat_df["season"].astype(str).unique().tolist()) if not feat_df.empty and "season" in feat_df.columns else [str(y) for y in range(2008, 2027)]

# ---------------------------------------------------------------------------
# FR-3.1: Input form
# ---------------------------------------------------------------------------
st.subheader("📋 Match Setup")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1 (Batting / Home)", options=teams, index=0, key="p_team1")
    team2_opts = [t for t in teams if t != team1]
    team2 = st.selectbox("Team 2 (Bowling / Away)", options=team2_opts if team2_opts else teams, index=0, key="p_team2")
    venue = st.selectbox("Venue", options=venues, index=0, key="p_venue")

with col2:
    toss_winner_opts = [team1, team2]
    toss_winner = st.selectbox("Toss Winner", options=toss_winner_opts, index=0, key="p_toss_winner")
    toss_dec_sel = st.selectbox("Toss Decision", options=["Batting First", "Fielding First"], index=0, key="p_toss_dec")
    toss_decision = "bat" if toss_dec_sel == "Batting First" else "field"
    season = st.selectbox("Season", options=seasons, index=len(seasons) - 1, key="p_season")

predict_btn = st.button("🔮 Predict Match Winner", use_container_width=True, type="primary")

if predict_btn:
    st.session_state["pred_team1"] = team1
    st.session_state["pred_team2"] = team2
    st.session_state["pred_toss_winner"] = toss_winner
    st.session_state["pred_toss_decision"] = toss_decision
    st.session_state["pred_venue"] = venue
    st.session_state["pred_season"] = season
    st.session_state["pred_done"] = True

if st.session_state.get("pred_done") and clf_models:
    t1 = st.session_state["pred_team1"]
    t2 = st.session_state["pred_team2"]
    tw = st.session_state["pred_toss_winner"]
    td = st.session_state["pred_toss_decision"]
    ven = st.session_state["pred_venue"]
    sea = st.session_state["pred_season"]

    # Build feature vector
    team1_enc = encode_value(le, "team1", t1)
    team2_enc = encode_value(le, "team2", t2)
    toss_winner_enc = encode_value(le, "toss_winner", tw)
    toss_bat_first = 1 if td == "bat" else 0
    toss_winner_is_team1 = 1 if tw == t1 else 0
    venue_enc = encode_value(le, "venue", ven)
    season_idx = get_season_idx(sea)
    team1_h2h_wins = get_h2h_wins(feat_df, t1, t2)
    venue_team1_wins = get_venue_wins(feat_df, t1, ven)
    team1_elo = get_team_elo(feat_df, t1)
    team2_elo = get_team_elo(feat_df, t2)

    X_pred_df = pd.DataFrame([[
        team1_enc, team2_enc, toss_winner_enc, toss_bat_first,
        toss_winner_is_team1, venue_enc, season_idx,
        team1_h2h_wins, venue_team1_wins, team1_elo, team2_elo,
    ]], columns=CLASSIF_FEATURES)

    if scaler is not None:
        X_scaled_arr = scaler.transform(X_pred_df)
        X_scaled_df = pd.DataFrame(X_scaled_arr, columns=CLASSIF_FEATURES)
    else:
        X_scaled_df = X_pred_df

    # ---------------------------------------------------------------------------
    # FR-3.2: Run all classifiers
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("🔮 Prediction Results")

    with st.spinner("Running all 11 classifiers…"):
        proba_results = {}
        for name, model in clf_models.items():
            X_in = X_scaled_df if name in SCALE_MODELS else X_pred_df
            try:
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(X_in)[0, 1])
                else:
                    p = float(model.predict(X_in)[0])
            except Exception:
                p = 0.5
            proba_results[name] = p

    # Summary metrics
    stack_p = proba_results.get("stack", np.mean(list(proba_results.values())))
    avg_p = float(np.mean(list(proba_results.values())))

    winner_team = t1 if stack_p >= 0.5 else t2
    winner_prob = stack_p if stack_p >= 0.5 else (1.0 - stack_p)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("🏆 Predicted Winner", winner_team)
    m_col2.metric(f"🎯 Win Probability ({winner_team})", f"{winner_prob * 100:.1f}%")
    m_col3.metric("🥞 Stacking Ensemble P(team1)", f"{stack_p * 100:.1f}%", help="Master meta-classifier combining LR, RF & GB")
    m_col4.metric("📊 Consensus Average P(team1)", f"{avg_p * 100:.1f}%", help="Mean probability across all 11 classifiers")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # FR-3.3: Results table + bar chart
    # ---------------------------------------------------------------------------
    res_df = pd.DataFrame([
        {
            "Model": MODEL_LABELS.get(k, k),
            f"{t1} Win Prob": f"{v * 100:.1f}%",
            f"{t2} Win Prob": f"{(1 - v) * 100:.1f}%",
            "Raw P(team1)": v,
        }
        for k, v in proba_results.items()
    ])

    tab_results, tab_cm, tab_roc, tab_dt, tab_shap, tab_report, tab_formula = st.tabs([
        "📊 Results Table & Chart",
        "🔲 Confusion Matrix",
        "📈 ROC-AUC",
        "🌳 Decision Tree",
        "🔍 Feature Importance / SHAP",
        "📋 Classification Report",
        "📐 Formulas & Parameters",
    ])

    with tab_results:
        display_df = res_df[["Model", f"{t1} Win Prob", f"{t2} Win Prob"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name=t1,
            x=res_df["Model"],
            y=res_df["Raw P(team1)"] * 100,
            marker_color=get_team_color(t1),
            text=[f"{v*100:.1f}%" for v in res_df["Raw P(team1)"]],
            textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            name=t2,
            x=res_df["Model"],
            y=(1 - res_df["Raw P(team1)"]) * 100,
            marker_color=get_team_color(t2),
            text=[f"{(1-v)*100:.1f}%" for v in res_df["Raw P(team1)"]],
            textposition="outside",
        ))
        fig_bar.add_shape(
            type="line", x0=-0.5, x1=len(res_df) - 0.5, y0=50, y1=50,
            line=dict(color="white", width=1.5, dash="dash"),
        )
        fig_bar.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            xaxis_tickangle=-30,
            yaxis_title="Win Probability (%)",
            yaxis_range=[0, 115],
            legend=dict(orientation="h", y=1.1),
            height=460,
            title=f"Win Probability Comparison: {t1} vs {t2}",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------------------------------------------------------------------------
    # FR-3.4: Confusion matrix
    # ---------------------------------------------------------------------------
    with tab_cm:
        st.subheader("Confusion Matrix")
        sel_cm_model = st.selectbox(
            "Select model",
            options=[MODEL_LABELS.get(n, n) for n in clf_models.keys()],
            key="cm_model_sel",
        )
        sel_cm_key = [k for k, v in MODEL_LABELS.items() if v == sel_cm_model]
        sel_cm_key = sel_cm_key[0] if sel_cm_key else list(clf_models.keys())[0]

        if not feat_df.empty and "team1_won" in feat_df.columns:
            from sklearn.metrics import confusion_matrix
            from sklearn.model_selection import train_test_split

            X_all = feat_df[CLASSIF_FEATURES]
            y_all = feat_df["team1_won"].values.astype(int)
            _, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

            if scaler is not None:
                X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=CLASSIF_FEATURES)
            else:
                X_te_scaled = X_te

            model_cm = clf_models[sel_cm_key]
            X_cm_in = X_te_scaled if sel_cm_key in SCALE_MODELS else X_te
            y_pred_cm = model_cm.predict(X_cm_in)
            cm = confusion_matrix(y_te, y_pred_cm)

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=[f"{t2} Wins", f"{t1} Wins"],
                y=[f"{t2} Wins", f"{t1} Wins"],
                color_continuous_scale=[[0, IPL_THEME["navy"]], [1, IPL_THEME["orange"]]],
                text_auto=True,
            )
            fig_cm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                title=f"Confusion Matrix — {sel_cm_model}",
                height=380,
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Load features.parquet with team1_won column to show confusion matrix.")

    # ---------------------------------------------------------------------------
    # FR-3.5: ROC-AUC curves
    # ---------------------------------------------------------------------------
    with tab_roc:
        st.subheader("ROC-AUC Curves")
        if not feat_df.empty and "team1_won" in feat_df.columns:
            from sklearn.metrics import roc_curve, roc_auc_score
            from sklearn.model_selection import train_test_split

            X_all = feat_df[CLASSIF_FEATURES]
            y_all = feat_df["team1_won"].values.astype(int)
            _, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

            if scaler is not None:
                X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=CLASSIF_FEATURES)
            else:
                X_te_scaled = X_te

            fig_roc = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, (name, model) in enumerate(clf_models.items()):
                X_roc_in = X_te_scaled if name in SCALE_MODELS else X_te
                try:
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_roc_in)[:, 1]
                    else:
                        y_score = model.predict(X_roc_in).astype(float)
                    fpr, tpr, _ = roc_curve(y_te, y_score)
                    auc = roc_auc_score(y_te, y_score)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f"{MODEL_LABELS.get(name, name)} (AUC={auc:.3f})",
                        line=dict(color=colors[i % len(colors)]),
                        mode="lines",
                    ))
                except Exception:
                    pass

            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"), name="Random",
            ))
            fig_roc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                title="ROC-AUC — All 11 Classifiers",
                legend=dict(x=1.01, y=1),
                height=520,
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("features.parquet required for ROC curves.")

    # ---------------------------------------------------------------------------
    # FR-3.6: Decision Tree graphical visualisation
    # ---------------------------------------------------------------------------
    with tab_dt:
        st.subheader("🌳 Graphical Decision Tree Diagram")
        if "dt" in clf_models:
            dt_model = clf_models["dt"]

            ctrl_c1, ctrl_c2 = st.columns([1, 2])
            with ctrl_c1:
                max_d = st.slider(
                    "Tree Display Depth",
                    min_value=1,
                    max_value=int(dt_model.get_depth()),
                    value=3,
                    key="dt_depth_slider",
                    help="Limit displayed tree depth for visual clarity",
                )
            with ctrl_c2:
                view_type = st.radio(
                    "Visualization Mode",
                    options=["Interactive Vector Diagram (Graphviz)", "Matplotlib Tree Plot"],
                    horizontal=True,
                    key="dt_view_mode",
                )

            class_names = [f"{t2} Wins", f"{t1} Wins"]

            if view_type == "Interactive Vector Diagram (Graphviz)":
                from sklearn.tree import export_graphviz
                try:
                    dot_data = export_graphviz(
                        dt_model,
                        feature_names=CLASSIF_FEATURES,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        special_characters=True,
                        max_depth=max_d,
                    )
                    st.graphviz_chart(dot_data, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render Graphviz diagram: {e}")
            else:
                import matplotlib.pyplot as plt
                from sklearn.tree import plot_tree

                fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
                fig.patch.set_facecolor("#0F1117")
                ax.set_facecolor("#0F1117")

                annotations = plot_tree(
                    dt_model,
                    feature_names=CLASSIF_FEATURES,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    fontsize=8,
                    ax=ax,
                    max_depth=max_d,
                )

                # Style connecting edges and node borders for high visibility on dark theme
                for a in annotations:
                    if hasattr(a, "arrow_patch") and a.arrow_patch is not None:
                        a.arrow_patch.set_edgecolor("#FF6B00")
                        a.arrow_patch.set_linewidth(2.0)
                    bbox = a.get_bbox_patch()
                    if bbox is not None:
                        bbox.set_edgecolor("#FF6B00")
                        bbox.set_linewidth(1.2)

                plt.title(
                    f"Decision Tree Diagram (Max Depth = {max_d})",
                    fontsize=14,
                    fontweight="bold",
                    color="#FF6B00",
                    pad=15,
                )
                plt.tight_layout()
                st.pyplot(fig)


            with st.expander("📄 View Rules in Text Format"):
                from sklearn.tree import export_text
                try:
                    dt_text = export_text(
                        dt_model,
                        feature_names=CLASSIF_FEATURES,
                        max_depth=max_d,
                    )
                    st.code(dt_text, language="text")
                except Exception as e:
                    st.warning(f"Could not render DT text: {e}")
        else:
            st.info("Decision Tree model not loaded.")


    # ---------------------------------------------------------------------------
    # FR-3.7: SHAP & Feature Importance
    # ---------------------------------------------------------------------------
    with tab_shap:
        st.subheader("🔍 Feature Importance & SHAP Values")
        if "xgb" in clf_models:
            xgb_model = clf_models["xgb"]
            shap_success = False

            try:
                import shap as shap_lib
                explainer = shap_lib.TreeExplainer(xgb_model)
                shap_vals = explainer.shap_values(X_pred_df)
                if hasattr(shap_vals, "values"):
                    sv = shap_vals.values[0]
                elif isinstance(shap_vals, list):
                    sv = shap_vals[0][0] if shap_vals[0].ndim > 1 else shap_vals[0]
                elif shap_vals.ndim > 1:
                    sv = shap_vals[0]
                else:
                    sv = shap_vals

                shap_df = pd.DataFrame({
                    "Feature": CLASSIF_FEATURES,
                    "SHAP Value": sv,
                }).sort_values("SHAP Value", key=abs, ascending=False)

                fig_shap = px.bar(
                    shap_df,
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale=[[0, IPL_THEME["navy"]], [0.5, "#888"], [1, IPL_THEME["orange"]]],
                    title="SHAP Values for this Prediction (XGBoost)",
                )
                fig_shap.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0",
                    coloraxis_showscale=True,
                    yaxis=dict(autorange="reversed"),
                    height=420,
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.dataframe(shap_df, use_container_width=True, hide_index=True)
                if hasattr(explainer, "expected_value"):
                    exp_val = float(explainer.expected_value) if isinstance(explainer.expected_value, (float, int, np.number)) else float(explainer.expected_value[0])
                    st.caption(f"Expected base rate value: {exp_val:.4f}")
                shap_success = True
            except Exception:
                shap_success = False

            if not shap_success and hasattr(xgb_model, "feature_importances_"):
                importances = xgb_model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": CLASSIF_FEATURES,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False)

                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, IPL_THEME["navy"]], [1, IPL_THEME["orange"]]],
                    title="XGBoost Feature Importances",
                )
                fig_fi.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0",
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    height=420,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                st.dataframe(fi_df, use_container_width=True, hide_index=True)
        else:
            st.info("XGBoost model required for feature importance. Run train.py first.")

    # ---------------------------------------------------------------------------
    # FR-3.8: Classification report
    # ---------------------------------------------------------------------------
    with tab_report:
        st.subheader("📋 Classification Report")
        sel_rep_model = st.selectbox(
            "Select model",
            options=[MODEL_LABELS.get(n, n) for n in clf_models.keys()],
            key="rep_model_sel",
        )
        sel_rep_key = [k for k, v in MODEL_LABELS.items() if v == sel_rep_model]
        sel_rep_key = sel_rep_key[0] if sel_rep_key else list(clf_models.keys())[0]

        if not feat_df.empty and "team1_won" in feat_df.columns:
            from sklearn.metrics import classification_report
            from sklearn.model_selection import train_test_split

            X_all = feat_df[CLASSIF_FEATURES]
            y_all = feat_df["team1_won"].values.astype(int)
            _, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

            if scaler is not None:
                X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=CLASSIF_FEATURES)
            else:
                X_te_scaled = X_te

            model_rep = clf_models[sel_rep_key]
            X_rep_in = X_te_scaled if sel_rep_key in SCALE_MODELS else X_te
            y_pred_rep = model_rep.predict(X_rep_in)

            report_dict = classification_report(
                y_te, y_pred_rep,
                target_names=[f"{t2} Wins", f"{t1} Wins"],
                output_dict=True,
            )
            report_df = pd.DataFrame(report_dict).T.round(4)
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("features.parquet with team1_won column required.")

    # ---------------------------------------------------------------------------
    # FR-3.9: Algorithm formula display
    # ---------------------------------------------------------------------------
    with tab_formula:
        st.subheader("📐 Algorithm Formulas & Hyperparameters")
        sel_algo = st.selectbox(
            "Select algorithm",
            options=[MODEL_LABELS.get(n, n) for n in CLASSIF_NAMES],
            key="algo_sel",
        )
        sel_algo_key = [k for k, v in MODEL_LABELS.items() if v == sel_algo]
        sel_algo_key = sel_algo_key[0] if sel_algo_key else "lr"

        formula, hparams = ALGO_FORMULAS.get(sel_algo_key, ("", ""))

        st.markdown(f"#### {sel_algo}")
        if formula:
            st.latex(formula)
        st.markdown(f"**Hyperparameters:** `{hparams}`")

        # Global formulas common to all classifiers
        st.divider()
        st.markdown("#### Common Evaluation Metrics")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}")
        st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.latex(r"\text{AUC-ROC} = \int_0^1 \text{TPR}(t)\,d\,\text{FPR}(t)")

elif st.session_state.get("pred_done") and not clf_models:
    st.error("No classifier models available. Please run `py train.py` first.")
else:
    st.info("Fill in the match details above and click **Predict Match Winner** to see predictions.")
