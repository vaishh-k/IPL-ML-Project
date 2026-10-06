"""📈 Score Prediction — Final Score Forecast."""
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
SCORE_FEAT_PATH = PROC_DIR / "score_features.parquet"
FEAT_PATH = PROC_DIR / "features.parquet"

SCORE_FEATURES = [
    "over", "cum_runs", "cum_wickets", "run_rate",
    "venue_enc", "season_idx", "overs_limit",
]

REG_NAMES = ["linear", "ridge", "lasso", "xgb"]
REG_LABELS = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "xgb": "XGBoost Regressor",
}

# ---------------------------------------------------------------------------
# Resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_regressors():
    """Load all 4 regression models."""
    result = {"models": {}, "feat_cols": SCORE_FEATURES}
    for name in REG_NAMES:
        p = MODELS_DIR / f"score_{name}.joblib"
        m = load_model_safe(str(p))
        if m is not None:
            result["models"][name] = m
    return result


@st.cache_data(show_spinner=False)
def load_score_features() -> pd.DataFrame:
    if SCORE_FEAT_PATH.exists():
        return pd.read_parquet(SCORE_FEAT_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_match_features() -> pd.DataFrame:
    if FEAT_PATH.exists():
        return pd.read_parquet(FEAT_PATH)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_label_encoders() -> dict:
    p = MODELS_DIR / "label_encoders.joblib"
    if p.exists():
        return joblib.load(p)
    return {}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def encode_venue(le_dict: dict, venue: str) -> int:
    """Encode venue label safely."""
    if "venue" in le_dict:
        le = le_dict["venue"]
        classes = list(le.classes_)
        if venue in classes:
            return int(le.transform([venue])[0])
    return 0


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
st.title("📈 Score Prediction — Final Score Forecast")

with st.spinner("Loading regression models and data…"):
    reg_bundle = load_regressors()
    score_df = load_score_features()
    match_df = load_match_features()
    le = load_label_encoders()

reg_models = reg_bundle.get("models", {})

if not reg_models:
    st.warning("⚠️ No trained regression models found. Run `py train.py` first.")

# Dropdown options
venues = sorted(le["venue"].classes_.tolist()) if "venue" in le else (
    match_df["venue"].dropna().unique().tolist() if not match_df.empty and "venue" in match_df.columns else ["M Chinnaswamy Stadium", "Wankhede Stadium"]
)
teams = sorted(le["team"].classes_.tolist()) if "team" in le else (
    sorted(le["team1"].classes_.tolist()) if "team1" in le else ["Chennai Super Kings", "Mumbai Indians"]
)
seasons = sorted(match_df["season"].astype(str).unique().tolist()) if not match_df.empty and "season" in match_df.columns else [str(y) for y in range(2008, 2027)]

# ---------------------------------------------------------------------------
# FR-4.1: Input form
# ---------------------------------------------------------------------------
st.subheader("📋 Match State Input")

with st.form("score_form"):
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Batting Team", options=teams, index=0, key="sc_bat_team")
        overs_completed = st.slider("Overs Completed", min_value=1, max_value=20, value=10, step=1, key="sc_overs_comp",
                                  help="Overs completed so far (1 to 20)")
        runs_scored = st.number_input("Runs Scored So Far", min_value=0, max_value=300,
                                       value=80, step=1, key="sc_runs")
    with col2:
        venue = st.selectbox("Venue", options=venues, index=0, key="sc_venue")
        wickets = st.slider("Wickets Fallen", min_value=0, max_value=10, value=2, key="sc_wickets")
        season = st.selectbox("Season", options=seasons, index=len(seasons) - 1, key="sc_season")

    predict_btn = st.form_submit_button("🔮 Predict Final Score", use_container_width=True, type="primary")

if predict_btn:
    st.session_state["sc_pred_done"] = True
    st.session_state["sc_batting_team_val"] = batting_team
    st.session_state["sc_overs_val"] = overs_completed
    st.session_state["sc_runs_val"] = runs_scored
    st.session_state["sc_wickets_val"] = wickets
    st.session_state["sc_venue_val"] = venue
    st.session_state["sc_season_val"] = season

# ---------------------------------------------------------------------------
# FR-4.2 → FR-4.7: Prediction results
# ---------------------------------------------------------------------------
if st.session_state.get("sc_pred_done") and reg_models:
    team_v = st.session_state["sc_batting_team_val"]
    overs_comp_v = st.session_state["sc_overs_val"]
    runs_v = st.session_state["sc_runs_val"]
    wickets_v = st.session_state["sc_wickets_val"]
    venue_v = st.session_state["sc_venue_val"]
    season_v = st.session_state["sc_season_val"]


    # Feature calculation
    over_idx = max(0, overs_comp_v - 1)
    run_rate = float(runs_v) / max(float(overs_comp_v), 1.0)
    venue_enc = encode_venue(le, venue_v)
    season_idx = get_season_idx(season_v)
    overs_limit = 20

    X_sc_df = pd.DataFrame([[
        over_idx, runs_v, wickets_v, run_rate, venue_enc, season_idx, overs_limit
    ]], columns=SCORE_FEATURES)

    with st.spinner("Running all 4 regression models…"):
        predictions = {}
        for name, model in reg_models.items():
            try:
                pred = float(model.predict(X_sc_df)[0])
                predictions[name] = max(round(pred), int(runs_v))
            except Exception:
                predictions[name] = int(runs_v)

    st.divider()
    st.subheader("🔮 Score Forecast Results")

    pred_vals = list(predictions.values())
    mean_pred = float(np.mean(pred_vals))
    std_pred = float(np.std(pred_vals))
    min_pred = int(min(pred_vals))
    max_pred = int(max(pred_vals))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 Consensus Forecast", f"{round(mean_pred)} runs")
    c2.metric("🏏 Current Score", f"{runs_v}/{wickets_v}")
    c3.metric("⏱️ Overs Completed", f"{overs_comp_v} overs")
    c4.metric("📈 Current Run Rate", f"{run_rate:.2f} rpo")
    c5.metric("🎯 Model Spread", f"{min_pred} - {max_pred} (±{std_pred:.1f})")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_forecast, tab_scatter, tab_coef, tab_lasso0, tab_eval, tab_formula = st.tabs([
        "📊 Score Forecasts & Trajectory",
        "📈 Pred vs Actual Scatter",
        "📉 Feature Coefficients",
        "0️⃣ Lasso Selection",
        "🎓 Model Evaluation",
        "📐 Formulas",
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: Score Forecasts & Trajectory
    # ---------------------------------------------------------------------------
    with tab_forecast:
        f_col1, f_col2 = st.columns([1, 1])

        with f_col1:
            st.markdown("#### Model Predictions")
            pred_df = pd.DataFrame([
                {
                    "Model": REG_LABELS.get(k, k),
                    "Predicted Score": v,
                    "Projected RR": f"{v / 20.0:.2f} rpo",
                    "Runs to Add": f"+{v - runs_v}"
                }
                for k, v in predictions.items()
            ])
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            fig_pred = px.bar(
                pred_df,
                x="Model",
                y="Predicted Score",
                color="Predicted Score",
                color_continuous_scale=[[0, IPL_THEME["navy"]], [1, IPL_THEME["orange"]]],
                text="Predicted Score",
                title=f"Predicted Final Score — {team_v}",
            )
            fig_pred.add_hline(
                y=mean_pred, line_dash="dash", line_color="white",
                annotation_text=f"Consensus Mean: {mean_pred:.0f}", annotation_position="top right"
            )
            fig_pred.update_traces(textposition="outside")
            fig_pred.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                coloraxis_showscale=False,
                height=380,
                yaxis_range=[0, max(260, max_pred + 20)],
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        with f_col2:
            st.markdown("#### Projected Score Trajectory (Worm Chart)")
            # Build over-by-over trajectory from current over to over 20
            future_overs = list(range(overs_comp_v, 21))

            fig_worm = go.Figure()
            # Historical trajectory
            hist_overs = list(range(1, overs_comp_v + 1))
            hist_runs = [round(run_rate * ov) for ov in hist_overs]
            fig_worm.add_trace(go.Scatter(
                x=hist_overs,
                y=hist_runs,
                mode="lines+markers",
                name="Actual Score Trajectory",
                line=dict(color=IPL_THEME["orange"], width=3),
                marker=dict(size=6),
            ))

            # Projected trajectory for each model
            colors_map = {
                "linear": "#3B82F6",
                "ridge": "#10B981",
                "lasso": "#8B5CF6",
                "xgb": "#F59E0B"
            }
            for name, pred_val in predictions.items():
                rem_overs = 20 - overs_comp_v
                rem_runs = pred_val - runs_v
                rpo_req = rem_runs / max(1, rem_overs)

                proj_x = future_overs
                proj_y = [runs_v + round(rpo_req * (ov - overs_comp_v)) for ov in future_overs]

                fig_worm.add_trace(go.Scatter(
                    x=proj_x,
                    y=proj_y,
                    mode="lines",
                    name=f"{REG_LABELS.get(name, name)} ({pred_val} r)",
                    line=dict(color=colors_map.get(name, "#888"), dash="dot", width=2),
                ))

            fig_worm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_title="Overs",
                yaxis_title="Cumulative Runs",
                title=f"Innings Progression Forecast — {team_v}",
                height=480,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_worm, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Scatter plot predicted vs actual
    # -------------------------------------------------------------------------
    with tab_scatter:
        st.subheader("Predicted vs Actual Scores (Validation Set)")
        if not score_df.empty and "final_score" in score_df.columns:
            X_all_sc_df = score_df[SCORE_FEATURES]
            y_actual = score_df["final_score"].values

            scatter_fig = go.Figure()
            colors_sc = [IPL_THEME["orange"], IPL_THEME["navy"], IPL_THEME["purple"], "#22C55E"]

            for i, (name, model) in enumerate(reg_models.items()):
                try:
                    y_hat = model.predict(X_all_sc_df)
                    # Subsample 500 points for clear rendering
                    np.random.seed(42)
                    idx = np.random.choice(len(y_actual), min(500, len(y_actual)), replace=False)
                    scatter_fig.add_trace(go.Scatter(
                        x=y_actual[idx],
                        y=y_hat[idx],
                        mode="markers",
                        name=REG_LABELS.get(name, name),
                        marker=dict(color=colors_sc[i % len(colors_sc)], size=5, opacity=0.6),
                    ))
                except Exception:
                    pass

            if len(y_actual) > 0:
                mn, mx = float(y_actual.min()), float(y_actual.max())
                scatter_fig.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines",
                    name="Ideal 1:1 Match",
                    line=dict(dash="dash", color="white", width=1.5),
                ))

            scatter_fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_title="Actual Final Score",
                yaxis_title="Predicted Final Score",
                title="Predicted vs Actual Final Scores (500-sample subset)",
                height=480,
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("score_features.parquet required. Run `py train.py` first.")

    # -------------------------------------------------------------------------
    # TAB 3: Feature coefficient chart (ridge + lasso)
    # -------------------------------------------------------------------------
    with tab_coef:
        st.subheader("Feature Coefficients — Ridge & Lasso")

        coef_fig = go.Figure()
        show_any = False
        coef_colors = {"linear": "#3B82F6", "ridge": IPL_THEME["orange"], "lasso": IPL_THEME["purple"]}

        for name in ["linear", "ridge", "lasso"]:
            if name in reg_models:
                model = reg_models[name]
                if hasattr(model, "coef_"):
                    coefs = np.array(model.coef_).ravel()
                    if len(coefs) == len(SCORE_FEATURES):
                        coef_fig.add_trace(go.Bar(
                            name=REG_LABELS[name],
                            x=SCORE_FEATURES,
                            y=coefs,
                            marker_color=coef_colors.get(name, "#888"),
                            opacity=0.85,
                        ))
                        show_any = True

        if show_any:
            coef_fig.update_layout(
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_title="Feature",
                yaxis_title="Coefficient Weight",
                title="Linear / Ridge / Lasso Feature Weights",
                xaxis_tickangle=-30,
                height=420,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(coef_fig, use_container_width=True)
        else:
            st.info("Linear models required. Run train.py first.")

        st.latex(r"\hat{\theta}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y")
        st.latex(r"\hat{\theta}_{\text{Lasso}} \approx \arg\min_\theta \|y - X\theta\|^2 + \lambda\|\theta\|_1")

    # -------------------------------------------------------------------------
    # TAB 4: Lasso zeroed features
    # -------------------------------------------------------------------------
    with tab_lasso0:
        st.subheader("Lasso — Regularization Feature Selection")
        if "lasso" in reg_models and hasattr(reg_models["lasso"], "coef_"):
            coefs = np.array(reg_models["lasso"].coef_).ravel()
            lasso_df = pd.DataFrame({
                "Feature": SCORE_FEATURES,
                "Coefficient": coefs,
                "Zeroed": np.abs(coefs) < 1e-10,
            }).sort_values("Coefficient", key=abs, ascending=False)

            zeroed = lasso_df[lasso_df["Zeroed"]]["Feature"].tolist()
            active = lasso_df[~lasso_df["Zeroed"]]["Feature"].tolist()

            col_a, col_z = st.columns(2)
            with col_a:
                st.success(f"**Active Features ({len(active)}):** {', '.join(active) if active else 'None'}")
            with col_z:
                st.warning(f"**Zeroed-Out Features ({len(zeroed)}):** {', '.join(zeroed) if zeroed else 'None'}")

            fig_l0 = px.bar(
                lasso_df,
                x="Feature",
                y="Coefficient",
                color="Zeroed",
                color_discrete_map={True: "#888", False: IPL_THEME["purple"]},
                labels={"Coefficient": "Lasso Weight"},
                title="Lasso Coefficients (Grey = Zeroed by L1 Penalization)",
            )
            fig_l0.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_tickangle=-30,
                height=380,
            )
            st.plotly_chart(fig_l0, use_container_width=True)
        else:
            st.info("Lasso model required. Run `py train.py` first.")

    # -------------------------------------------------------------------------
    # TAB 5: Model Evaluation Metrics
    # -------------------------------------------------------------------------
    with tab_eval:
        st.subheader("🎓 Regression Evaluation Metrics")
        if not score_df.empty and "final_score" in score_df.columns:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split

            X_score = score_df[SCORE_FEATURES]
            y_score = score_df["final_score"].values
            _, X_te, _, y_te = train_test_split(X_score, y_score, test_size=0.2, random_state=42)

            eval_rows = []
            n = len(y_te)
            p = X_te.shape[1]

            for name, model in reg_models.items():
                try:
                    y_pred = model.predict(X_te)
                    mae = mean_absolute_error(y_te, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                    r2 = r2_score(y_te, y_pred)
                    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                    eval_rows.append({
                        "Model": REG_LABELS.get(name, name),
                        "MAE (runs)": f"{mae:.2f}",
                        "RMSE (runs)": f"{rmse:.2f}",
                        "R² Score": f"{r2:.4f}",
                        "Adjusted R²": f"{adj_r2:.4f}",
                    })
                except Exception:
                    pass

            if eval_rows:
                st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
        else:
            st.info("score_features.parquet required for evaluation metrics.")

    # -------------------------------------------------------------------------
    # TAB 6: Formulas
    # -------------------------------------------------------------------------
    with tab_formula:
        st.subheader("📐 Mathematical Formulation")
        st.markdown("#### Regression Objectives")
        st.latex(r"h_\theta(x) = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n")
        st.latex(r"\text{Linear Regression Loss: } J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2")
        st.latex(r"\text{Ridge Loss (L2): } J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda \sum_{j=1}^n \theta_j^2")
        st.latex(r"\text{Lasso Loss (L1): } J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda \sum_{j=1}^n |\theta_j|")

        st.divider()
        st.markdown("#### XGBoost Regressor Objective")
        st.latex(r"\mathcal{L}(\phi) = \sum_{i} l(y_i, \hat{y}_i) + \sum_k \Omega(f_k), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2")
        st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^n|y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2}")

else:
    st.info("Fill in the match state above and click **Predict Final Score** to see forecasts.")
