"""🏠 Home — Dataset Overview & IPL Summary."""
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
import plotly.express as px
import plotly.graph_objects as go

from src.utils.helpers import apply_theme, get_project_root

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.markdown(apply_theme(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
PROC_DIR = get_project_root() / "data" / "processed"
FEAT_PATH = PROC_DIR / "features.parquet"

IPL_THEME = {"navy": "#1F3864", "orange": "#FF6B00", "purple": "#7C3AED"}


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEAT_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🏠 Home — Dataset Overview & IPL Summary")

if not FEAT_PATH.exists():
    st.error(
        "⚠️ `data/processed/features.parquet` not found. "
        "Please run `py train.py` first to generate processed data."
    )
    st.stop()

with st.spinner("Loading match data…"):
    df = load_features()

# ---------------------------------------------------------------------------
# FR-1.1: Metric cards
# ---------------------------------------------------------------------------
st.subheader("📊 Dataset at a Glance")

n_matches = len(df)
n_seasons = df["season"].nunique() if "season" in df.columns else 19
n_venues = df["venue"].nunique() if "venue" in df.columns else 0
n_pom = df["player_of_match"].nunique() if "player_of_match" in df.columns else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("🏏 Total Matches", f"{n_matches:,}")
m2.metric("📅 Seasons", str(n_seasons))
m3.metric("🏟️ Unique Venues", str(n_venues))
m4.metric("🏆 POM Players", str(n_pom))

st.divider()

# ---------------------------------------------------------------------------
# FR-1.2: Season-wise match count bar chart
# ---------------------------------------------------------------------------
st.subheader("📅 Matches per Season")

if "season" in df.columns:
    season_counts = (
        df.groupby("season").size().reset_index(name="matches").sort_values("season")
    )
    fig_season = px.bar(
        season_counts,
        x="season",
        y="matches",
        labels={"season": "Season", "matches": "Matches"},
        color="matches",
        color_continuous_scale=[[0, IPL_THEME["navy"]], [1, IPL_THEME["orange"]]],
        text="matches",
    )
    fig_season.update_traces(textposition="outside")
    fig_season.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        coloraxis_showscale=False,
        xaxis=dict(type="category", tickangle=-45),
        height=380,
    )
    st.plotly_chart(fig_season, use_container_width=True)
else:
    st.info("Season column not found in features.parquet.")

st.divider()

# ---------------------------------------------------------------------------
# FR-1.3: Season Top Teams (Most Match Wins per Season)
# ---------------------------------------------------------------------------
st.subheader("🏆 Season Top Teams (Most Match Wins per Season)")

if "winner" in df.columns and "season" in df.columns:
    season_top = (
        df[df["winner"].notna()]
        .groupby(["season", "winner"])
        .size()
        .reset_index(name="wins")
        .sort_values(["season", "wins"], ascending=[True, False])
        .drop_duplicates("season")
        .sort_values("season")
        .rename(columns={"season": "Season", "winner": "Top Team", "wins": "Match Wins"})
    )

    st.dataframe(
        season_top,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Winner or season column not found in features.parquet.")

st.divider()

# ---------------------------------------------------------------------------
# FR-1.4: Top 15 venues by match count
# ---------------------------------------------------------------------------
st.subheader("🏟️ Top 15 Venues by Match Count")

if "venue" in df.columns:
    venue_counts = (
        df["venue"].value_counts().head(15).reset_index()
    )
    venue_counts.columns = ["Venue", "Matches"]

    fig_venues = px.bar(
        venue_counts,
        x="Matches",
        y="Venue",
        orientation="h",
        color="Matches",
        color_continuous_scale=[[0, IPL_THEME["navy"]], [1, IPL_THEME["orange"]]],
        text="Matches",
    )
    fig_venues.update_traces(textposition="outside")
    fig_venues.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        height=480,
    )
    st.plotly_chart(fig_venues, use_container_width=True)
else:
    st.info("Venue column not found in features.parquet.")

st.divider()

# ---------------------------------------------------------------------------
# FR-1.5: Toss decision distribution — pie chart
# ---------------------------------------------------------------------------
st.subheader("🎲 Toss Decision Distribution")

if "toss_decision" in df.columns:
    toss_counts = df["toss_decision"].value_counts().reset_index()
    toss_counts.columns = ["Decision", "Count"]

    fig_toss = px.pie(
        toss_counts,
        names="Decision",
        values="Count",
        color_discrete_sequence=[IPL_THEME["orange"], IPL_THEME["navy"], IPL_THEME["purple"]],
        hole=0.4,
    )
    fig_toss.update_traces(textposition="inside", textinfo="percent+label")
    fig_toss.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        showlegend=True,
        height=380,
    )

    pie_col, _ = st.columns([1, 1])
    with pie_col:
        st.plotly_chart(fig_toss, use_container_width=True)
else:
    st.info("toss_decision column not found in features.parquet.")
