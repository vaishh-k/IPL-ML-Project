"""👤 Player Profiles — Individual Analytics & Elo Rankings."""
from __future__ import annotations

import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


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

from src.utils.helpers import get_project_root, apply_theme

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.markdown(apply_theme(), unsafe_allow_html=True)

NAVY   = "#1F3864"
ORANGE = "#FF6B00"
PURPLE = "#7C3AED"

ROLE_COLOURS = {
    "🏏 Batter": "#3498db",
    "🧤 WK-Batter": "#e74c3c",
    "🔄 All-Rounder": ORANGE,
    "🎳 Bowler": PURPLE,
}

WK_LIST = {
    "MS Dhoni", "KL Rahul", "RR Pant", "SV Samson", "KD Karthik", "JC Buttler",
    "Q de Kock", "Ishan Kishan", "WP Saha", "PA Patel", "N Pooran", "PD Salt",
    "AB de Villiers", "KS Bharat", "J Sharma", "JM Bairstow", "D Padikkal",
    "Samson", "Kishan", "Karthik", "Rishabh Pant"
}

root = get_project_root()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_elo() -> pd.DataFrame:
    p = root / "data/processed/elo_ratings.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["player_name", "elo_rating", "match_count",
                                  "pom_count", "win_count", "win_rate"])


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    p = root / "data/processed/features.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_raw_csv() -> pd.DataFrame:
    p = root / "IPL.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_player_career_stats() -> pd.DataFrame:
    """Compute rich, 100% accurate career batting & bowling stats from raw IPL ball-by-ball data."""
    raw = load_raw_csv()
    if raw.empty:
        return pd.DataFrame()

    # 1. Batting career stats
    bat = raw.groupby("batter").agg(
        runs=("runs_batter", "sum"),
        balls=("ball", "count"),
        fours=("runs_batter", lambda x: (x == 4).sum()),
        sixes=("runs_batter", lambda x: (x == 6).sum())
    ).reset_index()

    match_bat = raw.groupby(["batter", "match_id"])["runs_batter"].sum().reset_index()
    high_scores = match_bat.groupby("batter")["runs_batter"].max().reset_index(name="high_score")
    fifties = match_bat.groupby("batter")["runs_batter"].apply(lambda x: ((x >= 50) & (x < 100)).sum()).reset_index(name="fifties")
    hundreds = match_bat.groupby("batter")["runs_batter"].apply(lambda x: (x >= 100).sum()).reset_index(name="hundreds")

    bat_stats = pd.merge(bat, high_scores, on="batter", how="left")
    bat_stats = pd.merge(bat_stats, fifties, on="batter", how="left")
    bat_stats = pd.merge(bat_stats, hundreds, on="batter", how="left")

    # 2. Bowling career stats
    bowl = raw.groupby("bowler").agg(
        wickets=("wicket_kind", lambda x: x.notna().sum()),
        balls_bowled=("ball", "count"),
        runs_conceded=("runs_total", "sum")
    ).reset_index()

    match_bowl = raw.groupby(["bowler", "match_id"]).agg(
        w=("wicket_kind", lambda x: x.notna().sum()),
        r=("runs_total", "sum")
    ).reset_index()
    match_bowl["fig_score"] = match_bowl["w"] * 100 - match_bowl["r"]
    best_bowl = match_bowl.sort_values("fig_score", ascending=False).groupby("bowler").first().reset_index()
    best_bowl["best_bowling"] = best_bowl["w"].astype(str) + "/" + best_bowl["r"].astype(str)

    bowl_stats = pd.merge(bowl, best_bowl[["bowler", "best_bowling"]], on="bowler", how="left")

    # Combine
    players = pd.merge(bat_stats, bowl_stats, left_on="batter", right_on="bowler", how="outer")
    players["player_name"] = players["batter"].fillna(players["bowler"])
    players = players.fillna(0)

    # Calculate exact roles
    def _assign_role(row):
        p = row["player_name"]
        r = row["runs"]
        w = row["wickets"]
        if p in WK_LIST:
            return "🧤 WK-Batter"
        if r >= 600 and w >= 25:
            return "🔄 All-Rounder"
        if w >= 20 and r < w * 30:
            return "🎳 Bowler"
        if r >= 300:
            return "🏏 Batter"
        if w > 0:
            return "🎳 Bowler"
        return "🏏 Batter"

    players["role"] = players.apply(_assign_role, axis=1)
    players["sr"] = np.where(players["balls"] > 0, (players["runs"] / players["balls"]) * 100, 0.0)
    players["econ"] = np.where(players["balls_bowled"] > 0, (players["runs_conceded"] / players["balls_bowled"]) * 6.0, 0.0)

    return players


with st.spinner("Loading player analytics & career records…"):
    elo_df = load_elo()
    feat_df = load_features()
    player_stats_df = load_player_career_stats()

# Merge Elo and career stats
if not elo_df.empty and not player_stats_df.empty:
    merged_players = pd.merge(elo_df, player_stats_df, on="player_name", how="left").fillna(0)
elif not elo_df.empty:
    merged_players = elo_df.copy()
    merged_players["role"] = "🏏 Player"
else:
    merged_players = player_stats_df.copy()

all_players = sorted(merged_players["player_name"].unique().tolist()) if not merged_players.empty else []

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_player_row(name: str) -> pd.Series | None:
    if merged_players.empty:
        return None
    sub = merged_players[merged_players["player_name"] == name]
    return sub.iloc[0] if not sub.empty else None


def season_pom(name: str) -> pd.DataFrame:
    """Count POM by season for a player."""
    if feat_df.empty or "player_of_match" not in feat_df.columns:
        return pd.DataFrame()
    sub = feat_df[feat_df["player_of_match"] == name]
    if sub.empty or "season" not in sub.columns:
        return pd.DataFrame()
    return sub.groupby("season").size().reset_index(name="POM Count")


def team_history(name: str) -> pd.DataFrame:
    """Teams the player appeared for (winning team when awarded POM)."""
    if feat_df.empty or "player_of_match" not in feat_df.columns:
        return pd.DataFrame()
    sub = feat_df[feat_df["player_of_match"] == name]
    if sub.empty or "winner" not in sub.columns:
        return pd.DataFrame()
    counts = sub.groupby("winner").size().reset_index(name="POM Appearances")
    counts.rename(columns={"winner": "Team"}, inplace=True)
    return counts.sort_values("POM Appearances", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f'<h1 style="color:{ORANGE}">👤 Player Profiles</h1>'
    f'<p style="color:#ccc">Individual career analytics, Elo rankings, verified role badges, '
    f'and side-by-side comparisons.</p>',
    unsafe_allow_html=True,
)
st.divider()

if not all_players:
    st.warning("No player data found. Run `py train.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# FR-6.1 — Player Selector
# ---------------------------------------------------------------------------
col_sel1, col_sel2 = st.columns([2, 2])

with col_sel1:
    player = st.selectbox("🔍 Select Player Profile", all_players, key="p1_sel")

with col_sel2:
    compare_options = ["— None —"] + [p for p in all_players if p != player]
    compare_player = st.selectbox("🔁 Compare with Player…", compare_options, key="p2_sel")

st.divider()

# ---------------------------------------------------------------------------
# FR-6.7 — Role badge
# ---------------------------------------------------------------------------
p_row = get_player_row(player)
if p_row is not None:
    role_label = str(p_row.get("role", "🏏 Batter"))
    badge_bg = ROLE_COLOURS.get(role_label, ORANGE)
    st.markdown(
        f'<span style="background:{badge_bg};color:#fff;padding:8px 18px;'
        f'border-radius:20px;font-weight:bold;font-size:1.1rem;letter-spacing:0.5px;">'
        f'{role_label}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

# ---------------------------------------------------------------------------
# FR-6.2 — Metric Cards
# ---------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
if p_row is not None:
    m1.metric("🏏 Matches", f"{int(p_row.get('match_count', 0)):,}")
    m2.metric("⚡ Elo Rating", f"{float(p_row.get('elo_rating', 1500)):.1f}")
    m3.metric("🏆 POM Awards", int(p_row.get("pom_count", 0)))
    m4.metric("💥 Career Runs", f"{int(p_row.get('runs', 0)):,}")
    m5.metric("🎯 Career Wickets", int(p_row.get("wickets", 0)))
else:
    m1.metric("🏏 Matches", "–")
    m2.metric("⚡ Elo Rating", "–")
    m3.metric("🏆 POM Awards", "–")
    m4.metric("💥 Career Runs", "–")
    m5.metric("🎯 Career Wickets", "–")

st.divider()

# ---------------------------------------------------------------------------
# Career Batting & Bowling Breakdown
# ---------------------------------------------------------------------------
if p_row is not None:
    st.subheader("📊 Career Performance Breakdown")
    c_bat, c_bowl = st.columns(2)

    with c_bat:
        st.markdown("#### 🏏 Batting Stats")
        b_runs = int(p_row.get("runs", 0))
        b_balls = int(p_row.get("balls", 0))
        b_sr = float(p_row.get("sr", 0.0))
        b_hs = int(p_row.get("high_score", 0))
        b_50 = int(p_row.get("fifties", 0))
        b_100 = int(p_row.get("hundreds", 0))
        b_4s = int(p_row.get("fours", 0))
        b_6s = int(p_row.get("sixes", 0))

        bat_df = pd.DataFrame([
            {"Stat": "Total Runs", "Value": f"{b_runs:,}"},
            {"Stat": "Balls Faced", "Value": f"{b_balls:,}"},
            {"Stat": "Strike Rate", "Value": f"{b_sr:.2f}"},
            {"Stat": "Highest Score", "Value": str(b_hs)},
            {"Stat": "50s / 100s", "Value": f"{b_50} / {b_100}"},
            {"Stat": "Fours / Sixes", "Value": f"{b_4s} / {b_6s}"},
        ])
        st.dataframe(bat_df, use_container_width=True, hide_index=True)

    with c_bowl:
        st.markdown("#### 🎳 Bowling Stats")
        w_wkts = int(p_row.get("wickets", 0))
        w_balls = int(p_row.get("balls_bowled", 0))
        w_overs = round(w_balls / 6.0, 1)
        w_econ = float(p_row.get("econ", 0.0))
        w_runs = int(p_row.get("runs_conceded", 0))
        w_best = str(p_row.get("best_bowling", "0/0"))

        bowl_df = pd.DataFrame([
            {"Stat": "Total Wickets", "Value": str(w_wkts)},
            {"Stat": "Overs Bowled", "Value": str(w_overs)},
            {"Stat": "Runs Conceded", "Value": f"{w_runs:,}"},
            {"Stat": "Economy Rate", "Value": f"{w_econ:.2f}" if w_balls > 0 else "–"},
            {"Stat": "Best Figures", "Value": w_best if w_wkts > 0 else "–"},
        ])
        st.dataframe(bowl_df, use_container_width=True, hide_index=True)

    st.divider()

# ---------------------------------------------------------------------------
# FR-6.3 — Season-by-Season POM Frequency
# ---------------------------------------------------------------------------
st.subheader("📅 Player of the Match Awards by Season")

pom_df = season_pom(player)
if pom_df.empty:
    st.info(f"No Player-of-Match award records found in dataset for **{player}**.")
else:
    fig_pom = px.bar(
        pom_df,
        x="season",
        y="POM Count",
        title=f"{player} — POM Awards per Season",
        color_discrete_sequence=[ORANGE],
        text="POM Count",
    )
    fig_pom.update_traces(textposition="outside")
    fig_pom.update_layout(
        xaxis_title="Season",
        yaxis_title="POM Awards",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        height=380,
    )
    st.plotly_chart(fig_pom, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# FR-6.4 — Team History
# ---------------------------------------------------------------------------
st.subheader("🏟️ Team History (Match Appearances)")

teams_df = team_history(player)
if teams_df.empty:
    st.info(f"No team history records found for **{player}**.")
else:
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.dataframe(teams_df, use_container_width=True, hide_index=True)
    with col_t2:
        fig_teams = px.pie(
            teams_df,
            names="Team",
            values="POM Appearances",
            title=f"{player} — POM Awards by Team",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_teams.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            height=380,
        )
        st.plotly_chart(fig_teams, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# FR-6.6 — Compare Two Players Side-by-Side
# ---------------------------------------------------------------------------
if compare_player != "— None —":
    st.subheader(f"🔁 Head-to-Head Comparison: {player} vs {compare_player}")

    p1 = get_player_row(player)
    p2 = get_player_row(compare_player)

    def _cmp_dict(p_data, name_str) -> dict:
        if p_data is None:
            return {"Player": name_str, "Role": "–", "Elo Rating": "–", "Matches": 0, "Runs": 0, "Wickets": 0, "Strike Rate": "–", "Economy": "–"}
        return {
            "Player": name_str,
            "Role": str(p_data.get("role", "🏏 Batter")),
            "Elo Rating": f"{float(p_data.get('elo_rating', 1500)):.1f}",
            "Matches": int(p_data.get("match_count", 0)),
            "POM Awards": int(p_data.get("pom_count", 0)),
            "Career Runs": f"{int(p_data.get('runs', 0)):,}",
            "Batting SR": f"{float(p_data.get('sr', 0)):.2f}",
            "High Score": int(p_data.get("high_score", 0)),
            "Wickets": int(p_data.get("wickets", 0)),
            "Economy": f"{float(p_data.get('econ', 0)):.2f}" if int(p_data.get("balls_bowled", 0)) > 0 else "–",
        }

    cmp_table = pd.DataFrame([_cmp_dict(p1, player), _cmp_dict(p2, compare_player)])
    st.dataframe(cmp_table, use_container_width=True, hide_index=True)

    # Comparison Bar Chart
    metrics_comp = {
        "Career Runs": [int(p1.get("runs", 0)) if p1 is not None else 0, int(p2.get("runs", 0)) if p2 is not None else 0],
        "Wickets x10": [int(p1.get("wickets", 0)) * 10 if p1 is not None else 0, int(p2.get("wickets", 0)) * 10 if p2 is not None else 0],
        "POM Awards x50": [int(p1.get("pom_count", 0)) * 50 if p1 is not None else 0, int(p2.get("pom_count", 0)) * 50 if p2 is not None else 0],
        "Elo Rating": [round(float(p1.get("elo_rating", 1500))) if p1 is not None else 1500, round(float(p2.get("elo_rating", 1500))) if p2 is not None else 1500],
    }

    fig_cmp = go.Figure(data=[
        go.Bar(name=player, x=list(metrics_comp.keys()), y=[v[0] for v in metrics_comp.values()], marker_color=ORANGE),
        go.Bar(name=compare_player, x=list(metrics_comp.keys()), y=[v[1] for v in metrics_comp.values()], marker_color=NAVY),
    ])
    fig_cmp.update_layout(
        barmode="group",
        title=f"Metrics Comparison: {player} vs {compare_player}",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        height=420,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.divider()

# ---------------------------------------------------------------------------
# Global Elo Leaderboard — Top 30
# ---------------------------------------------------------------------------
st.subheader("🏅 Global Elo Leaderboard — Top 30 Players")

if not merged_players.empty:
    top30 = merged_players.sort_values("elo_rating", ascending=False).head(30).reset_index(drop=True)
    top30.index += 1

    disp_cols = ["player_name", "role", "elo_rating", "match_count", "pom_count", "runs", "wickets"]
    disp_cols = [c for c in disp_cols if c in top30.columns]

    top30_disp = top30[disp_cols].rename(columns={
        "player_name": "Player",
        "role": "Role",
        "elo_rating": "Elo Rating",
        "match_count": "Matches",
        "pom_count": "POM Awards",
        "runs": "Career Runs",
        "wickets": "Wickets",
    })

    top30_disp["Elo Rating"] = top30_disp["Elo Rating"].apply(lambda x: f"{float(x):.1f}")
    top30_disp["Career Runs"] = top30_disp["Career Runs"].apply(lambda x: f"{int(x):,}")
    top30_disp["Matches"] = top30_disp["Matches"].apply(lambda x: f"{int(x):,}")
    top30_disp["Wickets"] = top30_disp["Wickets"].apply(lambda x: f"{int(x):,}")

    st.dataframe(top30_disp, use_container_width=True)

    # Leaderboard bar chart (top 15)
    top15 = merged_players.nlargest(15, "elo_rating")
    fig_lb = px.bar(
        top15,
        x="player_name",
        y="elo_rating",
        color="role",
        color_discrete_map=ROLE_COLOURS,
        title="Top 15 Players by Elo Rating & Role",
        text="elo_rating",
    )
    fig_lb.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_lb.update_layout(
        xaxis_title="Player",
        yaxis_title="Elo Rating",
        xaxis={"tickangle": -45},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#fff",
        height=440,
    )
    st.plotly_chart(fig_lb, use_container_width=True)
else:
    st.info("No Elo data available. Run `py train.py` first.")

st.divider()
st.caption("👤 Player Profiles · IPL Prediction Model 2008–2026")
