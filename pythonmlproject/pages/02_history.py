"""📜 History & EDA — IPL Match Records."""
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

from src.utils.helpers import apply_theme, get_project_root

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.markdown(apply_theme(), unsafe_allow_html=True)

IPL_THEME = {"navy": "#1F3864", "orange": "#FF6B00", "purple": "#7C3AED"}
PROC_DIR = get_project_root() / "data" / "processed"
FEAT_PATH = PROC_DIR / "features.parquet"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEAT_PATH)


st.title("📜 History & EDA — IPL Match Records")

if not FEAT_PATH.exists():
    st.error(
        "⚠️ `data/processed/features.parquet` not found. "
        "Please run `py train.py` first."
    )
    st.stop()

with st.spinner("Loading match history…"):
    df = load_features()

# Ensure season is int/str for filtering
if "season" in df.columns:
    df["season"] = df["season"].astype(str)
    all_seasons = sorted(df["season"].unique())
else:
    all_seasons = []

all_teams = sorted(
    set(df["team1"].dropna().tolist() + df["team2"].dropna().tolist())
    if "team1" in df.columns and "team2" in df.columns
    else []
)

# ---------------------------------------------------------------------------
# Tabs layout
# ---------------------------------------------------------------------------
tab_filter, tab_teams, tab_toss, tab_pom, tab_score, tab_heatmap, tab_h2h = st.tabs([
    "🔍 Filter & Browse",
    "📊 Team Stats",
    "🎲 Toss Analysis",
    "🏅 Top POM",
    "📈 Score Dist.",
    "🗺️ Venue Heatmap",
    "⚔️ Head-to-Head",
])

# ===========================================================================
# TAB 1: Filter & Browse (FR-2.1, FR-2.2)
# ===========================================================================
with tab_filter:
    st.subheader("🔍 Filter Matches")

    # FR-2.1: Season filter
    if all_seasons:
        selected_seasons = st.multiselect(
            "Select Season(s)",
            options=all_seasons,
            default=all_seasons,
            help="Hold Ctrl/Cmd to select multiple seasons",
        )
        df_filt = df[df["season"].isin(selected_seasons)] if selected_seasons else df
    else:
        df_filt = df
        st.info("Season column not found — showing all matches.")

    st.caption(f"Showing {len(df_filt):,} matches")

    # FR-2.2: Filtered match table
    display_cols = [c for c in ["date", "team1", "team2", "winner", "venue",
                                 "team1_runs", "team2_runs"] if c in df_filt.columns]
    if display_cols:
        st.dataframe(
            df_filt[display_cols].sort_values("date", ascending=False)
            if "date" in display_cols
            else df_filt[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(df_filt.head(200), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2: Team Stats (FR-2.3)
# ===========================================================================
with tab_teams:
    st.subheader("📊 Win % by Team")

    if all_seasons:
        sel_seasons_team = st.multiselect(
            "Season filter",
            options=all_seasons,
            default=all_seasons,
            key="team_seasons",
        )
        df_t = df[df["season"].isin(sel_seasons_team)] if sel_seasons_team else df
    else:
        df_t = df

    if "team1" in df_t.columns and "winner" in df_t.columns:
        teams_all = sorted(set(df_t["team1"].dropna().tolist() + df_t.get("team2", pd.Series()).dropna().tolist()))
        records = []
        for team in teams_all:
            # Matches as team1 or team2
            as_t1 = df_t[df_t["team1"] == team]
            as_t2 = df_t[df_t.get("team2", pd.Series(dtype=str)) == team] if "team2" in df_t.columns else pd.DataFrame()
            total = len(as_t1) + len(as_t2)
            wins = (df_t["winner"] == team).sum()
            if total > 0:
                records.append({"team": team, "played": total, "won": int(wins), "lost": total - int(wins)})

        if records:
            team_df = pd.DataFrame(records).sort_values("won", ascending=False)
            team_df["win_pct"] = (team_df["won"] / team_df["played"] * 100).round(1)

            fig_teams = go.Figure()
            fig_teams.add_trace(go.Bar(
                name="Won",
                x=team_df["team"],
                y=team_df["won"],
                marker_color=IPL_THEME["orange"],
            ))
            fig_teams.add_trace(go.Bar(
                name="Lost",
                x=team_df["team"],
                y=team_df["lost"],
                marker_color=IPL_THEME["navy"],
            ))
            fig_teams.update_layout(
                barmode="stack",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_tickangle=-45,
                height=450,
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig_teams, use_container_width=True)

            st.dataframe(
                team_df[["team", "played", "won", "lost", "win_pct"]].rename(
                    columns={"team": "Team", "played": "Played", "won": "Won",
                             "lost": "Lost", "win_pct": "Win %"}
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("team1 or winner column not found.")

# ===========================================================================
# TAB 3: Toss Analysis (FR-2.4)
# ===========================================================================
with tab_toss:
    st.subheader("🎲 Toss Impact Analysis")

    if "toss_decision" in df.columns and "winner" in df.columns:
        # Win% when batting first vs fielding first
        if "toss_bat_first" in df.columns and "team1_won" in df.columns:
            toss_groups = df.groupby("toss_bat_first")["team1_won"].mean().reset_index()
            toss_groups["toss_bat_first"] = toss_groups["toss_bat_first"].map(
                {1: "Bat First", 0: "Field First", True: "Bat First", False: "Field First"}
            )
            toss_groups["win_pct"] = (toss_groups["team1_won"] * 100).round(1)
            toss_groups.rename(columns={"toss_bat_first": "Toss Decision"}, inplace=True)
        else:
            toss_groups = (
                df.assign(
                    toss_winner_won=(df["toss_winner"] == df["winner"])
                )
                .groupby("toss_decision")["toss_winner_won"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "win_pct", "count": "matches", "toss_decision": "Toss Decision"})
            )
            toss_groups["win_pct"] = (toss_groups["win_pct"] * 100).round(1)

        fig_toss = px.bar(
            toss_groups,
            x="Toss Decision",
            y="win_pct",
            color="Toss Decision",
            color_discrete_sequence=[IPL_THEME["orange"], IPL_THEME["navy"]],
            text="win_pct",
            labels={"win_pct": "Win %"},
            title="Toss Decision → Win %",
        )
        fig_toss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_toss.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            showlegend=False,
            height=380,
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_toss, use_container_width=True)

        # Season-level toss-win correlation
        if "season" in df.columns and "toss_winner" in df.columns:
            st.markdown("#### Toss Winner = Match Winner? (by season)")
            df_tw = df.copy()
            df_tw["toss_won_match"] = (df_tw["toss_winner"] == df_tw["winner"]).astype(int)
            season_tw = df_tw.groupby("season")["toss_won_match"].mean().reset_index()
            season_tw["pct"] = (season_tw["toss_won_match"] * 100).round(1)
            fig_stw = px.line(
                season_tw, x="season", y="pct",
                markers=True,
                labels={"season": "Season", "pct": "% Toss Winner Won"},
                color_discrete_sequence=[IPL_THEME["orange"]],
            )
            fig_stw.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                height=300,
            )
            st.plotly_chart(fig_stw, use_container_width=True)
    else:
        st.info("toss_decision or winner column not found.")

# ===========================================================================
# TAB 4: Top POM (FR-2.5)
# ===========================================================================
with tab_pom:
    st.subheader("🏅 Top 20 Player of the Match")

    if "player_of_match" in df.columns:
        pom_counts = (
            df["player_of_match"]
            .value_counts()
            .head(20)
            .reset_index()
        )
        pom_counts.columns = ["Player", "POM Awards"]

        fig_pom = px.bar(
            pom_counts,
            x="POM Awards",
            y="Player",
            orientation="h",
            color="POM Awards",
            color_continuous_scale=[[0, IPL_THEME["purple"]], [1, IPL_THEME["orange"]]],
            text="POM Awards",
        )
        fig_pom.update_traces(textposition="outside")
        fig_pom.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            height=580,
        )
        st.plotly_chart(fig_pom, use_container_width=True)
    else:
        st.info("player_of_match column not found.")

# ===========================================================================
# TAB 5: Score Distribution (FR-2.6)
# ===========================================================================
with tab_score:
    st.subheader("📈 Score Distribution")

    if "team1_runs" in df.columns:
        fig_hist = px.histogram(
            df,
            x="team1_runs",
            nbins=40,
            color_discrete_sequence=[IPL_THEME["orange"]],
            labels={"team1_runs": "Team 1 Total Runs"},
            title="Distribution of Team 1 Innings Scores",
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            bargap=0.05,
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box plot per season
        if "season" in df.columns:
            fig_box = px.box(
                df,
                x="season",
                y="team1_runs",
                color_discrete_sequence=[IPL_THEME["purple"]],
                labels={"season": "Season", "team1_runs": "Team 1 Runs"},
                title="Score Distribution by Season",
            )
            fig_box.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                xaxis_tickangle=-45,
                height=420,
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("team1_runs column not found.")

# ===========================================================================
# TAB 6: Venue Heatmap (FR-2.7)
# ===========================================================================
with tab_heatmap:
    st.subheader("🗺️ Venue Win% Heatmap (Top 10 Teams × Top 10 Venues)")

    if "team1" in df.columns and "venue" in df.columns and "winner" in df.columns:
        # Top 10 teams and top 10 venues by match count
        top_teams = (
            pd.concat([df["team1"], df.get("team2", pd.Series(dtype=str))])
            .value_counts()
            .head(10)
            .index.tolist()
        )
        top_venues = df["venue"].value_counts().head(10).index.tolist()

        df_hm = df[df["team1"].isin(top_teams) & df["venue"].isin(top_venues)].copy()
        if "team2" in df_hm.columns:
            df_hm2 = df[df["team2"].isin(top_teams) & df["venue"].isin(top_venues)].copy()
        else:
            df_hm2 = pd.DataFrame()

        # For each (team, venue) pair: win count / total
        records = []
        for team in top_teams:
            for venue in top_venues:
                as_t1 = df[(df["team1"] == team) & (df["venue"] == venue)]
                if "team2" in df.columns:
                    as_t2 = df[(df["team2"] == team) & (df["venue"] == venue)]
                else:
                    as_t2 = pd.DataFrame()
                total = len(as_t1) + len(as_t2)
                wins = (df[(df["venue"] == venue)]["winner"] == team).sum()
                records.append({
                    "team": team,
                    "venue": venue,
                    "win_pct": round(wins / total * 100, 1) if total > 0 else 0,
                    "total": total,
                })

        hm_df = pd.DataFrame(records)
        pivot = hm_df.pivot(index="team", columns="venue", values="win_pct").fillna(0)

        fig_hm = px.imshow(
            pivot,
            color_continuous_scale=[[0, IPL_THEME["navy"]], [0.5, "#8B4A00"], [1, IPL_THEME["orange"]]],
            labels=dict(color="Win %"),
            aspect="auto",
            title="Win % by Team × Venue",
        )
        fig_hm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
            height=480,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Required columns (team1, venue, winner) not found.")

# ===========================================================================
# TAB 7: H2H Comparison (FR-2.8)
# ===========================================================================
with tab_h2h:
    st.subheader("⚔️ Head-to-Head Comparison")

    if "team1" in df.columns and "team2" in df.columns and "winner" in df.columns:
        h2h_col1, h2h_col2 = st.columns(2)
        with h2h_col1:
            team_a = st.selectbox("Team A", options=all_teams, index=0, key="h2h_a")
        with h2h_col2:
            team_b_options = [t for t in all_teams if t != team_a]
            team_b = st.selectbox("Team B", options=team_b_options, index=0, key="h2h_b")

        if team_a and team_b:
            # All matches between the two teams (either order)
            h2h = df[
                ((df["team1"] == team_a) & (df["team2"] == team_b))
                | ((df["team1"] == team_b) & (df["team2"] == team_a))
            ].copy()

            st.markdown(f"### {team_a} vs {team_b}")
            st.caption(f"Total encounters: **{len(h2h)}**")

            if len(h2h) > 0:
                a_wins = (h2h["winner"] == team_a).sum()
                b_wins = (h2h["winner"] == team_b).sum()
                no_result = len(h2h) - a_wins - b_wins

                r1, r2, r3 = st.columns(3)
                r1.metric(f"{team_a} Wins", str(a_wins))
                r2.metric(f"{team_b} Wins", str(b_wins))
                r3.metric("No Result / Other", str(no_result))

                # Pie chart
                fig_h2h = px.pie(
                    values=[a_wins, b_wins, no_result],
                    names=[team_a, team_b, "No Result"],
                    color_discrete_sequence=[IPL_THEME["orange"], IPL_THEME["navy"], "#888"],
                    hole=0.4,
                )
                fig_h2h.update_traces(textinfo="percent+label")
                fig_h2h.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0",
                    height=360,
                )
                st.plotly_chart(fig_h2h, use_container_width=True)

                # Match-by-match table
                disp_h2h_cols = [c for c in ["date", "season", "venue", "toss_winner",
                                              "toss_decision", "winner", "team1_runs",
                                              "team2_runs"] if c in h2h.columns]
                if disp_h2h_cols:
                    st.dataframe(
                        h2h[disp_h2h_cols].sort_values("date", ascending=False)
                        if "date" in disp_h2h_cols
                        else h2h[disp_h2h_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info(f"No matches found between {team_a} and {team_b}.")
    else:
        st.info("team1, team2, or winner column not found.")
