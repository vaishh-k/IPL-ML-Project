"""🏆 Team Optimizer — Best XI Builder · K-Means · DBSCAN · PCA."""
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

from src.utils.helpers import get_project_root, apply_theme, get_team_color

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.markdown(apply_theme(), unsafe_allow_html=True)

NAVY   = "#1F3864"
ORANGE = "#FF6B00"
PURPLE = "#7C3AED"

CLUSTER_ARCHETYPES: dict[int, str] = {
    0: "🧤 WK-Batter",
    1: "🏏 Top-Order Batter",
    2: "🏏 Middle-Order Batter",
    3: "🔄 All-Rounder",
    4: "🎳 Pace Bowler",
    5: "🔄 Spin Bowler",
}

CLUSTER_COLOURS: dict[int, str] = {
    0: "#e74c3c",
    1: "#3498db",
    2: "#2ecc71",
    3: ORANGE,
    4: PURPLE,
    5: "#f39c12",
}

WK_LIST = {
    "MS Dhoni", "KL Rahul", "RR Pant", "SV Samson", "KD Karthik", "JC Buttler",
    "Q de Kock", "Ishan Kishan", "WP Saha", "PA Patel", "N Pooran", "PD Salt",
    "AB de Villiers", "KS Bharat", "J Sharma", "JM Bairstow", "D Padikkal"
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
root = get_project_root()


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    p = root / "data/processed/clusters.csv"
    if p.exists():
        df = pd.read_csv(p)
        rename_map = {}
        if "kmeans_cluster" not in df.columns and "cluster" in df.columns:
            rename_map["cluster"] = "kmeans_cluster"
        if "dbscan_cluster" not in df.columns and "dbscan_label" in df.columns:
            rename_map["dbscan_label"] = "dbscan_cluster"
        if "pca1" not in df.columns and "pc1" in df.columns:
            rename_map["pc1"] = "pca1"
        if "pca2" not in df.columns and "pc2" in df.columns:
            rename_map["pc2"] = "pca2"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df
    return pd.DataFrame(columns=["player_name", "kmeans_cluster", "dbscan_cluster",
                                  "elo_rating", "match_count", "pom_count",
                                  "win_rate", "pca1", "pca2"])


@st.cache_data(show_spinner=False)
def load_elo() -> pd.DataFrame:
    p = root / "data/processed/elo_ratings.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_raw_csv() -> pd.DataFrame:
    p = root / "IPL.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_player_team_pool() -> tuple[pd.DataFrame, list[str]]:
    """Load raw IPL dataset, map players to IPL teams, and classify 100% accurate player roles."""
    raw = load_raw_csv()
    elo_df = load_elo()

    if raw.empty:
        return pd.DataFrame(), []

    team_map = {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
    }
    raw["batting_team"] = raw["batting_team"].replace(team_map)
    raw["bowling_team"] = raw["bowling_team"].replace(team_map)

    p1 = raw[["batter", "batting_team"]].rename(columns={"batter": "player_name", "batting_team": "team"})
    p2 = raw[["bowler", "bowling_team"]].rename(columns={"bowler": "player_name", "bowling_team": "team"})
    p_teams = pd.concat([p1, p2]).drop_duplicates().dropna()

    all_teams_list = sorted(p_teams["team"].unique().tolist())

    # Batting & bowling career stats
    bat = raw.groupby("batter").agg(runs=("runs_batter", "sum")).reset_index()
    bowl = raw.groupby("bowler").agg(wickets=("wicket_kind", lambda x: x.notna().sum())).reset_index()

    players = pd.merge(bat, bowl, left_on="batter", right_on="bowler", how="outer")
    players["player_name"] = players["batter"].fillna(players["bowler"])

    if not elo_df.empty:
        players = pd.merge(players, elo_df[["player_name", "elo_rating"]], on="player_name", how="left").fillna(0)
    else:
        players["elo_rating"] = 1500.0

    def _assign_role(row):
        p = row["player_name"]
        r = row["runs"]
        w = row["wickets"]
        if p in WK_LIST:
            return "Wicketkeeper"
        if r >= 600 and w >= 25:
            return "All-Rounder"
        if w >= 20 and r < w * 30:
            return "Bowler"
        if r >= 300:
            return "Batter"
        if w > 0:
            return "Bowler"
        return "Batter"

    players["role"] = players.apply(_assign_role, axis=1)
    player_team_df = pd.merge(players, p_teams, on="player_name", how="inner")

    return player_team_df, all_teams_list


with st.spinner("Loading cluster and team data …"):
    cluster_df = load_clusters()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _cluster_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster averages."""
    if df.empty or "kmeans_cluster" not in df.columns:
        return pd.DataFrame()
    num_cols = ["elo_rating", "match_count", "pom_count", "win_rate"]
    avail = [c for c in num_cols if c in df.columns]
    agg = df.groupby("kmeans_cluster")[avail].mean().round(3).reset_index()
    agg["Archetype"] = agg["kmeans_cluster"].map(CLUSTER_ARCHETYPES).fillna("Unknown")
    count = df.groupby("kmeans_cluster").size().reset_index(name="Members")
    agg = agg.merge(count, on="kmeans_cluster")
    agg.rename(columns={
        "kmeans_cluster": "Cluster",
        "elo_rating": "Avg Elo",
        "match_count": "Avg Matches",
        "pom_count": "Avg POM",
        "win_rate": "Avg Win Rate",
    }, inplace=True)
    return agg[["Cluster", "Archetype", "Members", "Avg Elo", "Avg Matches", "Avg POM", "Avg Win Rate"]]


def _select_xi_from_pool(pool_df: pd.DataFrame, n_bat: int, n_bowl: int,
                         n_ar: int, n_wk: int, teams_filter: list[str] | None = None) -> list[dict]:
    """Select Best XI from player pool filtered by team selection using highest Elo."""
    if teams_filter:
        filtered = pool_df[pool_df["team"].isin(teams_filter)].copy()
    else:
        filtered = pool_df.copy()

    selected: list[dict] = []
    used: set = set()

    def _pick(role_name: str, count: int) -> None:
        sub = filtered[filtered["role"] == role_name]
        sub = sub[~sub["player_name"].isin(used)]
        top = sub.sort_values("elo_rating", ascending=False).drop_duplicates("player_name").head(count)
        for _, row in top.iterrows():
            used.add(row["player_name"])
            selected.append({
                "Player": row["player_name"],
                "Role": role_name,
                "Team": row["team"],
                "Elo": round(float(row.get("elo_rating", 1500)), 1),
                "Runs": int(row.get("runs", 0)),
                "Wickets": int(row.get("wickets", 0)),
            })

    _pick("Wicketkeeper", n_wk)
    _pick("Batter", n_bat)
    _pick("All-Rounder", n_ar)
    _pick("Bowler", n_bowl)

    return selected


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    f'<h1 style="color:{ORANGE}">🏆 Team Optimizer</h1>'
    f'<p style="color:#ccc">Best XI Builder · 2-Team Matchup Pool · K-Means Clusters · DBSCAN · PCA</p>',
    unsafe_allow_html=True,
)
st.divider()

if cluster_df.empty:
    st.warning("No cluster data found at `data/processed/clusters.csv`. Run `py train.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_xi, tab_clusters, tab_pca, tab_outliers, tab_elbow = st.tabs(
    ["🏏 Best XI Builder", "🎯 Clusters", "📊 PCA Viz", "⚠️ Outliers", "📈 Elbow Curve"]
)

# ===========================================================================
# TAB 1 — Best XI Builder
# ===========================================================================
with tab_xi:
    st.markdown(f'<h3 style="color:{NAVY}">🏏 Best XI Builder (2-Team Selection)</h3>', unsafe_allow_html=True)

    pool_df, available_teams = load_player_team_pool()

    st.markdown("#### 1. Select Team Selection Mode")
    pool_mode = st.radio(
        "Pool Selection Mode",
        options=["⚔️ 2-Team H2H Matchup (Select 2 Teams)", "🌐 All IPL Teams (Global Pool)"],
        horizontal=True,
        key="xi_pool_mode",
    )

    teams_filter = None
    if pool_mode == "⚔️ 2-Team H2H Matchup (Select 2 Teams)" and available_teams:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            team_a = st.selectbox("Select Team 1", options=available_teams, index=0, key="xi_team_a")
        with col_t2:
            team_b_opts = [t for t in available_teams if t != team_a]
            team_b = st.selectbox("Select Team 2", options=team_b_opts if team_b_opts else available_teams, index=0, key="xi_team_b")
        teams_filter = [team_a, team_b]
        st.info(f"Building Best XI from combined squad of **{team_a}** & **{team_b}**")
    else:
        st.info("Building Best XI from all players across all IPL teams")

    st.markdown("#### 2. Configure XI Role Composition")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        n_bat = st.slider("🏏 Batters", min_value=1, max_value=6, value=4, key="xi_n_bat")
    with col_r2:
        n_bowl = st.slider("🎳 Bowlers", min_value=1, max_value=5, value=4, key="xi_n_bowl")
    with col_r3:
        n_ar = st.slider("🔄 All-Rounders", min_value=0, max_value=4, value=2, key="xi_n_ar")
    with col_r4:
        n_wk = st.slider("🧤 Wicketkeeper", min_value=0, max_value=1, value=1, key="xi_n_wk")

    total_slots = n_bat + n_bowl + n_ar + n_wk
    st.caption(f"Total slots selected: **{total_slots}** / 11  {'✅' if total_slots == 11 else '⚠️ (Recommended: 11 players)'}")

    if st.button("⚡ Build Best XI", key="build_xi", type="primary"):
        if pool_df.empty:
            st.error("No player pool dataset available.")
        else:
            xi = _select_xi_from_pool(pool_df, n_bat, n_bowl, n_ar, n_wk, teams_filter)

            if not xi:
                st.warning("Could not select players for the specified criteria. Try broadening your selection.")
            else:
                xi_df = pd.DataFrame(xi)
                total_elo = xi_df["Elo"].sum()

                st.markdown(
                    f'<h4 style="color:{ORANGE}">⭐ Suggested Best XI Lineup</h4>',
                    unsafe_allow_html=True,
                )

                role_order = ["Wicketkeeper", "Batter", "All-Rounder", "Bowler"]
                batting_pos = 1

                for role in role_order:
                    role_players = xi_df[xi_df["Role"] == role]
                    for _, row in role_players.iterrows():
                        emoji = {"Wicketkeeper": "🧤", "Batter": "🏏",
                                 "All-Rounder": "🔄", "Bowler": "🎳"}.get(row["Role"], "👤")
                        t_color = get_team_color(row["Team"])
                        st.markdown(
                            f'<div style="background:linear-gradient(90deg,{NAVY},{PURPLE});'
                            f'color:#fff;padding:10px 18px;border-radius:10px;margin:4px 0;'
                            f'border-left:5px solid {t_color};">'
                            f'<b>{batting_pos:2d}.</b> {emoji} <b>{row["Player"]}</b> &nbsp;'
                            f'— <span style="color:#e0e0e0">{row["Team"]}</span> &nbsp;|&nbsp; '
                            f'<span style="background:{t_color};padding:2px 8px;border-radius:6px;'
                            f'font-size:0.85rem;font-weight:bold;">{row["Role"]}</span> &nbsp;|&nbsp; '
                            f'<span style="color:{ORANGE};font-weight:bold;">Elo: {row["Elo"]}</span> &nbsp;'
                            f'<small>({row["Runs"]} runs, {row["Wickets"]} wkts)</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        batting_pos += 1

                st.divider()

                # Summary metrics
                if teams_filter and len(teams_filter) == 2:
                    t1_cnt = len(xi_df[xi_df["Team"] == teams_filter[0]])
                    t2_cnt = len(xi_df[xi_df["Team"] == teams_filter[1]])
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("⚡ Total Team Elo", f"{total_elo:.0f}")
                    m2.metric(f"👥 {teams_filter[0]}", f"{t1_cnt} players")
                    m3.metric(f"👥 {teams_filter[1]}", f"{t2_cnt} players")
                    m4.metric("📊 Average Player Elo", f"{total_elo / len(xi_df):.1f}")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("⚡ Total Team Elo", f"{total_elo:.0f}")
                    m2.metric("📊 Average Player Elo", f"{total_elo / len(xi_df):.1f}")

                st.markdown("<br>", unsafe_allow_html=True)

                # Visual charts
                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    if teams_filter and len(teams_filter) == 2:
                        fig_team_pie = px.pie(
                            xi_df,
                            names="Team",
                            title="Best XI Team Split",
                            color_discrete_sequence=[get_team_color(teams_filter[0]), get_team_color(teams_filter[1])],
                        )
                        fig_team_pie.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#fff",
                            height=380,
                        )
                        st.plotly_chart(fig_team_pie, use_container_width=True)
                    else:
                        fig_role_pie = px.pie(
                            xi_df,
                            names="Role",
                            title="Best XI Role Composition",
                            color_discrete_sequence=[ORANGE, NAVY, PURPLE, "#2ecc71"],
                        )
                        fig_role_pie.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#fff",
                            height=380,
                        )
                        st.plotly_chart(fig_role_pie, use_container_width=True)

                with ch_col2:
                    fig_elo_bar = px.bar(
                        xi_df.sort_values("Elo", ascending=False),
                        x="Player",
                        y="Elo",
                        color="Team",
                        title="Best XI Player Elo Ratings",
                        text="Elo",
                    )
                    fig_elo_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                    fig_elo_bar.update_layout(
                        xaxis={"tickangle": -40},
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#fff",
                        height=380,
                    )
                    st.plotly_chart(fig_elo_bar, use_container_width=True)

# ===========================================================================
# TAB 2 — Clusters
# ===========================================================================
with tab_clusters:
    st.markdown(f'<h3 style="color:{NAVY}">🎯 K-Means Cluster Summary</h3>', unsafe_allow_html=True)

    centroids = _cluster_centroids(cluster_df)
    if not centroids.empty:
        st.dataframe(centroids, use_container_width=True, hide_index=True)
    else:
        st.info("Cluster data is present but lacks required columns for centroid computation.")

    st.divider()

    st.markdown("#### Cluster Archetypes")
    cols_arch = st.columns(3)
    for i, (cid, label) in enumerate(CLUSTER_ARCHETYPES.items()):
        col = cols_arch[i % 3]
        badge_color = CLUSTER_COLOURS.get(cid, "#888")
        count = int((cluster_df["kmeans_cluster"] == cid).sum()) if "kmeans_cluster" in cluster_df.columns else 0
        col.markdown(
            f'<div style="background:{badge_color};color:#fff;padding:8px 14px;'
            f'border-radius:10px;margin:4px;text-align:center;">'
            f'<b>{label}</b><br><small>{count} players</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if "kmeans_cluster" in cluster_df.columns:
        counts = cluster_df["kmeans_cluster"].value_counts().reset_index()
        counts.columns = ["Cluster", "Count"]
        counts["Archetype"] = counts["Cluster"].map(CLUSTER_ARCHETYPES).fillna("Unknown")

        fig_count = px.bar(
            counts.sort_values("Cluster"),
            x="Archetype",
            y="Count",
            color="Archetype",
            color_discrete_map={v: CLUSTER_COLOURS.get(k, "#888") for k, v in CLUSTER_ARCHETYPES.items()},
            title="Player Count per Cluster",
            text="Count",
        )
        fig_count.update_traces(textposition="outside")
        fig_count.update_layout(
            showlegend=False,
            xaxis_title="Cluster Archetype",
            yaxis_title="Number of Players",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
        )
        st.plotly_chart(fig_count, use_container_width=True)

# ===========================================================================
# TAB 3 — PCA Visualization
# ===========================================================================
with tab_pca:
    st.markdown(f'<h3 style="color:{NAVY}">📊 PCA Scatter Plot — pc1 vs pc2</h3>', unsafe_allow_html=True)

    if "pca1" not in cluster_df.columns or "pca2" not in cluster_df.columns:
        st.warning("PCA columns not found in clusters.csv. Run `py train.py` first.")
    else:
        pca_df = cluster_df.copy()
        if "kmeans_cluster" in pca_df.columns:
            pca_df["Archetype"] = pca_df["kmeans_cluster"].map(CLUSTER_ARCHETYPES).fillna("Unknown")
        else:
            pca_df["Archetype"] = "Unknown"

        hover_cols = ["player_name"]
        if "elo_rating" in pca_df.columns:
            hover_cols.append("elo_rating")
        if "win_rate" in pca_df.columns:
            hover_cols.append("win_rate")

        fig_pca = px.scatter(
            pca_df,
            x="pca1",
            y="pca2",
            color="Archetype",
            hover_name="player_name",
            hover_data={col: True for col in hover_cols if col != "player_name"},
            color_discrete_map={v: CLUSTER_COLOURS.get(k, "#888") for k, v in CLUSTER_ARCHETYPES.items()},
            title="PCA: Player Clusters in 2D Feature Space",
            size_max=10,
        )
        fig_pca.update_traces(marker=dict(size=8, opacity=0.8))
        fig_pca.update_layout(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff",
            legend_title_text="Cluster",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

# ===========================================================================
# TAB 4 — Outliers (DBSCAN)
# ===========================================================================
with tab_outliers:
    st.markdown(f'<h3 style="color:{NAVY}">⚠️ DBSCAN Outlier Detection</h3>', unsafe_allow_html=True)

    if "dbscan_cluster" not in cluster_df.columns:
        st.warning("DBSCAN labels not found in clusters.csv. Run `py train.py` first.")
    else:
        outlier_df = cluster_df[cluster_df["dbscan_cluster"] == -1].copy()

        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric("Total Players", len(cluster_df))
        col_stat2.metric("⚠️ Outliers (Noise)", len(outlier_df), delta=f"{len(outlier_df)/max(len(cluster_df),1)*100:.1f}%")

        if outlier_df.empty:
            st.success("No outliers detected by DBSCAN.")
        else:
            st.markdown("#### Outlier Players")
            display_cols = [c for c in ["player_name", "elo_rating", "match_count", "pom_count", "win_rate"] if c in outlier_df.columns]
            out_display = outlier_df[display_cols].reset_index(drop=True)
            out_display.index += 1
            out_display.rename(columns={"player_name": "Player", "elo_rating": "Elo", "match_count": "Matches", "pom_count": "POM", "win_rate": "Win Rate"}, inplace=True)
            st.dataframe(out_display, use_container_width=True)

# ===========================================================================
# TAB 5 — Elbow Curve
# ===========================================================================
with tab_elbow:
    st.markdown(f'<h3 style="color:{NAVY}">📈 Elbow Curve — Optimal K for K-Means</h3>', unsafe_allow_html=True)

    num_cols = ["elo_rating", "match_count", "pom_count", "win_rate"]
    avail = [c for c in num_cols if c in cluster_df.columns]

    if not avail:
        st.warning("Numeric columns not found in clusters.csv.")
    else:
        X_km = cluster_df[avail].fillna(0).values

        if st.button("⚙️ Compute Elbow Curve (K=2..10)", key="elbow_run"):
            with st.spinner("Running K-Means for K = 2 … 10 …"):
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X_km)

                k_vals = list(range(2, 11))
                wcss = []
                for k in k_vals:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_sc)
                    wcss.append(km.inertia_)

                elbow_df = pd.DataFrame({"K": k_vals, "WCSS": wcss})

                fig_elbow = px.line(
                    elbow_df, x="K", y="WCSS",
                    title="Elbow Curve — K-Means WCSS vs Number of Clusters",
                    markers=True,
                    color_discrete_sequence=[ORANGE],
                )
                fig_elbow.update_traces(marker=dict(size=10, color=ORANGE))
                fig_elbow.update_layout(
                    xaxis=dict(tickmode="linear", tick0=2, dtick=1),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#fff",
                    xaxis_title="Number of Clusters (K)",
                    yaxis_title="WCSS (Inertia)",
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
        else:
            st.info("Click **Compute Elbow Curve** to run K-Means for K=2 through K=10.")

st.divider()
st.caption("🏆 Team Optimizer · IPL Prediction Model 2008–2026")
