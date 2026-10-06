import streamlit as st
import pandas as pd
import plotly.express as px
from services.api_client import ApiClient

# ==============================================================================
# IPL ML LAB - UNSUPERVISED LEARNING (PLAYER CLUSTERING) COMPONENT
# ==============================================================================
# This module implements the unsupervised learning workflow including:
# 1. Mode selection for player profiling (Batsmen stats vs Bowlers stats).
# 2. Algorithm configuration (K-Means with cluster slider, DBSCAN with Epsilon/MinSamples).
# 3. Model metrics display (Silhouette Score, Davies-Bouldin Index).
# 4. Interactive 2D PCA cluster visualization and cluster profile characteristics tables.
# 5. Similar player search tool computing nearest Euclidean neighbors inside clusters.
# ==============================================================================

def render_unsupervised_page(api: ApiClient):
    # Header Section
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 2rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 2rem;'>
            <h2 style='margin:0; font-size:2.2rem; font-weight:700; color: #F97316;'>Unsupervised Learning Lab</h2>
            <p style='margin:0.5rem 0 0 0; color:#94A3B8; font-size:1.1rem;'>
                Discover patterns and structure in IPL players using K-Means and DBSCAN clustering.
            </p>
        </div>
    """, unsafe_allow_html=True)

    #  1. SELECT PROBLEM TYPE 
    problem_type = st.radio(
        "Choose Player Profiling Mode:",
        options=["batsman", "bowler"],
        format_func=lambda x: "Batsman Profiling" if x == "batsman" else "Bowler Profiling",
        horizontal=True,
        key="unsup_prob_type"
    )

    st.markdown("---")

    col_config, col_run = st.columns([2, 1])

    with col_config:
        st.markdown("### Algorithm Configuration")
        algo = st.selectbox(
            "Select Clustering Algorithm:",
            options=["kmeans", "dbscan"],
            format_func=lambda x: "K-Means Clustering" if x == "kmeans" else "DBSCAN (Density-Based)",
            key="unsup_algo"
        )

        params = {}
        if algo == "kmeans":
            params["n_clusters"] = st.slider(
                "Number of Clusters (K):",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                key="unsup_kmeans_k"
            )
        else:
            params["eps"] = st.slider(
                "Epsilon (Maximum neighborhood distance):",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                key="unsup_dbscan_eps"
            )
            params["min_samples"] = st.slider(
                "Min Samples (Min points to form a core density cluster):",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                key="unsup_dbscan_min"
            )

    with col_run:
        st.markdown("### Action")
        st.write("Run the clustering algorithm on player aggregates parsed from historical ball-by-ball IPL statistics.")
        run_btn = st.button("Cluster IPL Players", type="primary", use_container_width=True, key="run_unsup_btn")

    # Retrieve previous results or run if button clicked
    res_key = f"unsup_results_{problem_type}_{algo}"
    
    if run_btn:
        with st.spinner("Training clustering model and computing metrics..."):
            results = api.train_unsupervised(problem_type, algo, params)
            if "error" in results:
                st.error(f"Error training: {results.get('detail', 'Unknown error')}")
                st.session_state[res_key] = None
            else:
                st.session_state[res_key] = results
                st.toast("Clustering Complete!")

    results = st.session_state.get(res_key, None)

    if results:
        #  2. METRICS DISPLAY 
        st.markdown("### Clustering Quality Metrics")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Clusters Found", results["n_clusters"])
        with m_col2:
            sil = results.get("silhouette_score")
            st.metric(
                "Silhouette Score", 
                f"{sil:.4f}" if sil is not None else "N/A",
                help="Measures cluster separation (range: -1 to 1). Higher is better."
            )
        with m_col3:
            db_idx = results.get("davies_bouldin_index")
            st.metric(
                "Davies-Bouldin Index",
                f"{db_idx:.4f}" if db_idx is not None else "N/A",
                help="Measures cluster similarity. Lower is better."
            )

        #  3. VISUALIZATION (PCA 2D Projection) 
        st.markdown("### 2D PCA Cluster Visualization")
        st.caption("Each dot represents an IPL player, projected into 2D space from multiple stats using Principal Component Analysis (PCA).")

        df_players = pd.DataFrame(results["players"])
        
        # Format cluster color labels (e.g. Cluster 0, Cluster 1, Noise/Outliers)
        df_players['cluster_label'] = df_players['cluster'].apply(
            lambda c: "Noise / Outlier" if c == -1 else f"Cluster {c}"
        )
        
        feature_cols = results["feature_columns"]
        
        # Construct hover details
        hover_data = {c: True for c in feature_cols}
        hover_data['cluster_label'] = False
        hover_data['pc1'] = False
        hover_data['pc2'] = False

        fig = px.scatter(
            df_players,
            x="pc1",
            y="pc2",
            color="cluster_label",
            hover_name="player_name",
            hover_data=hover_data,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Player Clusters mapped on PC1 vs PC2"
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Principal Component 1"),
            yaxis=dict(title="Principal Component 2")
        )
        st.plotly_chart(fig, use_container_width=True)

        #  4. CLUSTER PROFILES 
        st.markdown("### Cluster Profiles & Characteristics")
        st.caption("Average statistics for players grouped inside each cluster to understand the cluster characteristics.")

        profiles = results["profiles"]
        profile_rows = []
        for cluster_id, details in profiles.items():
            row = {"Cluster": "Noise/Outliers" if cluster_id == "-1" else f"Cluster {cluster_id}", "Players Count": details["count"]}
            for feat, val in details["averages"].items():
                feat_name = feat.replace("_", " ").title()
                row[feat_name] = round(val, 2)
            profile_rows.append(row)
        
        df_profiles = pd.DataFrame(profile_rows)
        st.dataframe(df_profiles, use_container_width=True, hide_index=True)

        #  5. FIND SIMILAR PLAYERS 
        st.markdown("---")
        st.markdown("### Find Similar IPL Players")
        st.caption("Select a player to retrieve the nearest neighbors within their cluster (similarity computed using Euclidean distance).")

        # Fetch list of available players
        players_res = api.get_unsupervised_players(problem_type)
        if "error" not in players_res:
            player_list = players_res.get("players", [])
            selected_player = st.selectbox(
                "Select Player:",
                options=player_list,
                key=f"unsup_player_select_{problem_type}"
            )
            
            num_similar = st.slider("Top N Similar Players:", 3, 10, 5, key="unsup_top_n")

            if st.button("Search Similar Players", type="secondary", key="run_similar_btn"):
                similar_res = api.find_similar_players(problem_type, algo, selected_player, num_similar)
                if "error" in similar_res:
                    st.error(f"Error: {similar_res.get('detail', 'Unknown error')}")
                else:
                    st.markdown(f"#### Search Results for **{selected_player}** (Cluster {similar_res['cluster']})")
                    
                    sim_players = similar_res["similar_players"]
                    if not sim_players:
                        st.info("No other players found in this cluster.")
                    else:
                        df_sim = pd.DataFrame(sim_players)
                        # Rename columns for prettier table headers
                        renames = {"player_name": "Player Name", "distance": "Distance (Lower = More Similar)"}
                        for col in df_sim.columns:
                            if col not in ["player_name", "distance"]:
                                renames[col] = col.replace("_", " ").title()
                        df_sim.rename(columns=renames, inplace=True)
                        st.dataframe(df_sim, use_container_width=True, hide_index=True)
        else:
            st.warning("Failed to retrieve player list.")
    else:
        st.info("Click **'Cluster IPL Players'** above to run the machine learning clustering model and analyze player groups.")
