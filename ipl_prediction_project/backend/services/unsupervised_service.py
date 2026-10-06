# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: unsupervised_service.py
# Path: backend/services/unsupervised_service.py
# Description: Service layer implementing K-Means and DBSCAN clustering, silhouette scores, and Euclidean distance lookups.
# ==============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

# Global cache for aggregated data to avoid parsing 72MB csv repeatedly
_BATSMAN_STATS_CACHE = None
_BOWLER_STATS_CACHE = None
_TRAINED_CLUSTERING = {}

def get_player_stats(problem_type: str) -> pd.DataFrame:
    # [Function]: get_player_stats
    # [Description]: Implements and executes the Get Player Stats logic within this module pipeline.
    global _BATSMAN_STATS_CACHE, _BOWLER_STATS_CACHE
    
    csv_path = os.path.join(DATA_DIR, 'ipl_ball_by_ball.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError("ipl_ball_by_ball.csv not found")

    if problem_type == 'batsman':
        if _BATSMAN_STATS_CACHE is not None:
            return _BATSMAN_STATS_CACHE
        
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Group by batter
        # balls faced (wides don't count)
        df['is_ball_faced'] = (~df['extra_type'].str.contains('wides', na=False)).astype(int)
        # boundaries
        df['is_boundary'] = df['batter_runs'].isin([4, 6]).astype(int)
        
        agg = df.groupby('batter').agg(
            total_runs=('batter_runs', 'sum'),
            balls_faced=('is_ball_faced', 'sum'),
            boundaries=('is_boundary', 'sum')
        ).reset_index()
        
        # Filter batsmen with at least 100 balls faced to reduce noise
        agg = agg[agg['balls_faced'] >= 100].copy()
        
        # Calculate strike rate & boundary percentage
        agg['strike_rate'] = (agg['total_runs'] / agg['balls_faced']) * 100
        agg['boundary_percentage'] = (agg['boundaries'] / agg['balls_faced']) * 100
        
        # Final columns for clustering
        agg = agg[['batter', 'total_runs', 'strike_rate', 'balls_faced', 'boundary_percentage']].dropna()
        agg.rename(columns={'batter': 'player_name'}, inplace=True)
        _BATSMAN_STATS_CACHE = agg
        return agg

    elif problem_type == 'bowler':
        if _BOWLER_STATS_CACHE is not None:
            return _BOWLER_STATS_CACHE
        
        df = pd.read_csv(csv_path, low_memory=False)
        
        # balls bowled (wides and noballs don't count as legal balls bowled)
        df['is_ball_bowled'] = pd.to_numeric(df['valid_ball'], errors='coerce').fillna(1).astype(int)
        
        # bowler runs conceded: batter_runs + wides + noballs
        df['runs_conceded'] = pd.to_numeric(df['runs_bowler'], errors='coerce').fillna(0)
        
        # bowler wickets: is_wicket == 1, but excluding non-bowler dismissals like run out, retired hurt
        df['is_bowler_wicket'] = pd.to_numeric(df['bowler_wicket'], errors='coerce').fillna(0).astype(int)
        
        agg = df.groupby('bowler').agg(
            wickets=('is_bowler_wicket', 'sum'),
            balls_bowled=('is_ball_bowled', 'sum'),
            runs_conceded=('runs_conceded', 'sum')
        ).reset_index()
        
        # Filter bowlers with at least 60 balls bowled (10 overs)
        agg = agg[agg['balls_bowled'] >= 60].copy()
        
        # Calculate economy rate & bowling average
        agg['economy'] = (agg['runs_conceded'] / agg['balls_bowled']) * 6
        agg['bowling_average'] = np.where(agg['wickets'] > 0, agg['runs_conceded'] / agg['wickets'], agg['runs_conceded'])
        
        agg = agg[['bowler', 'wickets', 'economy', 'bowling_average', 'balls_bowled']].dropna()
        agg.rename(columns={'bowler': 'player_name'}, inplace=True)
        _BOWLER_STATS_CACHE = agg
        return agg
        
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")

def train_and_evaluate_clustering(problem_type: str, algorithm: str, params: dict) -> dict:
    # [Function]: train_and_evaluate_clustering
    # [Description]: Implements and executes the Train And Evaluate Clustering logic within this module pipeline.
    df_players = get_player_stats(problem_type).copy()
    if len(df_players) < 5:
        raise ValueError("Not enough player data to perform clustering.")

    # Features selection
    feature_cols = [c for c in df_players.columns if c != 'player_name']
    X = df_players[feature_cols].values

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Clustering
    if algorithm == 'kmeans':
        n_clusters = int(params.get('n_clusters', 4))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif algorithm == 'dbscan':
        eps = float(params.get('eps', 0.5))
        min_samples = int(params.get('min_samples', 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    df_players['cluster'] = labels

    # Calculate metrics
    unique_labels = set(labels)
    n_clusters_found = len(unique_labels - {-1})
    
    silhouette = None
    davies_bouldin = None
    
    # Silhouette & Davies-Bouldin require at least 2 clusters and not only noise
    if n_clusters_found >= 2:
        try:
            silhouette = float(silhouette_score(X_scaled, labels))
            davies_bouldin = float(davies_bouldin_score(X_scaled, labels))
        except Exception:
            pass

    # Project to 2D via PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df_players['pc1'] = X_pca[:, 0]
    df_players['pc2'] = X_pca[:, 1]

    # Cluster profiles (averages of raw metrics)
    profiles = {}
    for lbl in unique_labels:
        cluster_df = df_players[df_players['cluster'] == lbl]
        profiles[str(lbl)] = {
            'count': int(len(cluster_df)),
            'averages': cluster_df[feature_cols].mean().to_dict()
        }

    # Save trained results to in-memory store for real-time similarities
    cache_key = f"{problem_type}_{algorithm}"
    _TRAINED_CLUSTERING[cache_key] = {
        'df': df_players,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'X_scaled': X_scaled,
        'model': model
    }

    # Convert dataframe to json serializable records
    player_records = df_players[['player_name', 'cluster', 'pc1', 'pc2'] + feature_cols].to_dict(orient='records')

    return {
        'problem_type': problem_type,
        'algorithm': algorithm,
        'n_clusters': n_clusters_found,
        'silhouette_score': round(silhouette, 4) if silhouette is not None else None,
        'davies_bouldin_index': round(davies_bouldin, 4) if davies_bouldin is not None else None,
        'players': player_records,
        'profiles': profiles,
        'feature_columns': feature_cols
    }

def find_similar_players(problem_type: str, algorithm: str, player_name: str, top_n: int = 5) -> dict:
    # [Function]: find_similar_players
    # [Description]: Implements and executes the Find Similar Players logic within this module pipeline.
    cache_key = f"{problem_type}_{algorithm}"
    if cache_key not in _TRAINED_CLUSTERING:
        # Train with default parameters if not trained
        train_and_evaluate_clustering(problem_type, algorithm, {})
        
    trained = _TRAINED_CLUSTERING[cache_key]
    df = trained['df']
    
    if player_name not in df['player_name'].values:
        raise ValueError(f"Player '{player_name}' not found in the clustered dataset.")

    player_row = df[df['player_name'] == player_name].iloc[0]
    player_cluster = int(player_row['cluster'])
    
    # Filter players in the same cluster
    cluster_df = df[df['cluster'] == player_cluster].copy()
    
    if len(cluster_df) <= 1:
        return {
            'player_name': player_name,
            'cluster': player_cluster,
            'similar_players': []
        }
        
    # Get scaled values
    feature_cols = trained['feature_cols']
    scaler = trained['scaler']
    
    # Compute distances in scaled space
    all_players_scaled = scaler.transform(cluster_df[feature_cols].values)
    player_idx = cluster_df[cluster_df['player_name'] == player_name].index[0]
    target_idx_in_cluster = cluster_df.index.get_loc(player_idx)
    target_scaled = all_players_scaled[target_idx_in_cluster]
    
    distances = np.linalg.norm(all_players_scaled - target_scaled, axis=1)
    cluster_df['distance'] = distances
    
    # Exclude the player themselves, sort by distance
    similar = cluster_df[cluster_df['player_name'] != player_name].sort_values('distance')
    
    similar_records = similar[['player_name', 'distance'] + feature_cols].head(top_n).to_dict(orient='records')
    
    return {
        'player_name': player_name,
        'cluster': player_cluster,
        'similar_players': similar_records
    }
