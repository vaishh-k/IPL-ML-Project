import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

CLASSIF_FEATURES = [
    "team1_enc",
    "team2_enc",
    "toss_winner_enc",
    "toss_bat_first",
    "toss_winner_is_team1",
    "venue_enc",
    "season_idx",
    "team1_h2h_wins",
    "venue_team1_wins",
    "team1_elo",
    "team2_elo",
]

SCORE_FEATURES = [
    "over",
    "cum_runs",
    "cum_wickets",
    "run_rate",
    "venue_enc",
    "season_idx",
    "overs_limit",
]

SEQ_LEN = 20
N_SEQ_FEATURES = 5  # over_norm, cum_runs_norm, cum_wickets_norm, run_rate_norm, wickets_left_norm


# ---------------------------------------------------------------------------
# Classification features
# ---------------------------------------------------------------------------

def build_features(match_df: pd.DataFrame, elo_df: pd.DataFrame, cfg: dict) -> tuple:
    """
    Build match-level classification features.

    Returns
    -------
    feat_df : pd.DataFrame
        Contains CLASSIF_FEATURES + auxiliary columns + target 'team1_won'.
    les : dict
        Label encoders keyed by 'team' and 'venue'.
    """
    df = match_df.copy()

    # ── Binary target ───────────────────────────────────────────────────────
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

    # ── Toss binary features ────────────────────────────────────────────────
    df["toss_bat_first"] = (df["toss_decision"] == "bat").astype(int)
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)

    # ── Season index ────────────────────────────────────────────────────────
    df["season_idx"] = df["season"].astype(int) - 2008

    # ── Label encoders ──────────────────────────────────────────────────────
    les: dict = {}

    all_teams = pd.concat([df["team1"], df["team2"], df["toss_winner"]]).dropna().unique()
    team_le = LabelEncoder().fit(sorted(all_teams))
    les["team"] = team_le
    # Alias keys so pages can encode any team column with a single shared encoder
    les["team1"] = team_le
    les["team2"] = team_le
    les["toss_winner"] = team_le

    # Handle any unseen labels gracefully (replace with first known class)
    def safe_team_transform(series: pd.Series) -> np.ndarray:
        known = set(team_le.classes_)
        cleaned = series.apply(lambda x: x if x in known else team_le.classes_[0])
        return team_le.transform(cleaned)

    df["team1_enc"] = safe_team_transform(df["team1"])
    df["team2_enc"] = safe_team_transform(df["team2"])
    df["toss_winner_enc"] = safe_team_transform(df["toss_winner"])

    venue_le = LabelEncoder().fit(sorted(df["venue"].dropna().unique()))
    les["venue"] = venue_le

    def safe_venue_transform(series: pd.Series) -> np.ndarray:
        known = set(venue_le.classes_)
        cleaned = series.apply(lambda x: x if x in known else venue_le.classes_[0])
        return venue_le.transform(cleaned)

    df["venue_enc"] = safe_venue_transform(df["venue"])

    # ── Sort by date for historical leakage prevention ───────────────────────
    df = df.sort_values("date").reset_index(drop=True)

    # ── H2H and venue win counts (historical — no lookahead) ────────────────
    h2h_wins_list: list = []
    venue_wins_list: list = []

    for i, row in df.iterrows():
        prior = df.iloc[:i]

        # H2H: prior matches between team1 and team2 (in either order)
        mask_h2h = (
            ((prior["team1"] == row["team1"]) & (prior["team2"] == row["team2"])) |
            ((prior["team1"] == row["team2"]) & (prior["team2"] == row["team1"]))
        )
        h2h_games = prior[mask_h2h]
        t1_wins_h2h = int((h2h_games["winner"] == row["team1"]).sum())
        h2h_wins_list.append(t1_wins_h2h)

        # Venue wins: prior matches where team1 played at this venue as team1
        venue_mask = (prior["venue"] == row["venue"]) & (prior["team1"] == row["team1"])
        venue_t1_wins = int((prior[venue_mask]["winner"] == row["team1"]).sum())
        venue_wins_list.append(venue_t1_wins)

    df["team1_h2h_wins"] = h2h_wins_list
    df["venue_team1_wins"] = venue_wins_list

    # ── Elo ratings ─────────────────────────────────────────────────────────
    df["team1_elo"] = 1500.0
    df["team2_elo"] = 1500.0

    if elo_df is not None and len(elo_df) > 0 and "elo_rating" in elo_df.columns:
        # Build a player→elo lookup
        pom_elo: dict = dict(zip(elo_df["player_name"], elo_df["elo_rating"]))

        # Accumulate team Elo as the running average of all POM Elo scores
        # attributed to each team across prior matches.
        team_elo_sum: dict = {}
        team_elo_cnt: dict = {}

        sorted_df = df.sort_values("date").reset_index(drop=True)

        for i, row in sorted_df.iterrows():
            # Assign current team Elos
            t1 = row["team1"]
            t2 = row["team2"]
            t1_elo = (team_elo_sum.get(t1, 0) / team_elo_cnt[t1]
                      if t1 in team_elo_cnt else 1500.0)
            t2_elo = (team_elo_sum.get(t2, 0) / team_elo_cnt[t2]
                      if t2 in team_elo_cnt else 1500.0)
            sorted_df.at[i, "team1_elo"] = t1_elo
            sorted_df.at[i, "team2_elo"] = t2_elo

            # Update team Elo with POM score for the winning team
            pom = row.get("player_of_match", None)
            winner = row.get("winner", None)
            if pom and isinstance(pom, str) and pom in pom_elo:
                winning_team = winner if isinstance(winner, str) else None
                if winning_team:
                    elo_val = pom_elo[pom]
                    team_elo_sum[winning_team] = team_elo_sum.get(winning_team, 0) + elo_val
                    team_elo_cnt[winning_team] = team_elo_cnt.get(winning_team, 0) + 1

        df = sorted_df

    # ── Select output columns ────────────────────────────────────────────────
    keep_cols = CLASSIF_FEATURES + [
        "team1_won",
        "team1", "team2", "season", "venue",
        "toss_winner", "toss_decision", "date", "winner",
        "team1_runs", "team2_runs", "team1_wickets", "team2_wickets",
        "player_of_match",
    ]
    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    feat_df = df[keep_cols].copy()

    return feat_df, les


# ---------------------------------------------------------------------------
# Score prediction features
# ---------------------------------------------------------------------------

def build_score_features(
    raw_df: pd.DataFrame,
    match_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Build over-by-over score prediction features from innings 1 data.

    Each row corresponds to one (match, over) pair and the target is
    'final_score' — the total runs scored in that innings.

    Returns
    -------
    pd.DataFrame with columns SCORE_FEATURES + ['final_score']
    """
    venue_le = LabelEncoder().fit(sorted(match_df["venue"].dropna().unique()))

    records = []

    for match_id in match_df["match_id"].unique():
        match_rows = match_df[match_df["match_id"] == match_id]
        if len(match_rows) == 0:
            continue
        match_row = match_rows.iloc[0]

        final_score = int(match_row["team1_runs"])
        season_idx = int(match_row["season"]) - 2008
        venue = match_row["venue"]

        try:
            venue_enc = int(venue_le.transform([venue])[0])
        except Exception:
            venue_enc = 0

        overs_limit = int(match_row.get("overs_limit", 20))

        # Ball-by-ball for innings 1 of this match
        inn_df = raw_df[
            (raw_df["match_id"] == match_id) & (raw_df["innings"] == 1)
        ].copy()

        if len(inn_df) == 0:
            continue

        for ov in range(overs_limit):
            ov_data = inn_df[inn_df["over"] <= ov]
            if len(ov_data) == 0:
                continue

            cum_runs = int(ov_data["runs_total"].sum()) if "runs_total" in ov_data.columns else 0
            cum_wickets = int(ov_data["striker_out"].astype(bool).sum()) if "striker_out" in ov_data.columns else 0
            cum_wickets = min(cum_wickets, 10)
            run_rate = cum_runs / max(ov + 1, 1)

            records.append({
                "over": ov,
                "cum_runs": cum_runs,
                "cum_wickets": cum_wickets,
                "run_rate": run_rate,
                "venue_enc": venue_enc,
                "season_idx": season_idx,
                "overs_limit": overs_limit,
                "final_score": final_score,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sequence features for deep learning
# ---------------------------------------------------------------------------

def build_sequence_features(
    raw_df: pd.DataFrame,
    match_df: pd.DataFrame,
    cfg: dict,
) -> tuple:
    """
    Build per-match over sequences for DL models.

    Each sequence has shape (seq_len, N_SEQ_FEATURES):
        [over_norm, cum_runs_norm, cum_wickets_norm, run_rate_norm, wickets_left_norm]

    Returns
    -------
    X       : np.ndarray, shape (n_matches, seq_len, N_SEQ_FEATURES), float32
    y_win   : np.ndarray, shape (n_matches,), float32  — 1 if team1 won
    y_score : np.ndarray, shape (n_matches,), float32  — innings 1 final score
    """
    seq_len = cfg.get("deep_learning", {}).get("seq_len", SEQ_LEN)

    X_list, y_win_list, y_score_list = [], [], []

    for _, m in match_df.iterrows():
        mid = m["match_id"]
        inn_df = raw_df[
            (raw_df["match_id"] == mid) & (raw_df["innings"] == 1)
        ].copy()

        if len(inn_df) == 0:
            continue

        final_score = float(m["team1_runs"])
        y_win = 1.0 if m["winner"] == m["team1"] else 0.0

        seq = []
        for ov in range(seq_len):
            ov_data = inn_df[inn_df["over"] <= ov]
            cum_runs = float(ov_data["runs_total"].sum()) if len(ov_data) > 0 else 0.0
            cum_wkts = float(ov_data["striker_out"].astype(bool).sum()) if len(ov_data) > 0 else 0.0
            rr = cum_runs / max(ov + 1, 1)
            seq.append([
                ov / 19.0,                      # over_norm          [0, 1]
                min(cum_runs / 300.0, 1.0),     # cum_runs_norm
                min(cum_wkts / 10.0, 1.0),      # cum_wickets_norm
                min(rr / 15.0, 1.0),            # run_rate_norm
                max((10.0 - cum_wkts) / 10.0, 0.0),  # wickets_left_norm
            ])

        X_list.append(seq)
        y_win_list.append(y_win)
        y_score_list.append(final_score)

    X = np.array(X_list, dtype=np.float32)
    y_win = np.array(y_win_list, dtype=np.float32)
    y_score = np.array(y_score_list, dtype=np.float32)
    return X, y_win, y_score


# ---------------------------------------------------------------------------
# Player features for autoencoder
# ---------------------------------------------------------------------------

def build_player_features(elo_df: pd.DataFrame) -> tuple:
    """
    Build player-level feature matrix for the autoencoder.

    Expects elo_df to have columns: player_name, elo_rating, match_count,
    pom_count, win_rate (others are ignored).

    Returns
    -------
    X            : np.ndarray, shape (n_players, 4), float32, MinMax-scaled
    player_names : list[str]
    """
    df = elo_df.copy()
    feat_cols = ["elo_rating", "match_count", "pom_count", "win_rate"]
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0.0)

    X = MinMaxScaler().fit_transform(df[feat_cols].values.astype(float))
    player_names = df["player_name"].tolist()
    return X.astype(np.float32), player_names
