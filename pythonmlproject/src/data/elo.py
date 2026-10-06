import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Elo computation
# ---------------------------------------------------------------------------

def compute_elo(match_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute player Elo ratings from match records.

    Since the dataset provides only 'player_of_match' (not full team rosters),
    we track Elo for every player who appears as POM at least once.

    Elo update per match
    --------------------
    POM is (almost always) a member of the winning team, so:
        S_pom  = 1.0   (win)
        R_opp  = 1500  (league-average proxy for the opposing team)

    The expected score formula is:
        E = 1 / (1 + 10^((R_opp - R_pom) / 400))

    The rating update is:
        R_new = R_old + K * (S - E)

    Parameters
    ----------
    match_df : pd.DataFrame
        Output of loader.make_match_df(); must contain columns:
        date, player_of_match, winner, team1, team2.
    cfg : dict
        Project config loaded from config.yaml.

    Returns
    -------
    pd.DataFrame
        Columns: player_name, elo_rating, match_count, pom_count,
                 win_count, win_rate — sorted by elo_rating descending.
    """
    K: float = float(cfg.get("elo", {}).get("k_factor", 32))
    R0: float = float(cfg.get("elo", {}).get("initial_rating", 1500))

    ratings: dict = {}   # player_name -> current Elo float
    stats: dict = {}     # player_name -> {match_count, pom_count, win_count}

    # ── Sort chronologically ─────────────────────────────────────────────────
    df = match_df.copy()
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # ── Helper: initialise a new player ─────────────────────────────────────
    def _init(player: str) -> None:
        ratings[player] = R0
        stats[player] = {"match_count": 0, "pom_count": 0, "win_count": 0}

    def _expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    # ── Process each match ───────────────────────────────────────────────────
    for _, row in df.iterrows():
        pom = row.get("player_of_match", None)
        winner = row.get("winner", None)

        # Skip rows with missing POM or winner
        if pom is None or (isinstance(pom, float) and np.isnan(pom)):
            continue
        if winner is None or (isinstance(winner, float) and np.isnan(winner)):
            continue

        pom = str(pom).strip()
        winner = str(winner).strip()

        if not pom or pom.lower() in ("nan", "none", ""):
            continue
        if not winner or winner.lower() in ("nan", "none", ""):
            continue

        # Initialise player if first appearance
        if pom not in ratings:
            _init(pom)

        r_pom = ratings[pom]
        # Opponent proxy: league-average rating (1500)
        r_opp = 1500.0

        # POM is on the winning side by definition in almost all IPL records
        S_pom = 1.0
        E_pom = _expected(r_pom, r_opp)

        ratings[pom] = r_pom + K * (S_pom - E_pom)
        stats[pom]["match_count"] += 1
        stats[pom]["pom_count"] += 1
        stats[pom]["win_count"] += 1

    # ── Build output DataFrame ───────────────────────────────────────────────
    rows = []
    for player, elo in ratings.items():
        s = stats[player]
        mc = s["match_count"]
        wc = s["win_count"]
        rows.append({
            "player_name": player,
            "elo_rating": round(float(elo), 2),
            "match_count": mc,
            "pom_count": s["pom_count"],
            "win_count": wc,
            "win_rate": round(wc / mc, 4) if mc > 0 else 0.0,
        })

    result = pd.DataFrame(rows)
    if len(result) == 0:
        return result

    return (
        result
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_elo(elo_df: pd.DataFrame, path: str) -> None:
    """Save Elo ratings DataFrame to a CSV file."""
    elo_df.to_csv(path, index=False)
