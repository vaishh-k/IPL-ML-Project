import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from project root (2 levels up from src/data/)."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Season normalisation helper
# ---------------------------------------------------------------------------

def _normalise_season(val) -> int:
    """
    Convert a season value to a 4-digit integer year.

    Examples
    --------
    '2007/08' -> 2008  (take the second year in the slash pair)
    '2009/10' -> 2010
    2016      -> 2016
    '2016'    -> 2016
    """
    s = str(val).strip()
    # Format like '2007/08' or '2020/21'
    slash_match = re.match(r"(\d{4})/(\d{2})", s)
    if slash_match:
        base_year = int(slash_match.group(1))
        suffix = int(slash_match.group(2))
        # e.g. 2007/08 -> 2008; 2009/10 -> 2010
        century = base_year // 100 * 100
        second_year = century + suffix
        # Handle century rollover (e.g. 1999/00 -> 2000)
        if second_year < base_year:
            second_year += 100
        return second_year
    # Plain 4-digit year
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return int(s)


# ---------------------------------------------------------------------------
# Raw loader
# ---------------------------------------------------------------------------

def load_raw(cfg: dict) -> pd.DataFrame:
    """
    Load ball-by-ball IPL.csv, clean columns, normalise seasons and team names.
    """
    project_root = Path(__file__).parent.parent.parent
    raw_path = project_root / cfg["paths"]["raw_data"]
    if not raw_path.exists():
        fallback = project_root / "IPL.csv"
        if fallback.exists():
            raw_path = fallback

    df = pd.read_csv(raw_path, low_memory=False)

    # Drop any stray Unnamed index columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    # Normalise season column
    if "season" in df.columns:
        df["season"] = df["season"].apply(_normalise_season)

    # Normalise team names using aliases from config
    aliases: dict = cfg.get("team_aliases", {})
    if aliases:
        team_cols = [c for c in ("batting_team", "bowling_team", "toss_winner",
                                  "match_won_by") if c in df.columns]
        for col in team_cols:
            df[col] = df[col].replace(aliases)

    return df


# ---------------------------------------------------------------------------
# Match-level aggregation
# ---------------------------------------------------------------------------

def _parse_win_margin(win_outcome: str):
    """
    Parse win_by_runs and win_by_wickets from a win-outcome string.
    Returns (win_by_runs, win_by_wickets) as ints (0 if not applicable).

    Examples
    --------
    'Won by 45 runs'    -> (45, 0)
    'Won by 7 wickets'  -> (0, 7)
    'Tied'              -> (0, 0)
    """
    if not isinstance(win_outcome, str):
        return 0, 0
    lower = win_outcome.lower()
    runs_match = re.search(r"(\d+)\s*run", lower)
    wkts_match = re.search(r"(\d+)\s*wicket", lower)
    win_by_runs = int(runs_match.group(1)) if runs_match else 0
    win_by_wickets = int(wkts_match.group(1)) if wkts_match else 0
    return win_by_runs, win_by_wickets


def make_match_df(raw_df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Aggregate ball-by-ball DataFrame to match-level records.

    Derivation rules
    ----------------
    - team1  : batting_team in innings 1
    - team2  : bowling_team in innings 1
    - team1_runs    : max(team_runs) in innings 1  (last cumulative value)
    - team1_wickets : count of striker_out == True in innings 1
    - team2_runs    : max(team_runs) in innings 2  (0 if no innings 2)
    - team2_wickets : count of striker_out == True in innings 2
    - winner        : match_won_by (already alias-normalised)

    Filters applied
    ---------------
    - Exclude rows where result_type is in ('no result', 'N/A') or NaN
    - Exclude matches where win_outcome contains 'tie'

    Returns DataFrame with columns:
        match_id, date, season, city, venue, team1, team2,
        toss_winner, toss_decision, team1_runs, team1_wickets,
        team2_runs, team2_wickets, winner, result_type,
        win_by_runs, win_by_wickets, player_of_match, overs_limit
    """
    aliases: dict = (cfg or {}).get("team_aliases", {})

    # ── Work on a copy ──────────────────────────────────────────────────────
    df = raw_df.copy()

    # Ensure date is parsed
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Boolean striker_out if not already
    if "striker_out" in df.columns:
        df["striker_out"] = df["striker_out"].astype(bool)

    records = []

    for match_id, match_df in df.groupby("match_id"):
        inn1 = match_df[match_df["innings"] == 1]
        inn2 = match_df[match_df["innings"] == 2]

        if len(inn1) == 0:
            continue

        # ── Teams ───────────────────────────────────────────────────────────
        team1 = inn1["batting_team"].iloc[0]
        team2 = inn1["bowling_team"].iloc[0]
        # Apply aliases
        team1 = aliases.get(team1, team1)
        team2 = aliases.get(team2, team2)

        # ── Runs & wickets ──────────────────────────────────────────────────
        team1_runs = int(inn1["team_runs"].max()) if "team_runs" in inn1.columns else int(inn1["runs_total"].sum())
        team1_wickets = int(inn1["striker_out"].sum()) if "striker_out" in inn1.columns else 0

        if len(inn2) > 0:
            team2_runs = int(inn2["team_runs"].max()) if "team_runs" in inn2.columns else int(inn2["runs_total"].sum())
            team2_wickets = int(inn2["striker_out"].sum()) if "striker_out" in inn2.columns else 0
        else:
            team2_runs = 0
            team2_wickets = 0

        # ── Match-level metadata (from first row) ───────────────────────────
        first = match_df.iloc[0]

        date = first.get("date", pd.NaT)
        season = first.get("season", np.nan)
        city = first.get("city", "")
        venue = first.get("venue", "")
        toss_winner = str(first.get("toss_winner", "")).strip()
        toss_winner = aliases.get(toss_winner, toss_winner)
        toss_decision = str(first.get("toss_decision", "")).strip().lower()
        result_type = str(first.get("result_type", "")).strip().lower()
        match_won_by = str(first.get("match_won_by", "")).strip()
        match_won_by = aliases.get(match_won_by, match_won_by)
        win_outcome = str(first.get("win_outcome", "")).strip()
        player_of_match = first.get("player_of_match", "")
        overs_limit = first.get("overs", 20)

        # ── Winner (derive before filters so we can check it) ──────────────
        winner = match_won_by if match_won_by not in ("nan", "") else ""

        # ── Filters ─────────────────────────────────────────────────────────
        # result_type is NaN for normal completed matches; only "no result" or
        # "tie" rows have a non-null value — so exclude those explicitly.
        if result_type in ("no result", "tie"):
            continue
        # Belt-and-braces: also exclude if win_outcome string says tie
        if win_outcome and "tie" in win_outcome.lower():
            continue
        # Skip matches with no recorded winner
        if not winner or winner.lower() in ("nan", "none", ""):
            continue

        # ── Win margin ──────────────────────────────────────────────────────
        win_by_runs, win_by_wickets = _parse_win_margin(win_outcome)

        records.append({
            "match_id": match_id,
            "date": date,
            "season": season,
            "city": city,
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1_runs": team1_runs,
            "team1_wickets": team1_wickets,
            "team2_runs": team2_runs,
            "team2_wickets": team2_wickets,
            "winner": winner,
            "result_type": result_type,
            "win_by_runs": win_by_runs,
            "win_by_wickets": win_by_wickets,
            "player_of_match": player_of_match,
            "overs_limit": overs_limit,
        })

    result_df = pd.DataFrame(records)

    if len(result_df) == 0:
        return result_df

    # Final alias normalisation pass on the team columns
    for col in ("team1", "team2", "winner", "toss_winner"):
        if col in result_df.columns:
            result_df[col] = result_df[col].replace(aliases)

    result_df = result_df.sort_values("date").reset_index(drop=True)
    return result_df
