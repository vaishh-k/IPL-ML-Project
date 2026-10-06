# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: generate_sample_data.py
# Path: generate_sample_data.py
# Description: Utility script to generate mock IPL match and ball-by-ball datasets for sandbox testing.
# ==============================================================================

import pandas as pd
import numpy as np
import os

def generate_data():
    # [Function]: generate_data
    # [Description]: Implements and executes the Generate Data logic within this module pipeline.
    np.random.seed(42)
    n_rows = 500
    
    # 1. match_id
    match_ids = np.arange(1001, 1001 + n_rows)
    
    # 2. season
    seasons = np.random.choice([2019, 2020, 2021, 2022, 2023, 2024], size=n_rows)
    
    # 3. date
    base_dates = {
        2019: pd.date_range('2019-03-23', '2019-05-12'),
        2020: pd.date_range('2020-09-19', '2020-11-10'),
        2021: pd.date_range('2021-04-09', '2021-10-15'),
        2022: pd.date_range('2022-03-26', '2022-05-29'),
        2023: pd.date_range('2023-03-31', '2023-05-29'),
        2024: pd.date_range('2024-03-22', '2024-05-26')
    }
    dates = []
    for s in seasons:
        dt = np.random.choice(base_dates[s])
        dates.append(pd.to_datetime(dt).strftime('%Y-%m-%d'))
        
    # 4. venue
    venues = np.random.choice([
        'Wankhede Stadium', 'M. Chinnaswamy Stadium', 'Eden Gardens', 
        'Arun Jaitley Stadium', 'Narendra Modi Stadium', 'MA Chidambaram Stadium',
        'Rajiv Gandhi International Cricket Stadium', 'Punjab Cricket Association Stadium'
    ], size=n_rows)
    
    # 5. team1 and team2
    teams = ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'SRH', 'DC', 'PBKS']
    team1 = np.random.choice(teams, size=n_rows)
    team2 = []
    for t1 in team1:
        remaining_teams = [t for t in teams if t != t1]
        team2.append(np.random.choice(remaining_teams))
    team2 = np.array(team2)
    
    # 6. toss_winner and toss_decision
    toss_winners = []
    for t1, t2 in zip(team1, team2):
        toss_winners.append(np.random.choice([t1, t2]))
    toss_winners = np.array(toss_winners)
    toss_decisions = np.random.choice(['bat', 'field'], size=n_rows)
    
    # 7. runs_scored (with outliers)
    runs_scored = np.random.normal(loc=165, scale=25, size=n_rows).astype(int)
    # Add outliers
    runs_scored[10] = 310  # High outlier
    runs_scored[45] = 20   # Low outlier
    runs_scored[120] = 345 # High outlier
    runs_scored[210] = 35  # Low outlier
    runs_scored[330] = 295 # High outlier
    
    # 8. wickets_lost
    wickets_lost = np.random.randint(0, 11, size=n_rows)
    
    # 9. overs
    overs = np.random.choice([20.0, 19.1, 19.3, 19.5, 18.0, 15.0, 12.0, 8.0], p=[0.8, 0.05, 0.03, 0.02, 0.04, 0.03, 0.02, 0.01], size=n_rows)
    
    # 10. strike_rate (with correlation to runs_scored)
    strike_rate = (runs_scored / (overs * 6)) * 100
    # Add slight random noise to strike rate
    strike_rate += np.random.normal(0, 5, size=n_rows)
    strike_rate = np.round(strike_rate, 2)
    
    # 11. winner
    winner = []
    for t1, t2 in zip(team1, team2):
        winner.append(np.random.choice([t1, t2]))
    winner = np.array(winner)
    
    # 12. player_of_match
    players = ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'Jasprit Bumrah', 
               'Andre Russell', 'Jos Buttler', 'Rashid Khan', 'Shubman Gill', 
               'KL Rahul', 'Ravindra Jadeja', 'Hardik Pandya', 'Rishabh Pant']
    player_of_match = np.random.choice(players, size=n_rows)
    
    # 13. margin
    margin = np.random.randint(1, 100, size=n_rows)
    
    # 14. is_playoff
    is_playoff = np.random.choice([True, False], p=[0.1, 0.9], size=n_rows)
    
    # 15. league (constant column)
    league = ['IPL'] * n_rows
    
    df = pd.DataFrame({
        'match_id': match_ids,
        'season': seasons,
        'date': dates,
        'venue': venues,
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winners,
        'toss_decision': toss_decisions,
        'winner': winner,
        'runs_scored': runs_scored,
        'wickets_lost': wickets_lost,
        'overs': overs,
        'strike_rate': strike_rate,
        'player_of_match': player_of_match,
        'margin': margin,
        'is_playoff': is_playoff,
        'league': league
    })
    
    # Introduce missing values (NaN)
    # winner has missing values (no result due to rain)
    df.loc[df.sample(frac=0.04, random_state=1).index, 'winner'] = np.nan
    # player_of_match has missing values
    df.loc[df.sample(frac=0.06, random_state=2).index, 'player_of_match'] = np.nan
    # margin has missing values
    df.loc[df.sample(frac=0.08, random_state=3).index, 'margin'] = np.nan
    # overs has missing values
    df.loc[df.sample(frac=0.02, random_state=4).index, 'overs'] = np.nan
    # strike_rate can be NaN where overs is NaN
    df.loc[df['overs'].isna(), 'strike_rate'] = np.nan

    # Introduce duplicate rows (add 8 duplicates)
    duplicates = df.iloc[[5, 12, 100, 230, 310, 400, 420, 450]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/ipl_sample.csv', index=False)
    print("Sample IPL data created successfully. Shape:", df.shape)

if __name__ == '__main__':
    generate_data()
