"""
features.py
Builds the full feature matrix from ELO.csv + match_meta.csv.

Engineered features:
  pre_home_elo            - home team ELO before this game
  pre_away_elo            - away team ELO before this game
  elo_diff                - pre_home_elo - pre_away_elo
  home_prob_elo           - ELO-based win probability (Prob_Home from pipeline)
  home_form_5/10          - home team win rate in last 5/10 games (all venues)
  away_form_5/10          - away team win rate in last 5/10 games (all venues)
  home_elo_momentum       - home ELO change over last 5 games (positive = improving)
  away_elo_momentum       - away ELO change over last 5 games
  h2h_home_winrate        - home win rate in last 10 H2H meetings
  h2h_n_games             - number of H2H meetings in history
  rankings_weight         - World Rugby competition importance (0=exhibition, 1=WC)
                            (from match_meta.csv; defaults to 0.5 if not available)
  home_experience         - total games played by home team in dataset
  away_experience         - total games played by away team in dataset
  home_rest_days          - days since home team's last game (capped at 60)
  away_rest_days          - days since away team's last game (capped at 60)
  home_margin_form_5      - home team avg point differential in last 5 games
  away_margin_form_5      - away team avg point differential in last 5 games
  home_home_form_5        - home team win rate in last 5 HOME games specifically
  away_away_form_5        - away team win rate in last 5 AWAY games specifically

Target columns (all included, model picks what to use):
  result               - 'H' / 'D' / 'A'
  home_win             - 1 if home win, 0 otherwise
  away_win             - 1 if away win, 0 otherwise
  draw                 - 1 if draw, 0 otherwise

Outputs: features.csv
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

ELO_PATH    = os.path.join("..", "ELOR", "Datasets", "ELO.csv")
META_PATH   = "match_meta.csv"
OUT_PATH    = "features.csv"

DATE_FORMATS = ["%m/%d/%Y", "%d/%m/%Y", "%d %b %Y", "%Y-%m-%d"]

# Competition importance weights (0=exhibition, 1=World Cup)
# Inferred from competition name since the API rankingsWeight field is unpopulated
COMPETITION_WEIGHTS: list[tuple[str, float]] = [
    ("world cup",                1.00),
    ("rugby championship",       0.90),
    ("the rugby championship",   0.90),
    ("six nations",              0.90),
    ("tri nations",              0.90),
    ("four nations",             0.85),
    ("pacific nations cup",      0.75),
    ("nations cup",              0.75),
    ("autumn internationals",    0.75),
    ("autumn nations series",    0.75),
    ("summer series",            0.65),
    ("europe championship",      0.65),
    ("rugby europe",             0.60),
    ("americas rugby",           0.60),
    ("africa cup",               0.55),
    ("asia rugby",               0.55),
    ("oceania",                  0.55),
    ("super rugby",              0.50),
    ("premiership",              0.45),
    ("pro14",                    0.45),
    ("pro12",                    0.45),
    ("top 14",                   0.45),
    ("united rugby",             0.45),
    ("champions cup",            0.50),
    ("heineken",                 0.50),
    ("barbarians",               0.25),
    ("friendly",                 0.20),
    ("exhibition",               0.15),
]


def competition_weight(name: str) -> float:
    name_lower = name.lower()
    for keyword, weight in COMPETITION_WEIGHTS:
        if keyword in name_lower:
            return weight
    return 0.50  # unknown competition
INITIAL_ELO  = 1500.0
FORM_SHORT   = 5
FORM_LONG    = 10
H2H_WINDOW   = 10
MOMENTUM_N   = 5
REST_CAP     = 60   # cap rest days at 60 (season breaks shouldn't dominate)
MARGIN_SHORT = 5    # games for margin form


def parse_date(val: str) -> pd.Timestamp | None:
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(str(val).strip(), format=fmt)
        except ValueError:
            continue
    return None


def main():
    print("Loading ELO.csv...")
    elo = pd.read_csv(ELO_PATH)
    elo["ParsedDate"] = elo["Date"].apply(lambda v: parse_date(str(v)))
    elo = elo.dropna(subset=["ParsedDate"])
    elo = elo.sort_values("ParsedDate").reset_index(drop=True)

    # Load match metadata if available
    meta = None
    if os.path.exists(META_PATH):
        print("Loading match_meta.csv...")
        meta = pd.read_csv(META_PATH)
        meta["ParsedDate"] = pd.to_datetime(meta["Date"])
        # Merge key
        meta["_key"] = (meta["ParsedDate"].dt.strftime("%Y-%m-%d") + "|" +
                        meta["Home_Team"] + "|" + meta["Away_Team"])
        meta_lookup = dict(zip(meta["_key"],
                               zip(meta["Competition"], meta["Venue_Country"])))
    else:
        print("match_meta.csv not found — rankings_weight will default to 0.5")
        meta_lookup = {}

    # ── State tracking ─────────────────────────────────────────────────────────
    team_elo:        dict[str, float]  = defaultdict(lambda: INITIAL_ELO)
    team_results:    dict[str, deque]  = defaultdict(lambda: deque(maxlen=FORM_LONG))
    team_elo_hist:   dict[str, deque]  = defaultdict(lambda: deque(maxlen=MOMENTUM_N + 1))
    team_games:      dict[str, int]    = defaultdict(int)
    team_last_date:  dict[str, pd.Timestamp] = {}          # for rest days
    team_margins:    dict[str, deque]  = defaultdict(lambda: deque(maxlen=MARGIN_SHORT))
    team_home_res:   dict[str, deque]  = defaultdict(lambda: deque(maxlen=FORM_SHORT))
    team_away_res:   dict[str, deque]  = defaultdict(lambda: deque(maxlen=FORM_SHORT))

    # H2H: (home, away) -> deque of results (1=home win, 0=away win, 0.5=draw)
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    rows = []

    print(f"Engineering features for {len(elo):,} games...")
    for _, row in elo.iterrows():
        home = row["Home_Team"]
        away = row["Away_Team"]
        date_str = row["ParsedDate"].strftime("%Y-%m-%d")
        year = int(row["Year"])

        # Pre-game ELOs
        pre_home_elo = team_elo[home]
        pre_away_elo = team_elo[away]
        elo_diff = pre_home_elo - pre_away_elo

        # Recent form
        def win_rate(results: deque, n: int) -> float:
            recent = list(results)[-n:]
            return float(np.mean(recent)) if recent else 0.5

        home_form_5  = win_rate(team_results[home], FORM_SHORT)
        home_form_10 = win_rate(team_results[home], FORM_LONG)
        away_form_5  = win_rate(team_results[away], FORM_SHORT)
        away_form_10 = win_rate(team_results[away], FORM_LONG)

        # ELO momentum
        def momentum(elo_hist: deque) -> float:
            h = list(elo_hist)
            if len(h) < 2:
                return 0.0
            return h[-1] - h[0]

        home_elo_momentum = momentum(team_elo_hist[home])
        away_elo_momentum = momentum(team_elo_hist[away])

        # H2H
        pair_fwd = (home, away)
        pair_rev = (away, home)
        h2h_combined = list(h2h[pair_fwd]) + [1 - x for x in h2h[pair_rev]]
        h2h_home_winrate = float(np.mean(h2h_combined)) if h2h_combined else 0.5
        h2h_n_games = len(h2h_combined)

        # Experience
        home_experience = team_games[home]
        away_experience = team_games[away]

        # Rest days (days since last game, capped)
        current_date = row["ParsedDate"]
        home_rest = min((current_date - team_last_date[home]).days, REST_CAP) \
            if home in team_last_date else REST_CAP
        away_rest = min((current_date - team_last_date[away]).days, REST_CAP) \
            if away in team_last_date else REST_CAP

        # Margin-based form (avg point differential from team's perspective, last 5)
        def avg_margin(margins: deque) -> float:
            m = list(margins)
            return float(np.mean(m)) if m else 0.0

        home_margin_form_5 = avg_margin(team_margins[home])
        away_margin_form_5 = avg_margin(team_margins[away])

        # Venue-specific form
        home_home_form_5 = win_rate(team_home_res[home], FORM_SHORT)
        away_away_form_5 = win_rate(team_away_res[away], FORM_SHORT)

        # Competition metadata
        meta_key = f"{date_str}|{home}|{away}"
        if meta_key in meta_lookup:
            competition_name, venue_country = meta_lookup[meta_key]
            rankings_weight = competition_weight(competition_name)
        else:
            competition_name = ""
            rankings_weight = 0.5
            venue_country = ""

        # ELO pipeline probability
        home_prob_elo = float(row["Prob_Home"])

        # Outcome
        outcome = row["Home_Away_Draw"]  # 1=home, 0.5=draw, 0=away
        if outcome == 1.0:
            result = "H"
        elif outcome == 0.0:
            result = "A"
        else:
            result = "D"

        rows.append({
            "Date":              row["ParsedDate"],
            "Year":              year,
            "Home_Team":         home,
            "Away_Team":         away,
            "Competition":       competition_name,
            "Home_Score":        int(row["Home_Score"]),
            "Away_Score":        int(row["Away_Score"]),
            "pre_home_elo":      round(pre_home_elo, 1),
            "pre_away_elo":      round(pre_away_elo, 1),
            "elo_diff":          round(elo_diff, 1),
            "home_prob_elo":     round(home_prob_elo, 4),
            "home_form_5":       round(home_form_5, 4),
            "home_form_10":      round(home_form_10, 4),
            "away_form_5":       round(away_form_5, 4),
            "away_form_10":      round(away_form_10, 4),
            "home_elo_momentum": round(home_elo_momentum, 1),
            "away_elo_momentum": round(away_elo_momentum, 1),
            "h2h_home_winrate":  round(h2h_home_winrate, 4),
            "h2h_n_games":       h2h_n_games,
            "rankings_weight":   float(rankings_weight),
            "venue_country":     venue_country,
            "home_experience":   home_experience,
            "away_experience":   away_experience,
            "home_rest_days":    home_rest,
            "away_rest_days":    away_rest,
            "home_margin_form_5": round(home_margin_form_5, 2),
            "away_margin_form_5": round(away_margin_form_5, 2),
            "home_home_form_5":  round(home_home_form_5, 4),
            "away_away_form_5":  round(away_away_form_5, 4),
            "result":            result,
            "home_win":          int(outcome == 1.0),
            "away_win":          int(outcome == 0.0),
            "draw":              int(outcome == 0.5),
        })

        # ── Update state ───────────────────────────────────────────────────────
        home_elo_updated = float(row["Home_Rating_Updated"])
        away_elo_updated = float(row["Away_Rating_Updated"])

        team_elo[home] = home_elo_updated
        team_elo[away] = away_elo_updated

        team_results[home].append(outcome)
        team_results[away].append(1.0 - outcome)  # flip for away perspective

        team_elo_hist[home].append(home_elo_updated)
        team_elo_hist[away].append(away_elo_updated)

        team_games[home] += 1
        team_games[away] += 1

        # Update rest days tracker
        team_last_date[home] = current_date
        team_last_date[away] = current_date

        # Update margin form (from each team's perspective)
        margin = int(row["Home_Score"]) - int(row["Away_Score"])
        team_margins[home].append(margin)
        team_margins[away].append(-margin)

        # Update venue-specific form
        team_home_res[home].append(outcome)          # home team playing at home
        team_away_res[away].append(1.0 - outcome)    # away team playing away

        h2h[pair_fwd].append(outcome)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved features.csv: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Home wins: {df['home_win'].sum():,} ({df['home_win'].mean()*100:.1f}%)")
    print(f"  Away wins: {df['away_win'].sum():,} ({df['away_win'].mean()*100:.1f}%)")
    print(f"  Draws:     {df['draw'].sum():,} ({df['draw'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
