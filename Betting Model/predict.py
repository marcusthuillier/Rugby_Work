"""
predict.py
Predict the outcome of an upcoming match using the trained XGBoost model.

Usage:
    python predict.py --home "New Zealand" --away "South Africa"
    python predict.py --home "Ireland" --away "England" --neutral
    python predict.py --home "France" --away "Argentina" --competition "Rugby World Cup"
"""

import argparse
import os
import pickle
import math
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES_PATH  = "features.csv"
MODEL_PATH     = "model_xgb.json"
CALIBRATOR_PATH = "calibrator.pkl"

FEATURE_COLS = [
    "elo_diff",
    "home_prob_elo",
    "home_form_5",
    "home_form_10",
    "away_form_5",
    "away_form_10",
    "home_elo_momentum",
    "away_elo_momentum",
    "h2h_home_winrate",
    "h2h_n_games",
    "rankings_weight",
    "home_experience",
    "away_experience",
]

HOME_ADVANTAGE = 75

COMPETITION_WEIGHTS = {
    "rugby world cup": 1.0,
    "world cup": 1.0,
    "six nations": 0.9,
    "rugby championship": 0.9,
    "the rugby championship": 0.9,
    "autumn internationals": 0.8,
    "autumn nations series": 0.8,
    "summer series": 0.7,
    "pacific nations cup": 0.7,
    "nations cup": 0.7,
    "test match": 0.6,
    "friendly": 0.3,
    "exhibition": 0.2,
}


def elo_prob(home_elo: float, away_elo: float, neutral: bool = False) -> float:
    ha = 0 if neutral else HOME_ADVANTAGE
    return 1.0 / (1 + math.pow(10, (away_elo - (home_elo + ha)) / 400))


def get_competition_weight(competition: str) -> float:
    comp_lower = competition.lower()
    for key, weight in COMPETITION_WEIGHTS.items():
        if key in comp_lower:
            return weight
    return 0.5  # Default: unknown competition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home",        required=True)
    parser.add_argument("--away",        required=True)
    parser.add_argument("--neutral",     action="store_true")
    parser.add_argument("--competition", default="", help="Competition name (optional)")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CALIBRATOR_PATH):
        print("Model not found. Run model.py first.")
        return

    if not os.path.exists(FEATURES_PATH):
        print("features.csv not found. Run features.py first.")
        return

    # Load model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_PATH)
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)

    # Load feature history to get team stats
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")

    home = args.home
    away = args.away

    def get_team_stats(team: str) -> dict:
        home_games = df[df["Home_Team"] == team].tail(10)
        away_games = df[df["Away_Team"] == team].tail(10)
        all_games = pd.concat([home_games, away_games]).sort_values("Date")

        if all_games.empty:
            return {
                "elo": 1500.0,
                "form_5": 0.5,
                "form_10": 0.5,
                "momentum": 0.0,
                "experience": 0,
            }

        last = all_games.iloc[-1]
        if last["Home_Team"] == team:
            current_elo = last["pre_home_elo"] + (last["pre_home_elo"] * 0.01)  # rough
            # Use pre_home_elo as best approximation
            current_elo = float(last["pre_home_elo"])
        else:
            current_elo = float(last["pre_away_elo"])

        # Better: get from get_rank.csv
        rankings_path = os.path.join("..", "ELOR", "Datasets", "get_rank.csv")
        if os.path.exists(rankings_path):
            rankings = pd.read_csv(rankings_path)
            match = rankings[rankings["Team"] == team]
            if not match.empty:
                current_elo = float(match.iloc[0]["ELO"])

        # Form: last N games as home or away
        results = []
        for _, row in all_games.iterrows():
            if row["Home_Team"] == team:
                results.append(float(row["home_win"]))
            else:
                results.append(1.0 - float(row["home_win"]) if row["away_win"] else 0.5)

        form_5  = float(np.mean(results[-5:]))  if len(results) >= 1 else 0.5
        form_10 = float(np.mean(results[-10:])) if len(results) >= 1 else 0.5

        # ELO momentum (approx from pre-game ELOs)
        elo_series = []
        for _, row in all_games.tail(6).iterrows():
            elo_series.append(row["pre_home_elo"] if row["Home_Team"] == team
                              else row["pre_away_elo"])
        momentum = float(elo_series[-1] - elo_series[0]) if len(elo_series) >= 2 else 0.0

        experience = len(df[(df["Home_Team"] == team) | (df["Away_Team"] == team)])

        return {
            "elo": current_elo,
            "form_5": form_5,
            "form_10": form_10,
            "momentum": momentum,
            "experience": experience,
        }

    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    # H2H
    h2h_df = df[((df["Home_Team"] == home) & (df["Away_Team"] == away)) |
                ((df["Home_Team"] == away) & (df["Away_Team"] == home))].tail(10)
    h2h_home_wins = 0
    h2h_total = len(h2h_df)
    for _, row in h2h_df.iterrows():
        if row["Home_Team"] == home:
            h2h_home_wins += row["home_win"]
        else:
            h2h_home_wins += row["away_win"]
    h2h_winrate = h2h_home_wins / h2h_total if h2h_total > 0 else 0.5

    home_elo = home_stats["elo"]
    away_elo = away_stats["elo"]
    home_prob_elo = elo_prob(home_elo, away_elo, neutral=args.neutral)
    elo_diff = home_elo - away_elo

    competition_weight = get_competition_weight(args.competition)

    features = np.array([[
        elo_diff,
        home_prob_elo,
        home_stats["form_5"],
        home_stats["form_10"],
        away_stats["form_5"],
        away_stats["form_10"],
        home_stats["momentum"],
        away_stats["momentum"],
        h2h_winrate,
        h2h_total,
        competition_weight,
        home_stats["experience"],
        away_stats["experience"],
    ]])

    prob_home = calibrator.predict_proba(features)[0][1]
    prob_not_home = 1.0 - prob_home

    venue = "neutral venue" if args.neutral else f"{home} home"
    comp_str = f" ({args.competition})" if args.competition else ""

    print()
    print(f"  {home} vs {away}{comp_str}")
    print(f"  Venue: {venue}")
    print(f"  ELO:   {home} {home_elo:.0f}  |  {away} {away_elo:.0f}")
    print()
    print(f"  {'Model: ' + home:<36} {prob_home*100:.1f}%")
    print(f"  {'Model: ' + away:<36} {prob_not_home*100:.1f}%")
    print()
    print(f"  {'ELO baseline: ' + home:<36} {home_prob_elo*100:.1f}%")
    print(f"  {'ELO baseline: ' + away:<36} {(1-home_prob_elo)*100:.1f}%")
    print()
    edge = prob_home - home_prob_elo
    if abs(edge) > 0.03:
        direction = "overestimates" if edge < 0 else "underestimates"
        print(f"  Note: ELO {direction} {home}'s chances by {abs(edge)*100:.1f}pp")
    print()
    print(f"  H2H ({h2h_total} games): {home} win rate {h2h_winrate*100:.0f}%")
    print()


if __name__ == "__main__":
    main()
