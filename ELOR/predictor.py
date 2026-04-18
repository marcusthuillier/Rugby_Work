"""
predictor.py
Predicts win probability for any two teams using their current ELO ratings.

Usage:
    python predictor.py --home "New Zealand" --away "South Africa"
    python predictor.py --home "Ireland" --away "England" --neutral

Can also be imported:
    from predictor import predict
    result = predict("New Zealand", "South Africa")
"""

import argparse
import math
import os
import pandas as pd

RANKINGS_PATH = os.path.join("Datasets", "get_rank.csv")
HOME_ADVANTAGE = 75  # ELO points, same as functions.py


def _probability(rating_a: float, rating_b: float) -> float:
    """Win probability for team A against team B."""
    return 1.0 / (1 + math.pow(10, (rating_b - rating_a) / 400))


def predict(home_team: str, away_team: str, neutral: bool = False,
            home_advantage: float = HOME_ADVANTAGE) -> dict:
    """
    Predict win probabilities for home_team vs away_team.

    Returns a dict with keys:
        home_team, away_team, home_elo, away_elo,
        home_win_prob, away_win_prob, neutral
    """
    rankings = pd.read_csv(RANKINGS_PATH)
    ratings = dict(zip(rankings["Team"], rankings["ELO"]))

    if home_team not in ratings:
        raise ValueError(f"Unknown team: '{home_team}'. Check get_rank.csv for valid names.")
    if away_team not in ratings:
        raise ValueError(f"Unknown team: '{away_team}'. Check get_rank.csv for valid names.")

    home_elo = float(ratings[home_team])
    away_elo = float(ratings[away_team])

    if neutral:
        adjusted_home = home_elo
    else:
        adjusted_home = home_elo + home_advantage

    home_prob = _probability(adjusted_home, away_elo)
    away_prob = 1.0 - home_prob

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
        "neutral": neutral,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict rugby match outcome using ELO ratings")
    parser.add_argument("--home", required=True, help="Home team name (e.g. 'New Zealand')")
    parser.add_argument("--away", required=True, help="Away team name (e.g. 'South Africa')")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue (no home advantage)")
    args = parser.parse_args()

    try:
        r = predict(args.home, args.away, neutral=args.neutral)
    except ValueError as e:
        print(f"Error: {e}")
        return

    venue = "neutral venue" if r["neutral"] else f"{r['home_team']} home"
    print()
    print(f"  Match:  {r['home_team']} vs {r['away_team']}")
    print(f"  Venue:  {venue}")
    print(f"  ELO:    {r['home_team']} {r['home_elo']:.0f}  |  {r['away_team']} {r['away_elo']:.0f}")
    print()
    print(f"  {r['home_team']:<30} {r['home_win_prob']*100:.1f}%")
    print(f"  {r['away_team']:<30} {r['away_win_prob']*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
