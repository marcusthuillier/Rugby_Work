"""
six_nations_live.py
Fetches upcoming Six Nations fixtures from The Odds API and compares
ELO-implied win probabilities against current bookmaker lines to surface value bets.

Requires: ODDS_API_KEY environment variable (or --api-key flag)
          ../ELOR/Datasets/get_rank.csv  (current ELO ratings)

Output: six_nations_picks.csv
  Columns: Date, Home, Away, Home_ELO, Away_ELO,
           ELO_Home_Prob, Bookie_Home_Odds, Bookie_Away_Odds,
           Bookie_Home_Prob, Bookie_Away_Prob,
           Home_Edge, Away_Edge, Pick, Pick_Odds, Pick_Edge

Usage:
    set ODDS_API_KEY=your_key_here
    python six_nations_live.py

    python six_nations_live.py --api-key YOUR_KEY
    python six_nations_live.py --api-key YOUR_KEY --min-edge 0.03
"""

import argparse
import os
import math
import requests
import pandas as pd
from datetime import datetime, timezone

SPORT_KEY     = "rugbyunion_six_nations"
BASE_URL      = "https://api.the-odds-api.com/v4"
ELO_PATH      = os.path.join("..", "ELOR", "Datasets", "get_rank.csv")
OUT_PATH      = "six_nations_picks.csv"
HOME_ADVANTAGE = 75.0  # ELO points — matches ELOR system

# Map from The Odds API team names to our ELO team names
TEAM_ALIASES = {
    "England Rugby":  "England",
    "France Rugby":   "France",
    "Ireland Rugby":  "Ireland",
    "Italy Rugby":    "Italy",
    "Scotland Rugby": "Scotland",
    "Wales Rugby":    "Wales",
    "England":        "England",
    "France":         "France",
    "Ireland":        "Ireland",
    "Italy":          "Italy",
    "Scotland":       "Scotland",
    "Wales":          "Wales",
}


def elo_prob(home_elo: float, away_elo: float, neutral: bool = False) -> float:
    """ELO win probability for home team. Adds HOME_ADVANTAGE unless neutral venue."""
    adj = 0.0 if neutral else HOME_ADVANTAGE
    return 1.0 / (1 + math.pow(10, (away_elo - (home_elo + adj)) / 400))


def fetch_upcoming(api_key: str) -> list[dict]:
    """Fetch upcoming Six Nations fixtures with best available odds."""
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey":     api_key,
            "regions":    "uk,eu",
            "markets":    "h2h",
            "oddsFormat": "decimal",
        },
        timeout=15,
    )
    if resp.status_code == 401:
        raise ValueError("Invalid API key.")
    if resp.status_code == 422:
        print("No upcoming Six Nations fixtures found in the API right now.")
        return []
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"Credits remaining: {remaining}")
    return resp.json()


def best_odds(event: dict, home_team: str, away_team: str) -> tuple[float, float, float] | None:
    """Return (home_odds, draw_odds, away_odds) from sharpest bookmaker available."""
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    def extract(bm):
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            h = prices.get(home_team)
            a = prices.get(away_team)
            d = prices.get("Draw", 0.0)
            if h and a:
                return h, d, a
        return None

    # Prefer Pinnacle (sharpest), then Bet365, then first available
    for preferred in ("pinnacle", "bet365", "williamhill"):
        bm = next((b for b in bookmakers if preferred in b.get("key", "").lower()), None)
        if bm:
            result = extract(bm)
            if result:
                return result

    for bm in bookmakers:
        result = extract(bm)
        if result:
            return result

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",   default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--min-edge",  type=float, default=0.02,
                        help="Minimum edge (model prob - implied prob) to flag as value")
    parser.add_argument("--neutral",   action="store_true",
                        help="Treat all venues as neutral (no home advantage adjustment)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set ODDS_API_KEY or use --api-key.")
        return

    # Load ELO ratings
    if not os.path.exists(ELO_PATH):
        print(f"ERROR: ELO ratings not found at {ELO_PATH}")
        return
    elo_df = pd.read_csv(ELO_PATH)
    elo_map = dict(zip(elo_df["Team"], elo_df["ELO"]))
    print(f"Loaded ELO ratings for {len(elo_map)} teams")

    # Fetch fixtures
    print(f"Fetching upcoming {SPORT_KEY} fixtures...")
    events = fetch_upcoming(args.api_key)

    if not events:
        print("No fixtures to process.")
        return

    rows = []
    for event in events:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = TEAM_ALIASES.get(home_raw, home_raw)
        away = TEAM_ALIASES.get(away_raw, away_raw)

        commence = event.get("commence_time", "")
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            date_str = commence

        home_elo = elo_map.get(home)
        away_elo = elo_map.get(away)

        if home_elo is None or away_elo is None:
            missing = home if home_elo is None else away
            print(f"  WARNING: No ELO rating for '{missing}' — skipping {home} vs {away}")
            continue

        odds = best_odds(event, home_raw, away_raw)
        if odds is None:
            print(f"  WARNING: No odds found for {home} vs {away} — skipping")
            continue

        home_odds, draw_odds, away_odds = odds
        bookie_home_prob = round(1.0 / home_odds, 4)
        bookie_away_prob = round(1.0 / away_odds, 4)

        elo_home_prob = round(elo_prob(home_elo, away_elo, neutral=args.neutral), 4)
        elo_away_prob = round(1.0 - elo_home_prob, 4)

        home_edge = round(elo_home_prob - bookie_home_prob, 4)
        away_edge = round(elo_away_prob - bookie_away_prob, 4)

        best_edge = max(home_edge, away_edge)
        if home_edge >= away_edge:
            pick = f"{home} (Home)"
            pick_odds = home_odds
            pick_edge = home_edge
        else:
            pick = f"{away} (Away)"
            pick_odds = away_odds
            pick_edge = away_edge

        value = "YES" if pick_edge >= args.min_edge else "no"

        rows.append({
            "Date":              date_str,
            "Home":              home,
            "Away":              away,
            "Home_ELO":          int(home_elo),
            "Away_ELO":          int(away_elo),
            "ELO_Home_Prob":     f"{elo_home_prob:.1%}",
            "Bookie_Home_Odds":  home_odds,
            "Bookie_Away_Odds":  away_odds,
            "Bookie_Home_Prob":  f"{bookie_home_prob:.1%}",
            "Bookie_Away_Prob":  f"{bookie_away_prob:.1%}",
            "Home_Edge":         f"{home_edge:+.1%}",
            "Away_Edge":         f"{away_edge:+.1%}",
            "Best_Pick":         pick,
            "Pick_Odds":         pick_odds,
            "Pick_Edge":         f"{pick_edge:+.1%}",
            "Value_Bet":         value,
        })

    if not rows:
        print("No rows to save.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} fixtures to {OUT_PATH}")
    print()
    print(df[["Date", "Home", "Away", "ELO_Home_Prob",
              "Bookie_Home_Prob", "Home_Edge", "Best_Pick", "Pick_Edge", "Value_Bet"]].to_string(index=False))

    value_bets = df[df["Value_Bet"] == "YES"]
    if not value_bets.empty:
        print(f"\n{'='*60}")
        print(f"VALUE BETS (edge >= {args.min_edge:.0%})")
        print(f"{'='*60}")
        print(value_bets[["Date", "Best_Pick", "Pick_Odds", "Pick_Edge"]].to_string(index=False))
    else:
        print(f"\nNo value bets found at {args.min_edge:.0%} edge threshold.")


if __name__ == "__main__":
    main()
