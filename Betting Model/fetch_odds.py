"""
fetch_odds.py
Fetches historical 1X2 rugby union odds from The Odds API.

Setup (one-time):
  1. Register free at https://the-odds-api.com  (takes 30 seconds)
  2. Copy your API key
  3. Set it: set ODDS_API_KEY=your_key_here   (Windows)
             export ODDS_API_KEY=your_key_here  (Mac/Linux)
  OR pass it directly: python fetch_odds.py --api-key YOUR_KEY

Credit cost:
  The Odds API gives 500 free credits/month on the free tier.
  Historical odds cost ~1 credit per 10 events queried.
  A full backtest (2020-2025, ~1,500 matches) costs roughly 200-400 credits.
  Extra credits are $10 per 10,000 — very cheap.

Usage:
  python fetch_odds.py --api-key YOUR_KEY
  python fetch_odds.py --api-key YOUR_KEY --from-year 2022 --to-year 2025
  python fetch_odds.py --api-key YOUR_KEY --sports six-nations rugby-championship

Output: odds_raw.csv
  Columns: Date, Home_Team, Away_Team, Home_Odds, Draw_Odds, Away_Odds,
           Competition, Source
"""

import argparse
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://api.the-odds-api.com/v4"

# Rugby union sport keys on The Odds API
SPORT_KEYS = {
    "six-nations":          "rugbyunion_six_nations",
    "rugby-championship":   "rugbyunion_the_rugby_championship",
    "world-cup":            "rugbyunion_world_cup",
    "epcr-champions-cup":   "rugbyunion_epcr_champions_cup",
    "premiership":          "rugbyunion_premiership",
    "super-rugby":          "rugbyunion_super_rugby_pacific",
    "united-rugby":         "rugbyunion_united_rugby_championship",
    "autumn-nations":       "rugbyunion_autumn_nations_series",
}

ALL_SPORT_KEYS = list(SPORT_KEYS.values())


def get_available_sports(api_key: str) -> list:
    """Return all active rugby-union sports from the API."""
    resp = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": api_key, "all": "true"},
        timeout=15,
    )
    resp.raise_for_status()
    sports = resp.json()
    rugby = [s for s in sports if "rugbyunion" in s.get("key", "")]
    return rugby


def get_historical_odds(api_key: str, sport_key: str, date_str: str,
                        regions: str = "uk,eu") -> list[dict]:
    """
    Fetch historical 1X2 odds snapshot for a sport on a given date.
    date_str: ISO 8601 e.g. "2023-02-04T12:00:00Z"
    """
    resp = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds-history",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h",
            "date": date_str,
            "oddsFormat": "decimal",
        },
        timeout=15,
    )
    if resp.status_code == 401:
        raise ValueError("Invalid API key. Check your ODDS_API_KEY.")
    if resp.status_code == 402:
        raise ValueError("Insufficient credits. Top up at the-odds-api.com.")
    if resp.status_code == 422:
        return []  # No data for this sport/date
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"    [{sport_key} | {date_str[:10]}] {remaining} credits remaining")
    return resp.json()


def parse_event(event: dict, sport_key: str) -> dict | None:
    """Parse a single event from The Odds API into our format."""
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")

    # Parse date (strip time component for matching)
    commence = event.get("commence_time", "")
    if not commence:
        return None
    try:
        date_obj = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        date_str = date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None

    # Find the best bookmaker (prefer Pinnacle for sharp odds, else first available)
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    def get_odds_from(bookmaker: dict) -> tuple[float, float, float] | None:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            home_odds = outcomes.get(home_team)
            away_odds = outcomes.get(away_team)
            draw_odds = outcomes.get("Draw")
            if home_odds and away_odds:
                return home_odds, draw_odds or 0.0, away_odds
        return None

    # Try Pinnacle first (sharpest odds)
    pinnacle = next((b for b in bookmakers if "pinnacle" in b.get("key", "").lower()), None)
    odds = get_odds_from(pinnacle) if pinnacle else None

    # Fall back to first bookmaker with data
    if odds is None:
        for bm in bookmakers:
            odds = get_odds_from(bm)
            if odds:
                break

    if odds is None:
        return None

    home_odds, draw_odds, away_odds = odds

    return {
        "Date":        date_str,
        "Home_Team":   home_team,
        "Away_Team":   away_team,
        "Home_Odds":   round(home_odds, 3),
        "Draw_Odds":   round(draw_odds, 3),
        "Away_Odds":   round(away_odds, 3),
        "Competition": sport_key.replace("rugbyunion_", "").replace("_", " ").title(),
        "Source":      "the-odds-api.com",
    }


def fetch_season(api_key: str, sport_key: str, year: int) -> list[dict]:
    """
    Fetch historical odds for a full season by sampling one snapshot per week.
    Rugby union seasons run roughly Feb-Nov for most competitions.
    """
    rows = []
    seen_events = set()

    # Sample every 7 days through the year
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%dT12:00:00Z")
        try:
            events = get_historical_odds(api_key, sport_key, date_str)
        except ValueError as e:
            print(f"  ERROR: {e}")
            return rows
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            time.sleep(2)
            current += timedelta(days=7)
            continue

        for event in events:
            eid = event.get("id", "")
            if eid in seen_events:
                continue
            seen_events.add(eid)
            parsed = parse_event(event, sport_key)
            if parsed:
                rows.append(parsed)

        current += timedelta(days=7)
        time.sleep(0.5)  # polite rate limiting

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key",   default=os.environ.get("ODDS_API_KEY", ""),
                        help="The Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--from-year", type=int, default=2020)
    parser.add_argument("--to-year",   type=int, default=datetime.today().year)
    parser.add_argument("--sports",    nargs="+",
                        choices=list(SPORT_KEYS.keys()) + ["all"],
                        default=["all"],
                        help="Which competitions to fetch")
    args = parser.parse_args()

    if not args.api_key:
        print("""
ERROR: No API key provided.

To get a free API key:
  1. Go to https://the-odds-api.com
  2. Click 'Get API Key' (free, no credit card needed)
  3. Run: set ODDS_API_KEY=your_key_here
     Then: python fetch_odds.py

Note: Historical odds cost credits (~200-400 for a full backtest).
Extra credits: $10 per 10,000 at the-odds-api.com/pricing
        """)
        return

    # Determine which sports to fetch
    if "all" in args.sports:
        target_sport_keys = ALL_SPORT_KEYS
    else:
        target_sport_keys = [SPORT_KEYS[s] for s in args.sports]

    print(f"Fetching odds for {args.from_year}-{args.to_year}")
    print(f"Sports: {', '.join(target_sport_keys)}")
    print()

    # Check available sports
    try:
        available = get_available_sports(args.api_key)
        available_keys = {s["key"] for s in available}
        print(f"API has {len(available)} rugby-union competitions available")
    except Exception as e:
        print(f"Could not list sports: {e}")
        available_keys = set(target_sport_keys)

    all_rows = []
    for sport_key in target_sport_keys:
        if sport_key not in available_keys:
            print(f"  Skipping {sport_key} (not in API)")
            continue

        for year in range(args.from_year, args.to_year + 1):
            print(f"  {sport_key} / {year}...")
            rows = fetch_season(args.api_key, sport_key, year)
            all_rows.extend(rows)
            print(f"    {len(rows)} matches found")

    if not all_rows:
        print("\nNo odds data returned. Check your API key and credit balance.")
        return

    df = pd.DataFrame(all_rows)
    df.drop_duplicates(subset=["Date", "Home_Team", "Away_Team"], inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)

    out_path = "odds_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} matches to {out_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(df["Competition"].value_counts().to_string())
