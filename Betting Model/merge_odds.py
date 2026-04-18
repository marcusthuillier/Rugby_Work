"""
merge_odds.py
Joins odds_raw.csv (from fetch_odds.py) with backtest_results.csv on
Date + Home_Team + Away_Team.

Handles team name mismatches between our ELO data (World Rugby API names)
and The Odds API names via a fuzzy match + manual alias table.

Output: backtest_with_odds.csv
"""

import os
import pandas as pd
from difflib import get_close_matches

BACKTEST_PATH = "backtest_results.csv"
ODDS_PATH     = "odds_raw.csv"
OUT_PATH      = "backtest_with_odds.csv"

# Manual aliases: Odds API name -> World Rugby API name
# Add more here if you see unmatched teams in the output
TEAM_ALIASES: dict[str, str] = {
    "All Blacks":               "New Zealand",
    "New Zealand All Blacks":   "New Zealand",
    "Springboks":               "South Africa",
    "Wallabies":                "Australia",
    "Los Pumas":                "Argentina",
    "Les Bleus":                "France",
    "British and Irish Lions":  "British & Irish Lions",
    "British & Irish Lions":    "British & Irish Lions",
    "USA":                      "United States",
    "US Eagles":                "United States",
}


def normalise_name(name: str) -> str:
    """Apply manual aliases, then lowercase + strip for matching."""
    name = TEAM_ALIASES.get(name, name)
    return name.strip().lower()


def fuzzy_match(name: str, candidates: list[str], cutoff: float = 0.82) -> str | None:
    """Return best fuzzy match from candidates, or None if below cutoff."""
    norm = normalise_name(name)
    norm_candidates = {normalise_name(c): c for c in candidates}
    matches = get_close_matches(norm, list(norm_candidates.keys()),
                                n=1, cutoff=cutoff)
    return norm_candidates[matches[0]] if matches else None


def main():
    if not os.path.exists(BACKTEST_PATH):
        print(f"ERROR: {BACKTEST_PATH} not found. Run backtest.py first.")
        return
    if not os.path.exists(ODDS_PATH):
        print(f"ERROR: {ODDS_PATH} not found. Run fetch_odds.py first.")
        return

    print("Loading data...")
    bt = pd.read_csv(BACKTEST_PATH, parse_dates=["Date"])
    odds = pd.read_csv(ODDS_PATH, parse_dates=["Date"])

    # Normalise dates to date only (no time)
    bt["_date"]   = bt["Date"].dt.date
    odds["_date"] = odds["Date"].dt.date

    # Build odds lookup: (date, norm_home, norm_away) -> odds row
    odds_lookup: dict[tuple, dict] = {}
    for _, row in odds.iterrows():
        key = (row["_date"],
               normalise_name(row["Home_Team"]),
               normalise_name(row["Away_Team"]))
        odds_lookup[key] = {
            "Home_Odds": row["Home_Odds"],
            "Draw_Odds": row["Draw_Odds"],
            "Away_Odds": row["Away_Odds"],
            "Competition_Odds": row.get("Competition", ""),
        }

    all_bt_teams = set(bt["Home_Team"].unique()) | set(bt["Away_Team"].unique())

    matched = 0
    fuzzy_matched = 0
    unmatched = []

    home_odds_col  = []
    draw_odds_col  = []
    away_odds_col  = []
    comp_odds_col  = []

    for _, row in bt.iterrows():
        date = row["_date"]
        home = row["Home_Team"]
        away = row["Away_Team"]

        # Exact match (after alias resolution)
        key = (date, normalise_name(home), normalise_name(away))
        if key in odds_lookup:
            o = odds_lookup[key]
            home_odds_col.append(o["Home_Odds"])
            draw_odds_col.append(o["Draw_Odds"])
            away_odds_col.append(o["Away_Odds"])
            comp_odds_col.append(o["Competition_Odds"])
            matched += 1
            continue

        # Fuzzy: try different team names for this date
        date_entries = [(k, v) for k, v in odds_lookup.items() if k[0] == date]
        found = False
        for (d, oh, oa), v in date_entries:
            home_match = fuzzy_match(home, [h for (_, h, _) in odds_lookup if _ == date])
            away_match = fuzzy_match(away, [a for (_, _, a) in odds_lookup if _ == date])
            if (oh == normalise_name(home) or home_match == oh) and \
               (oa == normalise_name(away) or away_match == oa):
                home_odds_col.append(v["Home_Odds"])
                draw_odds_col.append(v["Draw_Odds"])
                away_odds_col.append(v["Away_Odds"])
                comp_odds_col.append(v["Competition_Odds"])
                fuzzy_matched += 1
                found = True
                break

        if not found:
            home_odds_col.append(None)
            draw_odds_col.append(None)
            away_odds_col.append(None)
            comp_odds_col.append(None)
            unmatched.append(f"{date} | {home} vs {away}")

    bt["Home_Odds"]       = home_odds_col
    bt["Draw_Odds"]       = draw_odds_col
    bt["Away_Odds"]       = away_odds_col
    bt["Competition_Odds"] = comp_odds_col

    bt.drop(columns=["_date"], inplace=True)

    with_odds = bt.dropna(subset=["Home_Odds"])
    total = len(bt)
    n_matched = matched + fuzzy_matched

    print(f"\nMatching results:")
    print(f"  Total backtest games:  {total}")
    print(f"  Exact matches:         {matched}")
    print(f"  Fuzzy matches:         {fuzzy_matched}")
    print(f"  Unmatched:             {len(unmatched)}  ({len(unmatched)/total*100:.1f}%)")
    print(f"  Match rate:            {n_matched/total*100:.1f}%")

    if unmatched:
        print(f"\nFirst 20 unmatched games:")
        for u in unmatched[:20]:
            print(f"  {u}")

    bt.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(bt)} rows to {OUT_PATH}")
    print(f"  ({len(with_odds)} have real odds, {total - len(with_odds)} will use ELO fallback)")


if __name__ == "__main__":
    main()
