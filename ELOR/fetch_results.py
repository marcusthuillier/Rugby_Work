"""
fetch_results.py
Fetches new international rugby XV match results from the World Rugby API
and appends them to Datasets/fixtures.csv, filling the gap to today's date.

Usage:
    python fetch_results.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# World Rugby PulseLive API
BASE_URL = "https://api.wr-rims-prod.pulselive.com/rugby/v3/match"

# Only include men's XV international rugby
INCLUDED_SPORTS = {"MRU"}

# Keywords in competition names that indicate non-XV or non-senior rugby
EXCLUDED_KEYWORDS = [
    "sevens", "7s", "women", "u20", "under 20", "under-20",
    "girls", "olympic", "youth", "junior", "u18", "under 18",
    "u19", "under 19",
]


def should_exclude(match: dict) -> bool:
    """Return True if this match should be excluded from the dataset."""
    sport = match.get("sport", "")
    if sport not in INCLUDED_SPORTS:
        return True

    competition = match.get("competition", "").lower()
    if any(kw in competition for kw in EXCLUDED_KEYWORDS):
        return True

    return False


def parse_match(match: dict) -> dict | None:
    """Parse a World Rugby API match object into the fixtures.csv row format.

    Returns None if the match cannot be parsed or should be skipped.
    """
    if match.get("status") != "C":
        return None  # Not completed

    if should_exclude(match):
        return None

    teams = match.get("teams", [])
    scores = match.get("scores", [])

    if len(teams) < 2 or len(scores) < 2:
        return None

    try:
        home_score = int(scores[0])
        away_score = int(scores[1])
    except (ValueError, TypeError):
        return None

    date_label = match.get("time", {}).get("label", "")
    if not date_label:
        return None

    try:
        date_obj = datetime.strptime(date_label, "%Y-%m-%d")
    except ValueError:
        return None

    # Format date as M/D/YYYY (matches recent entries in fixtures.csv)
    date_str = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"

    return {
        "Year": date_obj.year,
        "Home_Team": teams[0]["name"],
        "Home_Score": home_score,
        "Away_Score": away_score,
        "Away_Team": teams[1]["name"],
        "Date": date_str,
    }


def fetch_matches(start_date: str | None, end_date: str) -> list[dict]:
    """Fetch all completed XV rugby matches up to end_date.

    Args:
        start_date: ISO date string e.g. "2021-06-14", or None for full history.
        end_date: ISO date string e.g. "2026-03-29"

    Returns:
        List of match dicts in fixtures.csv format.
    """
    all_matches = []
    page = 1

    while True:
        params = {
            "language": "en",
            "pageSize": 100,
            "page": page,
            "sort": "asc",
            "statuses": "C",
            "endDate": end_date,
        }
        if start_date:
            params["startDate"] = start_date

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Error on page {page}: {e}")
            break

        page_info = data.get("pageInfo", {})
        num_pages = page_info.get("numPages", 1)
        total = page_info.get("numEntries", 0)
        content = data.get("content", [])

        if not content:
            break

        for match in content:
            row = parse_match(match)
            if row is not None:
                all_matches.append(row)

        print(f"  Page {page:>3}/{num_pages} | {len(all_matches):>5} matches collected "
              f"(total API entries: {total})")

        if page >= num_pages:
            break

        page += 1
        time.sleep(0.2)  # Polite rate limiting

    return all_matches


def get_last_date(df: pd.DataFrame) -> str:
    """Extract the most recent date from fixtures.csv as an ISO date string.

    Tries multiple date formats. Falls back to using the Year column.
    Returns a date string like "2021-06-13".
    """
    FORMATS = ["%m/%d/%Y", "%d/%m/%Y", "%d %b %Y", "%Y-%m-%d", "%-d %b %Y"]

    latest = None

    for val in df["Date"]:
        parsed = None
        for fmt in FORMATS:
            try:
                parsed = datetime.strptime(str(val).strip(), fmt)
                break
            except ValueError:
                continue

        if parsed and (latest is None or parsed > latest):
            latest = parsed

    if latest:
        return latest.strftime("%Y-%m-%d")

    # Fallback: use max year
    max_year = int(df["Year"].max())
    return f"{max_year}-12-31"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true",
                        help="Full rebuild: fetch all history from the API and replace fixtures.csv")
    args = parser.parse_args()

    fixtures_path = os.path.join("Datasets", "fixtures.csv")
    end_date = datetime.today().strftime("%Y-%m-%d")

    if args.replace:
        # Pull the complete history from the API and replace the file
        print("Full rebuild mode: fetching all completed men's XV matches from the API...")
        print(f"End date: {end_date}")
        print()
        all_matches = fetch_matches(start_date=None, end_date=end_date)
        if not all_matches:
            print("No matches returned from API.")
            return
        df = pd.DataFrame(all_matches, columns=["Year", "Home_Team", "Home_Score", "Away_Score", "Away_Team", "Date"])
        df.drop_duplicates(subset=["Date", "Home_Team", "Away_Team"], keep="first", inplace=True)
        df.to_csv(fixtures_path, index=False)
        print(f"\nSaved fixtures.csv: {len(df)} total matches ({df['Year'].min()}-{df['Year'].max()})")
        return

    # Default: incremental update (append only new matches)
    if not os.path.exists(fixtures_path):
        print(f"Error: {fixtures_path} not found. Run from the ELOR directory.")
        return

    df_existing = pd.read_csv(fixtures_path)
    print(f"Existing fixtures.csv: {len(df_existing)} matches (up to {df_existing['Year'].max()})")

    last_date = get_last_date(df_existing)
    start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching new matches: {start_date} to {end_date}")
    print()

    new_matches = fetch_matches(start_date=start_date, end_date=end_date)

    if not new_matches:
        print("\nNo new matches found. fixtures.csv is already up to date.")
        return

    df_new = pd.DataFrame(new_matches, columns=["Year", "Home_Team", "Home_Score", "Away_Score", "Away_Team", "Date"])
    print(f"\nFetched {len(df_new)} new matches")

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    before = len(df_combined)
    df_combined.drop_duplicates(subset=["Date", "Home_Team", "Away_Team"], keep="first", inplace=True)
    if len(df_combined) < before:
        print(f"Removed {before - len(df_combined)} duplicate entries")

    df_combined.to_csv(fixtures_path, index=False)
    print(f"\nSaved fixtures.csv: {len(df_combined)} total matches "
          f"({df_combined['Year'].min()}-{df_combined['Year'].max()})")
    print(f"Added {len(df_combined) - len(df_existing)} net new matches")


if __name__ == "__main__":
    main()
