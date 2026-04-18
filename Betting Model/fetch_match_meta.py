"""
fetch_match_meta.py
Fetches enriched match metadata from the World Rugby API:
  - Competition name & World Rugby rankings weight (0=exhibition, 1=World Cup)
  - Venue country (used to infer neutral ground)

Saves: match_meta.csv
Columns: Date, Home_Team, Away_Team, Competition, Rankings_Weight, Venue_Country

Usage:
    python fetch_match_meta.py               # 2010 to today
    python fetch_match_meta.py --from 2000   # custom start year
"""

import argparse
import os
import time
import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api.wr-rims-prod.pulselive.com/rugby/v3/match"
INCLUDED_SPORTS = {"MRU"}
OUT_PATH = "match_meta.csv"


def fetch_year(year: int) -> list[dict]:
    rows = []
    page = 1
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    while True:
        params = {
            "language": "en",
            "pageSize": 100,
            "page": page,
            "sort": "asc",
            "statuses": "C",
            "startDate": start,
            "endDate": end,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Error (year {year}, page {page}): {e}")
            break

        content = data.get("content", [])
        page_info = data.get("pageInfo", {})
        num_pages = page_info.get("numPages", 1)

        if not content:
            break

        for m in content:
            if m.get("sport", "") not in INCLUDED_SPORTS:
                continue
            if m.get("status") != "C":
                continue

            teams = m.get("teams", [])
            if len(teams) < 2:
                continue

            date_label = m.get("time", {}).get("label", "")
            if not date_label:
                continue

            venue_country = (m.get("venue") or {}).get("country", "")

            # Rankings weight comes from the event (competition) metadata
            events = m.get("events", [])
            rankings_weight = 0.0
            if events:
                rankings_weight = events[0].get("rankingsWeight") or 0.0

            competition = m.get("competition", "")

            rows.append({
                "Date": date_label,
                "Home_Team": teams[0]["name"],
                "Away_Team": teams[1]["name"],
                "Competition": competition,
                "Rankings_Weight": float(rankings_weight),
                "Venue_Country": venue_country,
            })

        if page >= num_pages:
            break
        page += 1
        time.sleep(0.15)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_year", type=int, default=2010)
    args = parser.parse_args()

    current_year = datetime.today().year
    years = range(args.from_year, current_year + 1)

    all_rows = []
    for year in years:
        rows = fetch_year(year)
        all_rows.extend(rows)
        print(f"  {year}: {len(rows):>4} matches  (total so far: {len(all_rows)})")

    if not all_rows:
        print("No data fetched.")
        return

    df = pd.DataFrame(all_rows)
    df.drop_duplicates(subset=["Date", "Home_Team", "Away_Team"], inplace=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} matches to {OUT_PATH}")
    print(f"Rankings weight range: {df['Rankings_Weight'].min():.2f} - {df['Rankings_Weight'].max():.2f}")
    print(df.groupby("Competition")["Rankings_Weight"].first()
          .sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
