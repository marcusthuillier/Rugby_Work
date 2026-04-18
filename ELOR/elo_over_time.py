"""
elo_over_time.py
Builds a long-format ELO history table: one row per team per game played.

Outputs: elo_history.csv  (columns: Date, Year, Team, ELO)
"""

import pandas as pd
import os

ELO_PATH = os.path.join("Datasets", "ELO.csv")
OUT_PATH = "elo_history.csv"

DATE_FORMATS = ["%m/%d/%Y", "%d/%m/%Y", "%d %b %Y", "%Y-%m-%d"]


def parse_date(val: str) -> pd.Timestamp | None:
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(val, format=fmt)
        except ValueError:
            continue
    return None


def main():
    df = pd.read_csv(ELO_PATH)

    # Parse dates
    df["ParsedDate"] = df["Date"].apply(lambda v: parse_date(str(v).strip()))
    df = df.dropna(subset=["ParsedDate"])
    df = df.sort_values("ParsedDate").reset_index(drop=True)

    home_rows = df[["ParsedDate", "Year", "Home_Team", "Home_Rating_Updated"]].copy()
    home_rows.columns = ["Date", "Year", "Team", "ELO"]

    away_rows = df[["ParsedDate", "Year", "Away_Team", "Away_Rating_Updated"]].copy()
    away_rows.columns = ["Date", "Year", "Team", "ELO"]

    history = pd.concat([home_rows, away_rows], ignore_index=True)
    history["ELO"] = history["ELO"].round(1)
    history = history.sort_values(["Team", "Date"]).reset_index(drop=True)

    history.to_csv(OUT_PATH, index=False)

    last_date = history["Date"].max().date()
    with open("last_updated.txt", "w") as f:
        f.write(str(last_date))

    print(f"Saved {len(history):,} rows to {OUT_PATH}")
    print(f"Teams covered: {history['Team'].nunique()}")
    print(f"Date range: {history['Date'].min().date()} to {last_date}")
    print(f"Wrote last_updated.txt: {last_date}")


if __name__ == "__main__":
    main()
