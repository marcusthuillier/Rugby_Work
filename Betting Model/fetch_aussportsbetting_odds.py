"""
fetch_aussportsbetting_odds.py
Parses the historical Super Rugby odds Excel file from aussportsbetting.com
and converts it into odds_raw.csv — the same format expected by merge_odds.py.

Manual step required:
    1. Go to https://www.aussportsbetting.com/data/historical-rugby-union-results-and-odds-data/
    2. Download the Excel file
    3. Save it as rugby_union.xlsx in this directory (Betting Model/)

Then run:
    python fetch_aussportsbetting_odds.py
    python merge_odds.py
    python real_odds_backtest.py

Output: odds_raw.csv
  Columns: Date, Home_Team, Away_Team, Home_Odds, Draw_Odds, Away_Odds, Competition, Source
"""

import os
import sys
import pandas as pd

IN_PATH  = "rugby_union.xlsx"
OUT_PATH = "odds_raw.csv"

# Columns the aussportsbetting sheet typically uses.
# We'll detect them by lowercase matching so minor header changes don't break things.
DATE_COLS     = ["date"]
HOME_COLS     = ["home team", "home"]
AWAY_COLS     = ["away team", "away"]
HOME_ODD_COLS = ["odds home", "home win", "h odds", "1", "home_odds", "oddsw1"]
DRAW_ODD_COLS = ["odds draw", "draw", "x odds", "draw odds", "oddsx"]
AWAY_ODD_COLS = ["odds away", "away win", "a odds", "2", "away_odds", "oddsw2"]


def find_col(df_cols: list[str], candidates: list[str]) -> str | None:
    """Return first column name (case-insensitive) that matches a candidate."""
    lower_map = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def main():
    if not os.path.exists(IN_PATH):
        print(f"""
ERROR: {IN_PATH} not found.

Steps to get it:
  1. Open in your browser:
     https://www.aussportsbetting.com/data/historical-rugby-union-results-and-odds-data/
  2. Download the Excel file
  3. Save it as:  {os.path.abspath(IN_PATH)}
  4. Re-run this script
""")
        sys.exit(1)

    print(f"Reading {IN_PATH}...")
    try:
        xl = pd.ExcelFile(IN_PATH)
        print(f"  Sheets found: {xl.sheet_names}")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)

    all_rows = []

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        cols = df.columns.tolist()
        cols_lower = [c.lower() for c in cols]

        print(f"\n  Sheet '{sheet}': {len(df)} rows, columns: {cols[:10]}")

        date_col  = find_col(cols, DATE_COLS)
        home_col  = find_col(cols, HOME_COLS)
        away_col  = find_col(cols, AWAY_COLS)
        h_odd_col = find_col(cols, HOME_ODD_COLS)
        d_odd_col = find_col(cols, DRAW_ODD_COLS)
        a_odd_col = find_col(cols, AWAY_ODD_COLS)

        missing = [n for n, c in [("Date", date_col), ("Home", home_col),
                                   ("Away", away_col), ("Home odds", h_odd_col),
                                   ("Away odds", a_odd_col)] if c is None]
        if missing:
            print(f"  Skipping — could not find columns: {missing}")
            print(f"  Available: {cols}")
            continue

        for _, row in df.iterrows():
            try:
                date_val = pd.to_datetime(row[date_col], dayfirst=True)
                if pd.isna(date_val):
                    continue
                date_str = date_val.strftime("%Y-%m-%d")
            except Exception:
                continue

            home = str(row[home_col]).strip()
            away = str(row[away_col]).strip()
            if not home or not away or home == "nan" or away == "nan":
                continue

            try:
                home_odds = float(row[h_odd_col])
                away_odds = float(row[a_odd_col])
            except (ValueError, TypeError):
                continue

            try:
                draw_odds = float(row[d_odd_col]) if d_odd_col else 0.0
            except (ValueError, TypeError):
                draw_odds = 0.0

            if home_odds <= 1.0 or away_odds <= 1.0:
                continue  # clearly bad data

            all_rows.append({
                "Date":        date_str,
                "Home_Team":   home,
                "Away_Team":   away,
                "Home_Odds":   round(home_odds, 3),
                "Draw_Odds":   round(draw_odds, 3),
                "Away_Odds":   round(away_odds, 3),
                "Competition": sheet,
                "Source":      "aussportsbetting.com",
            })

    if not all_rows:
        print("\nNo rows parsed. Check the column names above and update the mapping in this script.")
        sys.exit(1)

    out = pd.DataFrame(all_rows)
    out = out.drop_duplicates(subset=["Date", "Home_Team", "Away_Team"])
    out = out.sort_values("Date").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(out)} matches to {OUT_PATH}")
    print(f"Date range: {out['Date'].min()} to {out['Date'].max()}")
    print(f"Competitions:")
    print(out.groupby("Competition").size().to_string())
    print(f"\nNext steps:")
    print(f"  python merge_odds.py")
    print(f"  python real_odds_backtest.py")


if __name__ == "__main__":
    main()
