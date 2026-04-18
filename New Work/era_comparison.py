"""
era_comparison.py
Ranks the top teams by average ELO rating within each rugby era.

Eras:
  Pre-WW1      : up to 1914
  Interwar     : 1919-1939
  Post-WW2     : 1946-1986
  Professional : 1987-2003
  Modern       : 2004+

Outputs: era_rankings.csv  (columns: Era, Rank, Team, Avg_ELO)
"""

import pandas as pd
import os

ELO_PATH = os.path.join("..", "ELOR", "Datasets", "ELO.csv")
OUT_PATH = "era_rankings.csv"

ERAS = [
    ("Pre-WW1",       lambda y: y <= 1914),
    ("Interwar",      lambda y: 1919 <= y <= 1939),
    ("Post-WW2",      lambda y: 1946 <= y <= 1986),
    ("Professional",  lambda y: 1987 <= y <= 2003),
    ("Modern",        lambda y: y >= 2004),
]

TOP_N = 10


def main():
    df = pd.read_csv(ELO_PATH)

    # Build long-format ELO per game per team
    home = df[["Year", "Home_Team", "Home_Rating_Updated"]].copy()
    home.columns = ["Year", "Team", "ELO"]
    away = df[["Year", "Away_Team", "Away_Rating_Updated"]].copy()
    away.columns = ["Year", "Team", "ELO"]
    long = pd.concat([home, away], ignore_index=True)

    results = []
    for era_name, era_filter in ERAS:
        era_df = long[long["Year"].apply(era_filter)]
        if era_df.empty:
            continue
        avg = era_df.groupby("Team")["ELO"].mean().sort_values(ascending=False).head(TOP_N)
        for rank, (team, elo) in enumerate(avg.items(), start=1):
            results.append({"Era": era_name, "Rank": rank, "Team": team, "Avg_ELO": round(elo, 1)})

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved era rankings to {OUT_PATH}")

    for era_name, _ in ERAS:
        era_rows = out[out["Era"] == era_name]
        if era_rows.empty:
            continue
        print(f"\n{era_name}:")
        for _, row in era_rows.iterrows():
            print(f"  {row['Rank']:>2}. {row['Team']:<35} {row['Avg_ELO']:.0f}")


if __name__ == "__main__":
    main()
