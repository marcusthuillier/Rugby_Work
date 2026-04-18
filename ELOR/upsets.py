"""
upsets.py
Finds the greatest upsets in the dataset — games where the winner had the
lowest pre-match win probability.

Outputs: upsets.csv
"""

import pandas as pd
import os

ELO_PATH = os.path.join("Datasets", "ELO.csv")
OUT_PATH = "upsets.csv"


def main():
    df = pd.read_csv(ELO_PATH)

    # Drop draws (no upset possible)
    df = df[df["Home_Away_Draw"] != 0.5].copy()

    # Home win: Home_Away_Draw == 1, winner prob = Prob_Home
    # Away win: Home_Away_Draw == 0, winner prob = Prob_Away
    home_wins = df[df["Home_Away_Draw"] == 1.0].copy()
    home_wins["Winner"] = home_wins["Home_Team"]
    home_wins["Winner_Prob"] = home_wins["Prob_Home"]

    away_wins = df[df["Home_Away_Draw"] == 0.0].copy()
    away_wins["Winner"] = away_wins["Away_Team"]
    away_wins["Winner_Prob"] = away_wins["Prob_Away"]

    combined = pd.concat([home_wins, away_wins], ignore_index=True)

    # Upset score: how far below 0.5 the winner's probability was
    # Only games where winner was the underdog (prob < 0.5)
    upsets = combined[combined["Winner_Prob"] < 0.5].copy()
    upsets["Upset_Score"] = 0.5 - upsets["Winner_Prob"]

    upsets = upsets.sort_values("Upset_Score", ascending=False).head(50)

    out = upsets[["Date", "Home_Team", "Away_Team", "Home_Score", "Away_Score",
                  "Winner", "Winner_Prob", "Upset_Score"]].copy()
    out["Winner_Prob"] = out["Winner_Prob"].round(3)
    out["Upset_Score"] = out["Upset_Score"].round(3)

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} upsets to {OUT_PATH}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
