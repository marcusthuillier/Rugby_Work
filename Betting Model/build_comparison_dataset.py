"""
build_comparison_dataset.py
Builds the master comparison dataset for Super Rugby (2015-2026) combining:

  Real odds (aussportsbetting.com)
    - H2H closing odds (Home_Odds, Away_Odds)
    - Spread closing line (Real_Spread = e.g. -12.5 means home -12.5)
    - Total score line

  ELO-implied odds
    - Uses each team's ELO rating from elo_history.csv at match date
    - Computes win probability and implied spread

  XGBoost model odds (for games in the backtest window 2015-2025)
    - Loads trained model and generates predictions on the fly
    - Requires: model_xgb.json, calibrator.pkl, spread_model.json

Output: comparison_dataset.csv

Run: python build_comparison_dataset.py
"""

import math
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

EXCEL_PATH    = "super_rugby_raw.xlsx"
HISTORY_PATH  = os.path.join("..", "New Work", "elo_history.csv")
ELO_PATH      = os.path.join("..", "ELOR", "Datasets", "ELO.csv")
MODEL_PATH    = "model_xgb.json"
CALIBRATOR    = "calibrator.pkl"
SPREAD_MODEL  = "spread_model.json"
OUT_PATH      = "comparison_dataset.csv"

HOME_ADVANTAGE = 75
INITIAL_ELO    = 1500.0
SUPER_RUGBY_WEIGHT = 0.50   # competition weight for Super Rugby

FEATURE_COLS = [
    "elo_diff", "home_prob_elo",
    "home_form_5", "home_form_10",
    "away_form_5", "away_form_10",
    "home_elo_momentum", "away_elo_momentum",
    "h2h_home_winrate", "h2h_n_games",
    "rankings_weight",
    "home_experience", "away_experience",
]

TEAM_ALIASES = {
    "NSW Waratahs": "New South Wales Waratahs",
    "Waratahs":     "New South Wales Waratahs",
    "NSW":          "New South Wales Waratahs",
    "WA Force":     "Western Force",
    "Force":        "Western Force",
    "ACT Brumbies": "Brumbies",
    "Qld Reds":     "Queensland Reds",
    "QLD Reds":     "Queensland Reds",
    "Reds":         "Queensland Reds",
    "Melbourne":    "Melbourne Rebels",
    "Rebels":       "Melbourne Rebels",
    "Golden Lions": "Lions",
}


def resolve(name: str) -> str:
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def elo_prob(home_elo: float, away_elo: float) -> float:
    adj = home_elo + HOME_ADVANTAGE
    return 1.0 / (1 + math.pow(10, (away_elo - adj) / 400))


def no_vig(h_odds, a_odds):
    try:
        h, a = 1/float(h_odds), 1/float(a_odds)
        t = h + a
        return round(h/t, 4), round(a/t, 4)
    except Exception:
        return None, None


def elo_to_spread(prob: float) -> float:
    """Rough ELO → expected margin conversion."""
    return (prob - 0.5) * 40.0


def load_elo_history() -> dict[str, pd.DataFrame]:
    """Load elo_history.csv and return dict of team -> sorted DataFrame(Date, ELO)."""
    df = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")
    return {team: grp.reset_index(drop=True)
            for team, grp in df.groupby("Team")}


def get_elo_at(history: dict, team: str, before_date: pd.Timestamp) -> float:
    """Get a team's most recent ELO rating strictly before before_date."""
    if team not in history:
        return INITIAL_ELO
    grp = history[team]
    prior = grp[grp["Date"] < before_date]
    return float(prior.iloc[-1]["ELO"]) if len(prior) > 0 else INITIAL_ELO


def load_elo_game_history() -> pd.DataFrame:
    """Load ELO.csv for form/H2H feature computation."""
    DATE_FMTS = ["%m/%d/%Y", "%d/%m/%Y", "%d %b %Y", "%Y-%m-%d"]
    df = pd.read_csv(ELO_PATH)
    for fmt in DATE_FMTS:
        try:
            df["_date"] = pd.to_datetime(df["Date"], format=fmt, errors="coerce")
            if df["_date"].notna().mean() > 0.8:
                break
        except Exception:
            pass
    df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
    return df


def compute_form_h2h(game_hist: pd.DataFrame, home: str, away: str,
                     before_date: pd.Timestamp) -> dict:
    """Compute form and H2H features from historical games."""
    prior = game_hist[game_hist["_date"] < before_date]

    def win_rate(team: str, n: int) -> float:
        home_g = prior[prior["Home_Team"] == team].tail(n)[["Home_Away_Draw"]]
        away_g = prior[prior["Away_Team"] == team].tail(n)[["Home_Away_Draw"]]
        results = (
            list(home_g["Home_Away_Draw"]) +
            [1 - v for v in away_g["Home_Away_Draw"]]
        )
        results = results[-n:]
        return float(np.mean(results)) if results else 0.5

    def momentum(team: str, n: int = 5) -> float:
        home_g = prior[prior["Home_Team"] == team].tail(n)[["Home_Rating_Updated"]]
        away_g = prior[prior["Away_Team"] == team].tail(n)[["Away_Rating_Updated"]]
        vals = list(home_g["Home_Rating_Updated"]) + list(away_g["Away_Rating_Updated"])
        vals = vals[-n:]
        return float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0

    def h2h(home: str, away: str, n: int = 10):
        fwd = prior[(prior["Home_Team"] == home) & (prior["Away_Team"] == away)].tail(n)
        rev = prior[(prior["Home_Team"] == away) & (prior["Away_Team"] == home)].tail(n)
        results = (list(fwd["Home_Away_Draw"]) +
                   [1 - v for v in rev["Home_Away_Draw"]])
        return float(np.mean(results)) if results else 0.5, len(results)

    def exp(team: str) -> int:
        return len(prior[(prior["Home_Team"] == team) | (prior["Away_Team"] == team)])

    h2h_rate, h2h_n = h2h(home, away)

    return {
        "home_form_5":        win_rate(home, 5),
        "home_form_10":       win_rate(home, 10),
        "away_form_5":        win_rate(away, 5),
        "away_form_10":       win_rate(away, 10),
        "home_elo_momentum":  momentum(home),
        "away_elo_momentum":  momentum(away),
        "h2h_home_winrate":   h2h_rate,
        "h2h_n_games":        h2h_n,
        "home_experience":    exp(home),
        "away_experience":    exp(away),
    }


def main():
    print("Loading data...")
    excel = pd.read_excel(EXCEL_PATH, header=1)
    excel["Date"] = pd.to_datetime(excel["Date"], errors="coerce")
    excel = excel.dropna(subset=["Date", "Home Team", "Away Team"])
    excel = excel[excel["Date"].dt.year >= 2015].copy()
    excel = excel.sort_values("Date").reset_index(drop=True)
    print(f"  Super Rugby games (2015+): {len(excel):,}")

    elo_hist   = load_elo_history()
    game_hist  = load_elo_game_history()
    print(f"  ELO history entries: {sum(len(v) for v in elo_hist.values()):,}")
    print(f"  ELO game history:    {len(game_hist):,}")

    # Load models
    clf_model, calibrator, spread_model = None, None, None
    if all(os.path.exists(p) for p in [MODEL_PATH, CALIBRATOR, SPREAD_MODEL]):
        clf_model = xgb.XGBClassifier(); clf_model.load_model(MODEL_PATH)
        with open(CALIBRATOR, "rb") as f: calibrator = pickle.load(f)
        spread_model = xgb.XGBRegressor(); spread_model.load_model(SPREAD_MODEL)
        print("  Models loaded: win probability + spread")
    else:
        print("  Models not found (run model.py + spread_model.py first) — ELO only")

    rows = []
    print(f"\nProcessing {len(excel):,} games...")

    for i, ex in excel.iterrows():
        if i % 200 == 0:
            print(f"  {i}/{len(excel)}")

        date      = ex["Date"]
        home_raw  = str(ex["Home Team"])
        away_raw  = str(ex["Away Team"])
        home      = resolve(home_raw)
        away      = resolve(away_raw)

        # ── Real odds ──────────────────────────────────────────────────────────
        real_home_odds  = ex.get("Home Odds Close") or ex.get("Home Odds")
        real_away_odds  = ex.get("Away Odds Close") or ex.get("Away Odds")
        real_spread     = ex.get("Home Line Close")
        real_total      = ex.get("Total Score Close")

        real_home_prob, real_away_prob = no_vig(real_home_odds, real_away_odds)

        # ── ELO-implied ────────────────────────────────────────────────────────
        home_elo = get_elo_at(elo_hist, home, date)
        away_elo = get_elo_at(elo_hist, away, date)
        elo_home_prob  = round(elo_prob(home_elo, away_elo), 4)
        elo_away_prob  = round(1 - elo_home_prob, 4)
        elo_home_odds  = round(1 / elo_home_prob, 3)
        elo_away_odds  = round(1 / elo_away_prob, 3)
        elo_spread_est = round(elo_to_spread(elo_home_prob), 1)

        # ── Model predictions ──────────────────────────────────────────────────
        model_home_prob = model_away_prob = model_home_odds = model_away_odds = None
        model_spread    = None

        if clf_model is not None:
            form = compute_form_h2h(game_hist, home, away, date)
            elo_diff = home_elo - away_elo

            feat_vec = np.array([[
                elo_diff, elo_home_prob,
                form["home_form_5"], form["home_form_10"],
                form["away_form_5"], form["away_form_10"],
                form["home_elo_momentum"], form["away_elo_momentum"],
                form["h2h_home_winrate"], form["h2h_n_games"],
                SUPER_RUGBY_WEIGHT,
                form["home_experience"], form["away_experience"],
            ]])

            model_home_prob = round(float(calibrator.predict_proba(feat_vec)[0][1]), 4)
            model_away_prob = round(1 - model_home_prob, 4)
            model_home_odds = round(1 / model_home_prob, 3)  if model_home_prob > 0 else None
            model_away_odds = round(1 / model_away_prob, 3)  if model_away_prob > 0 else None
            model_spread    = round(float(spread_model.predict(feat_vec)[0]), 1)

        # ── Actual outcome ─────────────────────────────────────────────────────
        try:
            hs = float(ex["Home Score"]); as_ = float(ex["Away Score"])
            margin   = hs - as_
            home_win = 1 if margin > 0 else (0 if margin < 0 else None)
            result   = "H" if margin > 0 else ("A" if margin < 0 else "D")
        except Exception:
            hs = as_ = margin = home_win = result = None

        rows.append({
            "Date":           date.date(),
            "Year":           int(date.year),
            "Home_Team":      home,
            "Away_Team":      away,
            "Home_Score":     hs,
            "Away_Score":     as_,
            "Actual_Margin":  margin,
            "Result":         result,
            "Home_Win":       home_win,
            # Real odds
            "Real_Home_Odds": real_home_odds,
            "Real_Away_Odds": real_away_odds,
            "Real_Home_Prob": real_home_prob,
            "Real_Away_Prob": real_away_prob,
            "Real_Spread":    real_spread,
            "Real_Total":     real_total,
            # ELO
            "ELO_Home_ELO":   round(home_elo, 0),
            "ELO_Away_ELO":   round(away_elo, 0),
            "ELO_Home_Prob":  elo_home_prob,
            "ELO_Away_Prob":  elo_away_prob,
            "ELO_Home_Odds":  elo_home_odds,
            "ELO_Away_Odds":  elo_away_odds,
            "ELO_Spread":     elo_spread_est,
            # Model
            "Model_Home_Prob": model_home_prob,
            "Model_Away_Prob": model_away_prob,
            "Model_Home_Odds": model_home_odds,
            "Model_Away_Odds": model_away_odds,
            "Model_Spread":    model_spread,
            # Value indicators
            "Home_Edge_Model_vs_Real": (
                round(model_home_prob - real_home_prob, 4)
                if model_home_prob is not None and real_home_prob is not None else None),
            "Home_Edge_ELO_vs_Real": (
                round(elo_home_prob - real_home_prob, 4)
                if real_home_prob is not None else None),
            "Spread_Edge_Model_vs_Real": (
                round(model_spread - (-real_spread), 1)
                if model_spread is not None and real_spread is not None else None),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    has_all = df.dropna(subset=["Real_Home_Prob", "ELO_Home_Prob"])
    has_model = df.dropna(subset=["Model_Home_Prob"])
    print(f"\nSaved {len(df):,} games to {OUT_PATH}")
    print(f"  With real + ELO probs:     {len(has_all):,}")
    print(f"  With model predictions:    {len(has_model):,}")

    if len(has_all) > 0:
        print(f"\n--- Quick sanity check ({len(has_all)} games) ---")
        print(f"  Real home win prob (avg): {has_all['Real_Home_Prob'].mean()*100:.1f}%")
        print(f"  ELO  home win prob (avg): {has_all['ELO_Home_Prob'].mean()*100:.1f}%")
        if len(has_model) > 0:
            print(f"  Model home win prob (avg):{has_model['Model_Home_Prob'].mean()*100:.1f}%")
        print(f"  Actual home win rate:     {df['Home_Win'].mean()*100:.1f}%")
        print(f"\n  ELO vs Real correlation:  {has_all['ELO_Home_Prob'].corr(has_all['Real_Home_Prob']):.3f}")
        if len(has_model) > 0:
            m = has_model.dropna(subset=["Real_Home_Prob"])
            print(f"  Model vs Real correlation:{m['Model_Home_Prob'].corr(m['Real_Home_Prob']):.3f}")


if __name__ == "__main__":
    main()
