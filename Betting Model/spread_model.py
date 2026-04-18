"""
spread_model.py
Regression model predicting the match margin (Home_Score - Away_Score).

Why this matters for betting:
  - Asian handicap / spread markets: bookie sets a line (e.g. NZ -12.5),
    you bet whether the actual margin beats or misses the line.
  - If our model predicts +18 and the line is -12.5, we back NZ.

Models:
  - Linear Regression (baseline, interpretable)
  - XGBoost Regressor (main model)

Walk-forward backtest:
  - Train on games up to year T-1, predict year T (2015-2025)
  - For each game: record predicted margin vs actual margin
  - Handicap simulation: bet if model disagrees with a synthetic line

Outputs:
  spread_model.json       - trained XGBoost regressor
  spread_predictions.csv  - per-game predicted vs actual margins
  spread_summary.csv      - year-by-year accuracy stats

Usage:
  python spread_model.py
  python spread_model.py --start 2015 --end 2025 --handicap-threshold 5
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURES_PATH = "features.csv"

FEATURE_COLS = [
    "elo_diff",
    "home_prob_elo",
    "home_form_5",
    "home_form_10",
    "away_form_5",
    "away_form_10",
    "home_elo_momentum",
    "away_elo_momentum",
    "h2h_home_winrate",
    "h2h_n_games",
    "rankings_weight",
    "home_experience",
    "away_experience",
]

TARGET = "Margin"   # Home_Score - Away_Score (from ELO.csv via features.csv)


def train_linear(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)
    return pipe


def train_xgb_reg(X, y):
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, verbose=False)
    return model


def eval_regression(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Winner prediction accuracy (sign of predicted margin = actual winner)
    correct_winner = ((np.sign(y_pred) == np.sign(y_true)) |
                      ((y_pred == 0) & (y_true == 0))).mean()
    print(f"\n{name}")
    print(f"  MAE:             {mae:.2f} points  (avg miss per game)")
    print(f"  RMSE:            {rmse:.2f} points")
    print(f"  R²:              {r2:.4f}")
    print(f"  Winner accuracy: {correct_winner*100:.1f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WinnerAcc": correct_winner}


def simulate_handicap(y_true: np.ndarray, y_pred: np.ndarray,
                      synthetic_line: np.ndarray,
                      threshold: float, flat_stake: float = 1.0) -> dict:
    """
    Simulate Asian handicap betting where the bookmaker line is synthetic_line.
    We bet home (+line) when model predicts actual_margin > line + threshold.
    We bet away (-line) when model predicts actual_margin < line - threshold.
    """
    # model thinks home stronger than line says
    bet_home = (y_pred - synthetic_line) > threshold
    # model thinks away stronger than line says
    bet_away = (synthetic_line - y_pred) > threshold

    # Resolve bets (assume -1.1 / +1.1 or fair at 1.909 for AH bets)
    ah_odds = 1.909  # typical Asian handicap odds

    home_wins_cover  = y_true > synthetic_line
    away_wins_cover  = y_true < synthetic_line
    push             = y_true == synthetic_line

    home_pnl = np.where(bet_home,
        np.where(home_wins_cover, flat_stake * (ah_odds - 1),
            np.where(push, 0.0, -flat_stake)), 0.0)
    away_pnl = np.where(bet_away,
        np.where(away_wins_cover, flat_stake * (ah_odds - 1),
            np.where(push, 0.0, -flat_stake)), 0.0)

    n_bets   = bet_home.sum() + bet_away.sum()
    total_pnl = home_pnl.sum() + away_pnl.sum()
    total_staked = (bet_home.sum() + bet_away.sum()) * flat_stake

    roi = total_pnl / total_staked * 100 if total_staked > 0 else 0.0

    return {
        "n_bets": int(n_bets),
        "n_home": int(bet_home.sum()),
        "n_away": int(bet_away.sum()),
        "pnl":    round(total_pnl, 2),
        "roi%":   round(roi, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",              type=int,   default=2015)
    parser.add_argument("--end",                type=int,   default=2025)
    parser.add_argument("--handicap-threshold", type=float, default=5.0,
                        help="Min points disagreement with line to place a bet")
    parser.add_argument("--train-until",        type=int,   default=2019)
    args = parser.parse_args()

    print("Loading features.csv...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

    # Margin = Home_Score - Away_Score (sign convention: positive = home win)
    df[TARGET] = df["Home_Score"] - df["Away_Score"]
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)

    # ── Single train/test split for model evaluation ───────────────────────────
    train_df = df[df["Year"] <= args.train_until]
    test_df  = df[df["Year"] > args.train_until]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET].values

    print(f"Train: {len(train_df):,} games (up to {args.train_until}) | "
          f"Test: {len(test_df):,} games\n")

    # Baseline: just predict the mean margin
    mean_baseline = np.full_like(y_test, y_train.mean())
    eval_regression("Baseline (predict mean margin)", y_test, mean_baseline)

    # Linear regression
    lr = train_linear(X_train, y_train)
    lr_pred = lr.predict(X_test)
    eval_regression("Ridge Regression", y_test, lr_pred)

    # XGBoost regressor
    xgb_model = train_xgb_reg(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    eval_regression("XGBoost Regressor", y_test, xgb_pred)

    # Feature importance
    fi = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": xgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("\nTop feature importances:")
    print(fi.to_string(index=False))

    # ── Walk-forward backtest ──────────────────────────────────────────────────
    print(f"\n--- Walk-Forward Backtest ({args.start}-{args.end}) ---")

    all_preds = []
    year_rows = []

    for test_year in range(args.start, args.end + 1):
        train = df[df["Year"] < test_year]
        test  = df[df["Year"] == test_year]

        if len(train) < 500 or len(test) == 0:
            continue

        X_tr = train[FEATURE_COLS].values
        y_tr = train[TARGET].values
        X_te = test[FEATURE_COLS].values
        y_te = test[TARGET].values

        model = train_xgb_reg(X_tr, y_tr)
        preds = model.predict(X_te)

        mae  = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        win_acc = ((np.sign(preds) == np.sign(y_te)) | (preds == 0)).mean()

        # Synthetic handicap line: use ELO-implied expected margin
        # ELO gives P(home win). Rough margin estimate: ELO_diff / 20 * home_advantage
        elo_prob = test["home_prob_elo"].values
        # Convert win prob to expected margin (rough heuristic: each 1pp = ~0.4 points)
        synthetic_line = (elo_prob - 0.5) * 40  # rough ELO-based line

        ah = simulate_handicap(y_te, preds, synthetic_line,
                               threshold=args.handicap_threshold)

        test_out = test[["Date", "Year", "Home_Team", "Away_Team",
                          "Home_Score", "Away_Score", "home_prob_elo"]].copy()
        test_out["Margin_Actual"]    = y_te
        test_out["Margin_Predicted"] = preds.round(1)
        test_out["Margin_Error"]     = (preds - y_te).round(1)
        all_preds.append(test_out)

        year_rows.append({
            "Year":       test_year,
            "N_games":    len(test),
            "MAE":        round(mae, 2),
            "RMSE":       round(rmse, 2),
            "Winner_Acc": round(win_acc, 4),
            "AH_Bets":    ah["n_bets"],
            "AH_PnL":     ah["pnl"],
            "AH_ROI%":    ah["roi%"],
        })

        print(f"  {test_year}: MAE={mae:.1f}pts | winner_acc={win_acc*100:.1f}% | "
              f"AH {ah['n_bets']} bets, ROI {ah['roi%']:+.1f}%")

    # ── Save outputs ───────────────────────────────────────────────────────────
    if all_preds:
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_df.to_csv("spread_predictions.csv", index=False)
        print(f"\nSaved spread_predictions.csv ({len(preds_df):,} games)")

    summary_df = pd.DataFrame(year_rows)
    summary_df.to_csv("spread_summary.csv", index=False)
    print("Saved spread_summary.csv")

    # Save the model trained on full training set
    xgb_model.save_model("spread_model.json")
    print("Saved spread_model.json")

    # ── Aggregate AH backtest stats ────────────────────────────────────────────
    if year_rows:
        total_bets = sum(r["AH_Bets"] for r in year_rows)
        total_pnl  = sum(r["AH_PnL"] for r in year_rows)
        total_roi  = total_pnl / (total_bets * 1.0) * 100 if total_bets > 0 else 0

        print(f"\nAggregate handicap betting ({args.start}-{args.end}, "
              f"threshold={args.handicap_threshold}pts):")
        print(f"  Total AH bets: {total_bets}")
        print(f"  Total P&L:     {total_pnl:+.2f} units")
        print(f"  ROI:           {total_roi:+.2f}%")
        print()
        print("Note: AH odds assumed at 1.909 (typical -110 / standard AH).")
        print("Synthetic line uses ELO-implied margin — real bookie lines will differ.")
        print("To use real lines, add a 'Bookmaker_Line' column to spread_predictions.csv")
        print("and re-run simulate_handicap() with actual lines.")


if __name__ == "__main__":
    main()
