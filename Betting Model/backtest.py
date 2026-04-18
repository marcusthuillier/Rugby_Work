"""
backtest.py
Walk-forward backtest of the XGBoost betting model.

Strategy:
  - Train on all games up to year T-1
  - Predict games in year T
  - Roll forward one year at a time

Betting simulation:
  - Flat stake betting: bet 1 unit whenever model confidence > threshold
  - Kelly criterion betting: stake proportional to perceived edge
  - Simulate against ELO-implied odds (as a proxy for fair market odds)
    Real bookmaker odds include a margin (~5%), so real ROI would be lower.

Outputs:
  backtest_results.csv  - per-game predictions across all test years
  backtest_summary.csv  - year-by-year P&L summary

Usage:
    python backtest.py
    python backtest.py --start 2015 --end 2025 --threshold 0.55 --stake 1
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
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

TARGET = "home_win"


def train_xgb(X, y):
    if len(np.unique(y)) < 2:
        return None
    scale_pos_weight = max((y == 0).sum() / max((y == 1).sum(), 1), 0.1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, verbose=False)
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X, y)
    return cal


def kelly_stake(prob: float, implied_prob: float, max_stake: float = 5.0) -> float:
    """Kelly criterion stake. Returns 0 if no edge."""
    if implied_prob <= 0 or prob <= implied_prob:
        return 0.0
    odds = 1.0 / implied_prob  # decimal odds
    edge = prob - implied_prob
    f = edge / (odds - 1)      # Kelly fraction
    return min(max(f, 0), max_stake)


def simulate_bets(df: pd.DataFrame, prob_col: str, threshold: float,
                  flat_stake: float) -> pd.DataFrame:
    """
    Simulate flat-stake and Kelly bets where model prob > threshold.
    Odds come from ELO-implied probability (proxy for fair odds).
    A real bookie adds ~5% margin — we note this but don't adjust here.
    """
    df = df.copy()
    df["bet_home"] = df[prob_col] >= threshold
    df["implied_odds"] = 1.0 / df["home_prob_elo"].clip(0.01, 0.99)

    # Flat stake
    df["flat_stake"] = np.where(df["bet_home"], flat_stake, 0.0)
    df["flat_return"] = np.where(
        df["bet_home"] & (df["home_win"] == 1),
        df["flat_stake"] * df["implied_odds"],
        0.0,
    )
    df["flat_pnl"] = df["flat_return"] - df["flat_stake"]

    # Kelly stake
    df["kelly_stake"] = df.apply(
        lambda r: kelly_stake(r[prob_col], r["home_prob_elo"]) * flat_stake
        if r["bet_home"] else 0.0, axis=1
    )
    df["kelly_return"] = np.where(
        df["bet_home"] & (df["home_win"] == 1),
        df["kelly_stake"] * df["implied_odds"],
        0.0,
    )
    df["kelly_pnl"] = df["kelly_return"] - df["kelly_stake"]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     type=int,   default=2015)
    parser.add_argument("--end",       type=int,   default=2025)
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Minimum model probability to place a bet")
    parser.add_argument("--stake",     type=float, default=1.0,
                        help="Flat stake size per bet")
    args = parser.parse_args()

    print(f"Loading features.csv...")
    full = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    full = full.dropna(subset=FEATURE_COLS + [TARGET])
    full[FEATURE_COLS] = full[FEATURE_COLS].astype(float)

    all_preds = []
    year_summary = []

    for test_year in range(args.start, args.end + 1):
        train = full[full["Year"] < test_year]
        test  = full[full["Year"] == test_year]

        if len(train) < 500 or len(test) == 0:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train[TARGET].values
        X_test  = test[FEATURE_COLS].values
        y_test  = test[TARGET].values

        cal = train_xgb(X_train, y_train)
        if cal is None:
            continue

        probs = cal.predict_proba(X_test)[:, 1]

        test_out = test[["Date", "Year", "Home_Team", "Away_Team",
                         "Home_Score", "Away_Score", "result",
                         "home_win", "home_prob_elo"]].copy()
        test_out["prob_xgb"] = probs.round(4)

        test_out = simulate_bets(test_out, "prob_xgb", args.threshold, args.stake)
        all_preds.append(test_out)

        bets = test_out[test_out["bet_home"]]
        n_bets = len(bets)
        if n_bets > 0:
            win_rate    = bets["home_win"].mean()
            flat_roi    = bets["flat_pnl"].sum() / bets["flat_stake"].sum() * 100
            kelly_roi   = (bets["kelly_pnl"].sum() / bets["kelly_stake"].sum() * 100
                          if bets["kelly_stake"].sum() > 0 else 0)
        else:
            win_rate = flat_roi = kelly_roi = 0.0

        bs  = brier_score_loss(y_test, probs)
        ll  = log_loss(y_test, probs)
        acc = (probs.round() == y_test).mean()

        elo_acc = ((test["home_prob_elo"].values >= 0.5) == y_test).mean()

        year_summary.append({
            "Year":        test_year,
            "N_games":     len(test),
            "N_bets":      n_bets,
            "Accuracy":    round(acc, 4),
            "ELO_Acc":     round(elo_acc, 4),
            "Brier":       round(bs, 4),
            "LogLoss":     round(ll, 4),
            "Win_Rate":    round(win_rate, 4),
            "Flat_ROI%":   round(flat_roi, 2),
            "Kelly_ROI%":  round(kelly_roi, 2),
        })

        print(f"  {test_year}: {len(test):>4} games | "
              f"acc {acc*100:.1f}% (ELO {elo_acc*100:.1f}%) | "
              f"{n_bets} bets | "
              f"flat ROI {flat_roi:+.1f}% | kelly ROI {kelly_roi:+.1f}%")

    if not all_preds:
        print("No predictions generated.")
        return

    results_df  = pd.concat(all_preds, ignore_index=True)
    summary_df  = pd.DataFrame(year_summary)

    results_df.to_csv("backtest_results.csv", index=False)
    summary_df.to_csv("backtest_summary.csv", index=False)

    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Aggregate stats
    all_bets = results_df[results_df["bet_home"]]
    total_bets = len(all_bets)
    if total_bets > 0:
        total_flat_staked = all_bets["flat_stake"].sum()
        total_flat_pnl    = all_bets["flat_pnl"].sum()
        total_kelly_staked = all_bets["kelly_stake"].sum()
        total_kelly_pnl    = all_bets["kelly_pnl"].sum()

        print(f"\nAggregate ({args.start}-{args.end}, threshold={args.threshold}):")
        print(f"  Total bets:       {total_bets}")
        print(f"  Overall win rate: {all_bets['home_win'].mean()*100:.1f}%")
        print(f"  Flat ROI:         {total_flat_pnl/total_flat_staked*100:+.2f}%  "
              f"(P&L: {total_flat_pnl:+.2f} units)")
        print(f"  Kelly ROI:        {total_kelly_pnl/total_kelly_staked*100:+.2f}%  "
              f"(P&L: {total_kelly_pnl:+.2f} units)")
        print()
        print("NOTE: Odds are simulated from ELO (no bookie margin).")
        print("  Real bookmaker margins (~5%) will reduce ROI by ~5-8pp.")
        print("  To use real odds, add an Odds_Decimal column to backtest_results.csv")
        print("  and recompute returns using actual bookmaker prices.")
    else:
        print(f"\nNo bets placed at threshold={args.threshold}. Try lowering it.")

    print("\nSaved backtest_results.csv and backtest_summary.csv")


if __name__ == "__main__":
    main()
