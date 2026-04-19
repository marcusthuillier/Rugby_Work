"""
backtest.py
Walk-forward backtest comparing three models against the ELO baseline.

Models:
  - XGBoost (calibrated)        — non-linear, tree-based
  - Logistic Regression          — linear baseline, scaled features
  - Ensemble (XGB + LR blend)   — simple 50/50 probability average

Strategy:
  - Train on all games up to year T-1
  - Predict games in year T
  - Roll forward one year at a time (2015–2025)

Betting simulation:
  - Flat stake: bet 1 unit whenever model confidence > threshold
  - Kelly criterion: stake proportional to perceived edge
  - Odds proxy: ELO-implied fair odds (no bookie margin)
    Real bookmaker margins (~5%) will reduce ROI by ~5-8pp.

Outputs:
  backtest_results.csv   — per-game predictions (all three models)
  backtest_summary.csv   — year-by-year accuracy and P&L for all models

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


def train_lr(X, y):
    if len(np.unique(y)) < 2:
        return None
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def kelly_stake(prob: float, implied_prob: float, max_stake: float = 5.0) -> float:
    if implied_prob <= 0 or prob <= implied_prob:
        return 0.0
    odds = 1.0 / implied_prob
    edge = prob - implied_prob
    f = edge / (odds - 1)
    return min(max(f, 0), max_stake)


def simulate_bets(df: pd.DataFrame, prob_col: str, threshold: float,
                  flat_stake: float) -> pd.DataFrame:
    df = df.copy()
    df["bet_home"]     = df[prob_col] >= threshold
    df["implied_odds"] = 1.0 / df["home_prob_elo"].clip(0.01, 0.99)

    df["flat_stake"]  = np.where(df["bet_home"], flat_stake, 0.0)
    df["flat_return"] = np.where(
        df["bet_home"] & (df["home_win"] == 1),
        df["flat_stake"] * df["implied_odds"], 0.0)
    df["flat_pnl"] = df["flat_return"] - df["flat_stake"]

    df["kelly_stake"] = df.apply(
        lambda r: kelly_stake(r[prob_col], r["home_prob_elo"]) * flat_stake
        if r["bet_home"] else 0.0, axis=1)
    df["kelly_return"] = np.where(
        df["bet_home"] & (df["home_win"] == 1),
        df["kelly_stake"] * df["implied_odds"], 0.0)
    df["kelly_pnl"] = df["kelly_return"] - df["kelly_stake"]

    return df


def bet_stats(df: pd.DataFrame) -> tuple:
    bets = df[df["bet_home"]]
    n = len(bets)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    flat_roi  = bets["flat_pnl"].sum()  / bets["flat_stake"].sum()  * 100
    kelly_roi = (bets["kelly_pnl"].sum() / bets["kelly_stake"].sum() * 100
                 if bets["kelly_stake"].sum() > 0 else 0.0)
    win_rate  = bets["home_win"].mean()
    return flat_roi, kelly_roi, win_rate, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     type=int,   default=2015)
    parser.add_argument("--end",       type=int,   default=2025)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--stake",     type=float, default=1.0)
    args = parser.parse_args()

    print("Loading features.csv...")
    full = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    full = full.dropna(subset=FEATURE_COLS + [TARGET])
    full[FEATURE_COLS] = full[FEATURE_COLS].astype(float)

    all_preds    = []
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

        xgb_cal = train_xgb(X_train, y_train)
        lr_pipe  = train_lr(X_train, y_train)

        if xgb_cal is None or lr_pipe is None:
            continue

        prob_xgb      = xgb_cal.predict_proba(X_test)[:, 1]
        prob_lr       = lr_pipe.predict_proba(X_test)[:, 1]
        prob_ensemble = (prob_xgb + prob_lr) / 2.0

        acc_xgb      = (prob_xgb.round()      == y_test).mean()
        acc_lr        = (prob_lr.round()       == y_test).mean()
        acc_ensemble  = (prob_ensemble.round() == y_test).mean()
        acc_elo       = ((test["home_prob_elo"].values >= 0.5) == y_test).mean()

        bs = brier_score_loss(y_test, prob_xgb)
        ll = log_loss(y_test, prob_xgb)

        test_out = test[["Date", "Year", "Home_Team", "Away_Team",
                          "Home_Score", "Away_Score", "result",
                          "home_win", "home_prob_elo"]].copy()
        test_out["prob_xgb"]      = prob_xgb.round(4)
        test_out["prob_lr"]       = prob_lr.round(4)
        test_out["prob_ensemble"] = prob_ensemble.round(4)

        sim_xgb     = simulate_bets(test_out, "prob_xgb",      args.threshold, args.stake)
        sim_lr       = simulate_bets(test_out, "prob_lr",       args.threshold, args.stake)
        sim_ensemble = simulate_bets(test_out, "prob_ensemble", args.threshold, args.stake)

        flat_roi_xgb, kelly_roi_xgb, _, n_xgb = bet_stats(sim_xgb)
        flat_roi_lr,  kelly_roi_lr,  _, n_lr   = bet_stats(sim_lr)
        flat_roi_ens, kelly_roi_ens, _, n_ens  = bet_stats(sim_ensemble)

        year_summary.append({
            "Year":           test_year,
            "N_games":        len(test),
            "Accuracy":       round(acc_xgb,     4),
            "LR_Acc":         round(acc_lr,       4),
            "Ensemble_Acc":   round(acc_ensemble, 4),
            "ELO_Acc":        round(acc_elo,      4),
            "Brier":          round(bs, 4),
            "LogLoss":        round(ll, 4),
            "N_bets_XGB":     n_xgb,
            "N_bets_LR":      n_lr,
            "N_bets_Ens":     n_ens,
            "Flat_ROI%":      round(flat_roi_xgb, 2),
            "LR_Flat_ROI%":   round(flat_roi_lr,  2),
            "Ens_Flat_ROI%":  round(flat_roi_ens, 2),
            "Kelly_ROI%":     round(kelly_roi_xgb, 2),
            "LR_Kelly_ROI%":  round(kelly_roi_lr,  2),
            "Ens_Kelly_ROI%": round(kelly_roi_ens, 2),
        })

        # Results CSV uses XGB bet columns; attach LR + ensemble probs
        sim_xgb["prob_lr"]       = prob_lr.round(4)
        sim_xgb["prob_ensemble"] = prob_ensemble.round(4)
        all_preds.append(sim_xgb)

        print(f"  {test_year}: {len(test):>4} games | "
              f"XGB {acc_xgb*100:.1f}% | LR {acc_lr*100:.1f}% | "
              f"Ens {acc_ensemble*100:.1f}% | ELO {acc_elo*100:.1f}% | "
              f"XGB flat ROI {flat_roi_xgb:+.1f}%")

    if not all_preds:
        print("No predictions generated.")
        return

    results_df = pd.concat(all_preds, ignore_index=True)
    summary_df = pd.DataFrame(year_summary)

    results_df.to_csv("backtest_results.csv", index=False)
    summary_df.to_csv("backtest_summary.csv", index=False)

    print("\n" + "="*75)
    print("ACCURACY (walk-forward, 2015–2025)")
    print("="*75)
    print(summary_df[["Year", "N_games", "Accuracy", "LR_Acc",
                       "Ensemble_Acc", "ELO_Acc"]].to_string(index=False))

    print("\n" + "="*75)
    print("FLAT ROI — ELO-implied odds, threshold=" + str(args.threshold))
    print("="*75)
    print(summary_df[["Year", "N_bets_XGB", "Flat_ROI%",
                       "LR_Flat_ROI%", "Ens_Flat_ROI%"]].to_string(index=False))

    all_bets = results_df[results_df["bet_home"]]
    if len(all_bets) > 0:
        flat_staked  = all_bets["flat_stake"].sum()
        flat_pnl     = all_bets["flat_pnl"].sum()
        kelly_staked = all_bets["kelly_stake"].sum()
        kelly_pnl    = all_bets["kelly_pnl"].sum()

        print(f"\nAverage accuracy {args.start}–{args.end}:")
        print(f"  XGBoost:   {summary_df['Accuracy'].mean()*100:.2f}%")
        print(f"  LR:        {summary_df['LR_Acc'].mean()*100:.2f}%")
        print(f"  Ensemble:  {summary_df['Ensemble_Acc'].mean()*100:.2f}%")
        print(f"  ELO:       {summary_df['ELO_Acc'].mean()*100:.2f}%")

        print(f"\nAggregate XGB bets (threshold={args.threshold}):")
        print(f"  Total bets: {len(all_bets)}")
        print(f"  Flat ROI:   {flat_pnl/flat_staked*100:+.2f}%  "
              f"(P&L: {flat_pnl:+.2f} units)")
        print(f"  Kelly ROI:  {kelly_pnl/kelly_staked*100:+.2f}%  "
              f"(P&L: {kelly_pnl:+.2f} units)")
        print()
        print("NOTE: Odds are ELO-implied (no bookie margin).")
        print("  Real bookmaker margins (~5%) will reduce ROI by ~5-8pp.")

    print("\nSaved backtest_results.csv and backtest_summary.csv")


if __name__ == "__main__":
    main()
