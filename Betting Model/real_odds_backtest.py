"""
real_odds_backtest.py
Recomputes the backtest P&L using real bookmaker odds from backtest_with_odds.csv.

For games WITHOUT real odds, falls back to ELO-implied odds so the analysis
covers the full backtest window.

Betting logic:
  - Bet home when:  model prob (prob_xgb) > 1/Home_Odds  AND  prob_xgb >= threshold
  - i.e. we only bet when our model thinks the home team is undervalued by the bookie

  - Bet away when:  (1 - prob_xgb) > 1/Away_Odds  AND  (1 - prob_xgb) >= threshold
  - This adds away bets which the original backtest didn't include

Staking:
  - Flat stake: bet 1 unit per qualifying game
  - Kelly criterion: stake = (edge / (odds - 1)) * max_stake

Output: real_odds_summary.csv, real_odds_results.csv (per-game)

Usage:
  python real_odds_backtest.py
  python real_odds_backtest.py --threshold 0.58 --stake 2 --min-edge 0.03
"""

import argparse
import pandas as pd
import numpy as np

RESULTS_PATH = "backtest_with_odds.csv"
OUT_SUMMARY  = "real_odds_summary.csv"
OUT_RESULTS  = "real_odds_results.csv"


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Kelly criterion fraction. Returns 0 if no edge."""
    implied = 1.0 / decimal_odds
    edge = prob - implied
    if edge <= 0 or decimal_odds <= 1:
        return 0.0
    return edge / (decimal_odds - 1.0)


def run_backtest(df: pd.DataFrame, threshold: float, flat_stake: float,
                 kelly_cap: float, min_edge: float) -> pd.DataFrame:
    """Compute per-game bet decisions and P&L."""
    df = df.copy()

    # Fill missing real odds with ELO-implied odds
    df["_home_odds_used"] = np.where(
        df["Home_Odds"].notna(),
        df["Home_Odds"],
        1.0 / df["home_prob_elo"].clip(0.01, 0.99),
    )
    df["_away_odds_used"] = np.where(
        df["Away_Odds"].notna(),
        df["Away_Odds"],
        1.0 / (1 - df["home_prob_elo"]).clip(0.01, 0.99),
    )
    df["_odds_source"] = np.where(df["Home_Odds"].notna(), "real", "elo_implied")

    prob_home = df["prob_xgb"]
    prob_away = 1.0 - df["prob_xgb"]

    home_implied = 1.0 / df["_home_odds_used"]
    away_implied = 1.0 / df["_away_odds_used"]

    home_edge = prob_home - home_implied
    away_edge = prob_away - away_implied

    # Bet criteria
    df["bet_home"] = (prob_home >= threshold) & (home_edge >= min_edge)
    df["bet_away"] = (prob_away >= threshold) & (away_edge >= min_edge)

    # Flat stake
    df["flat_home_stake"]  = np.where(df["bet_home"], flat_stake, 0.0)
    df["flat_away_stake"]  = np.where(df["bet_away"], flat_stake, 0.0)
    df["flat_home_return"] = np.where(
        df["bet_home"] & (df["home_win"] == 1),
        flat_stake * df["_home_odds_used"], 0.0)
    df["flat_away_return"] = np.where(
        df["bet_away"] & (df["home_win"] == 0) & (df["result"] != "D"),
        flat_stake * df["_away_odds_used"], 0.0)
    df["flat_home_pnl"] = df["flat_home_return"] - df["flat_home_stake"]
    df["flat_away_pnl"] = df["flat_away_return"] - df["flat_away_stake"]
    df["flat_pnl_total"] = df["flat_home_pnl"] + df["flat_away_pnl"]

    # Kelly stake (capped)
    df["kelly_home_f"] = df.apply(
        lambda r: kelly_fraction(r["prob_xgb"], r["_home_odds_used"])
        if r["bet_home"] else 0.0, axis=1
    ).clip(0, kelly_cap)
    df["kelly_away_f"] = df.apply(
        lambda r: kelly_fraction(1 - r["prob_xgb"], r["_away_odds_used"])
        if r["bet_away"] else 0.0, axis=1
    ).clip(0, kelly_cap)

    df["kelly_home_stake"]  = df["kelly_home_f"] * flat_stake * 10
    df["kelly_away_stake"]  = df["kelly_away_f"] * flat_stake * 10
    df["kelly_home_return"] = np.where(
        df["bet_home"] & (df["home_win"] == 1),
        df["kelly_home_stake"] * df["_home_odds_used"], 0.0)
    df["kelly_away_return"] = np.where(
        df["bet_away"] & (df["home_win"] == 0) & (df["result"] != "D"),
        df["kelly_away_stake"] * df["_away_odds_used"], 0.0)
    df["kelly_home_pnl"] = df["kelly_home_return"] - df["kelly_home_stake"]
    df["kelly_away_pnl"] = df["kelly_away_return"] - df["kelly_away_stake"]
    df["kelly_pnl_total"] = df["kelly_home_pnl"] + df["kelly_away_pnl"]

    return df


def year_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(df["Year"].unique()):
        y = df[df["Year"] == year]
        bets = y[y["bet_home"] | y["bet_away"]]
        home_bets = y[y["bet_home"]]
        away_bets = y[y["bet_away"]]
        real_bets = bets[bets["_odds_source"] == "real"]

        flat_staked = bets["flat_home_stake"].sum() + bets["flat_away_stake"].sum()
        flat_pnl    = bets["flat_pnl_total"].sum()
        kelly_staked = bets["kelly_home_stake"].sum() + bets["kelly_away_stake"].sum()
        kelly_pnl   = bets["kelly_pnl_total"].sum()

        rows.append({
            "Year":          year,
            "N_games":       len(y),
            "N_bets":        len(bets),
            "N_home_bets":   len(home_bets),
            "N_away_bets":   len(away_bets),
            "N_real_odds":   len(real_bets),
            "Flat_ROI%":     round(flat_pnl / flat_staked * 100, 2) if flat_staked > 0 else 0,
            "Flat_PnL":      round(flat_pnl, 2),
            "Kelly_ROI%":    round(kelly_pnl / kelly_staked * 100, 2) if kelly_staked > 0 else 0,
            "Kelly_PnL":     round(kelly_pnl, 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.58,
                        help="Min model probability to consider a bet")
    parser.add_argument("--min-edge",  type=float, default=0.03,
                        help="Min edge over implied odds required to bet (e.g. 0.03 = 3pp)")
    parser.add_argument("--stake",     type=float, default=1.0,
                        help="Base stake unit")
    parser.add_argument("--kelly-cap", type=float, default=0.15,
                        help="Max Kelly fraction (fraction of bankroll)")
    args = parser.parse_args()

    if not pd.io.common.file_exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found.")
        print("  Run backtest.py first, then merge_odds.py to add real odds.")
        return

    print(f"Loading {RESULTS_PATH}...")
    df = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
    total = len(df)
    with_real = df["Home_Odds"].notna().sum()
    print(f"  {total} games total | {with_real} with real odds ({with_real/total*100:.1f}%) | "
          f"{total - with_real} using ELO fallback")

    print(f"\nRunning backtest (threshold={args.threshold}, min_edge={args.min_edge})...")
    results = run_backtest(df, args.threshold, args.stake, args.kelly_cap, args.min_edge)

    summary = year_summary(results)

    print("\n" + "="*75)
    print("YEAR-BY-YEAR RESULTS")
    print("="*75)
    print(summary.to_string(index=False))

    all_bets = results[results["bet_home"] | results["bet_away"]]
    total_bets = len(all_bets)

    if total_bets > 0:
        flat_staked  = (all_bets["flat_home_stake"] + all_bets["flat_away_stake"]).sum()
        flat_pnl     = all_bets["flat_pnl_total"].sum()
        kelly_staked = (all_bets["kelly_home_stake"] + all_bets["kelly_away_stake"]).sum()
        kelly_pnl    = all_bets["kelly_pnl_total"].sum()

        real_bets = all_bets[all_bets["_odds_source"] == "real"]
        elo_bets  = all_bets[all_bets["_odds_source"] == "elo_implied"]

        print(f"\n{'='*75}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*75}")
        print(f"  Total bets:          {total_bets}")
        print(f"  With real odds:      {len(real_bets)} ({len(real_bets)/total_bets*100:.1f}%)")
        print(f"  With ELO fallback:   {len(elo_bets)}")
        print()
        print(f"  Flat stake ROI:      {flat_pnl/flat_staked*100:+.2f}%  "
              f"(P&L: {flat_pnl:+.2f} units on {flat_staked:.0f} staked)")
        print(f"  Kelly ROI:           {kelly_pnl/kelly_staked*100:+.2f}%  "
              f"(P&L: {kelly_pnl:+.2f} units on {kelly_staked:.0f} staked)")
        print()

        if len(real_bets) > 0:
            r_flat_staked  = (real_bets["flat_home_stake"] + real_bets["flat_away_stake"]).sum()
            r_flat_pnl     = real_bets["flat_pnl_total"].sum()
            print(f"  REAL ODDS ONLY:")
            print(f"  Flat stake ROI:      {r_flat_pnl/r_flat_staked*100:+.2f}%  "
                  f"({len(real_bets)} bets)")
            print()

        print("  Note: Positive ROI on ELO-implied odds doesn't guarantee profit")
        print("  against real bookmakers (who add ~5-8% margin).")
        print("  Real odds ROI is the only reliable signal.")
    else:
        print(f"\nNo bets placed. Try lowering --threshold (currently {args.threshold}).")

    summary.to_csv(OUT_SUMMARY, index=False)
    results.to_csv(OUT_RESULTS, index=False)
    print(f"\nSaved {OUT_SUMMARY} and {OUT_RESULTS}")


if __name__ == "__main__":
    main()
