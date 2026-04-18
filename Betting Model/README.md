# Super Rugby Match Prediction & Spread Model

Match outcome and point spread prediction for Super Rugby (2015–2026). Includes backtesting against bookmaker odds.

## Files

### Data Pipeline
| File | Description |
|------|-------------|
| `fetch_match_meta.py` | Retrieves match metadata |
| `fetch_odds.py` | Collects bookmaker odds from aussportsbetting.com |
| `merge_odds.py` | Consolidates odds data |
| `build_comparison_dataset.py` | Builds the model comparison dataset |
| `features.py` | Feature engineering — ELO differentials, form, momentum, H2H |

### Models
| File | Description |
|------|-------------|
| `model.py` | Logistic Regression and XGBoost outcome models |
| `spread_model.py` | Point spread prediction model |
| `predict.py` | Prediction inference |
| `model_xgb.json` | Trained XGBoost weights |
| `spread_model.json` | Trained spread model weights |
| `model_lr.pkl` | Trained Logistic Regression |
| `calibrator.pkl` | Isotonic probability calibrator |

### Evaluation
| File | Description |
|------|-------------|
| `backtest.py` | Backtesting framework |
| `real_odds_backtest.py` | Backtest against actual bookmaker odds |
| `comparison_dashboard.py` | Streamlit dashboard — 6 analytical tabs |

### Datasets
| File | Description |
|------|-------------|
| `features.csv` | 13 engineered features per match |
| `match_meta.csv` | Match metadata |
| `super_rugby_raw.xlsx` | Raw Super Rugby results |
| `comparison_dataset.csv` | Model comparison data |
| `backtest_results.csv` | Detailed backtest outcomes |
| `backtest_summary.csv` | Backtest summary metrics |
| `model_eval.csv` | Per-game model predictions |
| `spread_predictions.csv` | Spread model predictions |
| `spread_summary.csv` | Spread model summary |
| `feature_importance.csv` | XGBoost feature importances |

## Features
- ELO differential and implied win probability
- Form: 5-game and 10-game win rates
- Momentum: ELO change over last 5 games
- Head-to-head win rate (last 10 meetings)
- Competition weight (World Rugby rankings weight)
- Games played (home and away)

## Setup

```bash
pip install -r requirements.txt
```

## Execution Order

The pipeline has two tracks that converge before the dashboard. Run the **Core Track** first. The **Odds Track** is optional — it enriches the backtest with real bookmaker odds.

### Core Track (required)

**Step 1 — Fetch match metadata** (competition weights, venue country):
```bash
python fetch_match_meta.py
```
Outputs: `match_meta.csv`

**Step 2 — Build feature matrix** (ELO differentials, form, momentum, H2H):
```bash
python features.py
```
Requires: `../ELOR/Datasets/ELO.csv`, `match_meta.csv`
Outputs: `features.csv`

**Step 3 — Train outcome and spread models:**
```bash
python model.py
python spread_model.py
```
Outputs: `model_xgb.json`, `model_lr.pkl`, `calibrator.pkl`, `spread_model.json`, `model_eval.csv`, `feature_importance.csv`, `spread_predictions.csv`

**Step 4 — Run backtest** (walk-forward, year by year):
```bash
python backtest.py
```
Outputs: `backtest_results.csv`, `backtest_summary.csv`

### Odds Track (optional — enriches backtest with real bookmaker odds)

**Step 5a — Fetch real bookmaker odds:**
```bash
python fetch_odds.py
```
Outputs: `odds_raw.csv`

**Step 5b — Merge odds onto backtest results:**
```bash
python merge_odds.py
```
Outputs: `backtest_with_odds.csv`

**Step 5c — Re-run backtest against real odds:**
```bash
python real_odds_backtest.py
```

### Final Step — Build comparison dataset and launch dashboard

**Step 6 — Build master comparison dataset** (combines ELO, model, and real odds):
```bash
python build_comparison_dataset.py
```
Requires: `comparison_dataset.csv`, trained models (`model_xgb.json`, `calibrator.pkl`, `spread_model.json`)
Outputs: `comparison_dataset.csv`

**Step 7 — Launch the dashboard:**
```bash
streamlit run comparison_dashboard.py
```

## Dashboard Tabs

1. **Calibration** — Brier scores measuring prediction accuracy vs outcomes
2. **Win Probability** — Scatter plots: ELO vs XGBoost vs market odds
3. **Spread Comparison** — Predicted vs actual game margins
4. **Value Bets** — High-confidence disagreements between model and bookmakers
5. **ROI Curves** — Simulated returns at varying probability thresholds
6. **Game Explorer** — Drill-down for individual matchups
