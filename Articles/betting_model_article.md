# I Built a Rugby Prediction Model With 71% Accuracy. Then I Tried to Bet With It.

> Winning 7 out of 10 games isn't enough.

---

A few months ago I set out to answer a straightforward question: can a machine learning model predict rugby match outcomes better than the market? I had 25,000+ historical games, ELO ratings going back to 1893, and a walk-forward backtesting framework. The accuracy numbers looked good. The betting numbers did not. Here's what the data showed and why the gap matters.

*This builds on Part 1 — the ELO rating system — which is worth reading first.*

---

## The dataset

The foundation is an ELO rating system applied to **25,448 rugby matches from 1893 to 2026** — internationals plus club competitions (Super Rugby, URC, Premiership, Top 14, Champions Cup). ELO ratings track each team's strength continuously, adjusting after every game based on result, margin, and competition importance.

For the prediction model I used results from 2015 onwards — enough history for the ratings to be well-calibrated, recent enough that competition structures are comparable to today.

---

## The methodology: walk-forward backtesting

The most important decision in any sports prediction backtest is not letting the model see the future.

A naive backtest trains on the full dataset and tests on the same data — which produces spectacular accuracy figures that mean nothing. The correct approach is **walk-forward**: for each year T from 2015 to 2025, train on every game before year T, predict games in year T only. Roll forward one year and repeat.

The model predicting 2023 has never seen a single 2023 game. Same constraint a real bettor faces. Test games were filtered to top-tier competitions only (Six Nations, Rugby Championship, Super Rugby, URC, Premiership, Top 14, Champions Cup) — including lower-tier qualifiers inflates game counts without adding useful signal.

---

## Feature engineering: beyond ELO

ELO alone gets you most of the way there, but it compresses everything about a team into one number. Nineteen features were engineered:

| Feature | Importance |
|---------|-----------|
| ELO difference | 34% |
| ELO win probability | 17% |
| Home venue win rate (last 5 home games) | 5.5% |
| Head-to-head win rate | 4.5% |
| Home team rest days | 3.6% |
| Home team experience | 3.4% |
| Away team margin form | 3.3% |

Three features added in the latest iteration — rest days, margin-based form (average point differential rather than binary win/loss), and venue-specific form (home record and away record tracked separately) — improved Logistic Regression accuracy by nearly 1 percentage point, evidence they carry genuine linear signal the base model was missing.

---

## The accuracy result

Over 11 years and 8,995 top-tier games:

| Model | Avg Accuracy |
|-------|-------------|
| XGBoost | **71.1%** |
| Ensemble (XGB + LR) | 70.9% |
| Logistic Regression | 70.5% |
| ELO baseline | 68.9% |

The model consistently outperforms raw ELO by ~2 percentage points every year. That's a real, stable improvement — not noise. XGBoost captures form streaks, rest advantages, and H2H patterns that ELO's single-number representation misses.

71% means correctly calling 7 out of every 10 games. So why does it fail as a betting tool?

---

## The real odds test

I obtained real bookmaker odds for 1,170 Super Rugby games from aussportsbetting.com — Pinnacle and bet365 closing lines from 2015 to 2025.

Simplest strategy: bet $100 on whichever team the model gives more than 50% probability to.

**Result: 70.7% win rate. -3.56% ROI. A loss of $4,165 on $117,000 staked.**

Winning nearly 3 in 4 bets and still losing money. The bookmaker's margin is why. Every bet has negative expected value baked in before the game starts — Pinnacle's overround means the implied probabilities on both sides sum to more than 100%. You need to be right *more than the odds imply*, not just right more often.

---

## The calibration finding

This is the most interesting result. Comparing model probability, bookmaker implied probability, and actual win rate across all 1,170 games:

| Model says | Bookie implies | Actual win rate |
|-----------|----------------|-----------------|
| 50–60% | 54% | 52% — bookie closer |
| 60–70% | 64% | 63% — bookie closer |
| 70–80% | 73% | 71% — bookie closer |
| 80–90% | 81% | 75% — bookie much closer |
| 90–100% | 90% | 88% — bookie closer |

In every bucket, Pinnacle's closing line is closer to the true outcome rate than the model. In the 80–90% range the gap is most stark: the model says 86%, the truth is 75%, the bookie already had it at 81%.

This isn't because Pinnacle has a better machine learning model. It's because their closing line is an **efficient price** — the result of professional bettors moving the line toward the true probability every time they spot a mispricing. By kick-off, all publicly available information is reflected. Our features (ELO, form, H2H, rest days) are all public. The market has already priced them in.

We also ran a proper train/test split: tune thresholds on 2015–2019, lock them, test on 2020–2025. The validation ROI of +15% on the chosen parameters became -12% on the test set — the classic overfitting signature.

---

## What would actually give edge

The model is genuinely useful as a predictive tool. 71% accuracy is real. But accuracy against outcomes and edge against the market are two different things. Real edge requires:

- **Information the market doesn't have** — injury news before it's public, team selection intel
- **Line shopping** — the same game at Pinnacle vs a soft book can differ by 5–10%
- **Niche markets** — bookmakers pay less attention to URC or Top 14 than internationals
- **Speed** — reacting faster than the market to new information

None of these come from a better historical model.

---

## The honest conclusion

The model works. The betting strategy doesn't — and that's the expected result. Demonstrating the efficient market hypothesis empirically in a sports context is more interesting than a backtest that games its parameters to show fake profit.

Full code and data at [github.com/marcusthuillier/Rugby_Work](https://github.com/marcusthuillier/Rugby_Work). Live ELO chart and model accuracy visuals at [marcusthuillier.com/lab](https://marcusthuillier.com/lab).
