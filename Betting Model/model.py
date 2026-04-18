"""
model.py
Trains a rugby match outcome predictor using ELO features + form features.

Models:
  - Logistic Regression (baseline)
  - XGBoost (main model)

Evaluation:
  - Accuracy, Brier score, log loss
  - Calibration curve
  - Feature importance

Outputs:
  model_xgb.json          - trained XGBoost model
  model_lr.pkl            - trained Logistic Regression
  calibrator.pkl          - isotonic calibrator for XGBoost probabilities
  model_eval.csv          - per-game predictions on the test set
  feature_importance.csv  - XGBoost feature importances

Usage:
    python model.py
    python model.py --train-until 2020 --test-from 2021
"""

import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURES_PATH = "features.csv"

# All numeric features used for modelling
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

TARGET = "home_win"  # binary: 1=home win, 0=draw or away win


def load_data(train_until: int, test_from: int):
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)

    train = df[df["Year"] <= train_until].copy()
    test  = df[df["Year"] >= test_from].copy()
    return train, test, df


def train_logistic(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400,
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
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)

    # Isotonic calibration — makes probabilities more reliable for betting
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_train, y_train)
    return model, calibrator


def evaluate(model_name: str, probs: np.ndarray, y_true: pd.Series, threshold: float = 0.5):
    preds = (probs >= threshold).astype(int)
    acc   = accuracy_score(y_true, preds)
    bs    = brier_score_loss(y_true, probs)
    ll    = log_loss(y_true, probs)
    print(f"\n{model_name}")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Brier score: {bs:.4f}  (lower = better calibrated)")
    print(f"  Log loss:    {ll:.4f}  (lower = better)")
    print(classification_report(y_true, preds, target_names=["Not Home Win", "Home Win"],
                                digits=3))
    return acc, bs, ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-until", type=int, default=2019,
                        help="Train on data up to and including this year")
    parser.add_argument("--test-from", type=int, default=2020,
                        help="Test on data from this year onward")
    args = parser.parse_args()

    print(f"Loading features.csv (train <= {args.train_until}, test >= {args.test_from})...")
    train, test, full = load_data(args.train_until, args.test_from)

    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    print(f"  Home win rate — train: {train[TARGET].mean()*100:.1f}%  "
          f"test: {test[TARGET].mean()*100:.1f}%")

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    # ── ELO baseline ───────────────────────────────────────────────────────────
    print("\n--- ELO Baseline (raw Prob_Home as predictor) ---")
    elo_probs = test["home_prob_elo"].values
    evaluate("ELO Baseline", elo_probs, y_test)

    # ── Logistic Regression ────────────────────────────────────────────────────
    print("\n--- Logistic Regression ---")
    lr = train_logistic(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    evaluate("Logistic Regression", lr_probs, y_test)

    # ── XGBoost ────────────────────────────────────────────────────────────────
    print("\n--- XGBoost (calibrated) ---")
    xgb_model, calibrator = train_xgboost(X_train, y_train)
    xgb_probs = calibrator.predict_proba(X_test)[:, 1]
    evaluate("XGBoost (calibrated)", xgb_probs, y_test)

    # ── Feature importance ─────────────────────────────────────────────────────
    importances = xgb_model.feature_importances_
    fi = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)
    print("\nFeature Importances (XGBoost):")
    print(fi.to_string(index=False))
    fi.to_csv("feature_importance.csv", index=False)

    # ── Save test predictions ──────────────────────────────────────────────────
    out = test[["Date", "Year", "Home_Team", "Away_Team",
                "Home_Score", "Away_Score", "result",
                "home_win", "home_prob_elo"]].copy()
    out["prob_xgb"]  = xgb_probs.round(4)
    out["prob_lr"]   = lr_probs.round(4)
    out["pred_xgb"]  = (xgb_probs >= 0.5).astype(int)
    out["edge_vs_elo"] = (out["prob_xgb"] - out["home_prob_elo"]).round(4)
    out.to_csv("model_eval.csv", index=False)
    print(f"\nSaved model_eval.csv ({len(out):,} test games)")

    # ── Save models ────────────────────────────────────────────────────────────
    xgb_model.save_model("model_xgb.json")
    with open("model_lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open("calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    print("Saved model_xgb.json, model_lr.pkl, calibrator.pkl")

    # ── Where does the model disagree with ELO? ───────────────────────────────
    big_edges = out[out["edge_vs_elo"].abs() > 0.05].copy()
    print(f"\nGames where XGBoost disagrees with ELO by >5pp: {len(big_edges)}")
    print(f"  XGBoost more confident in home: {(big_edges['edge_vs_elo'] > 0).sum()}")
    print(f"  XGBoost less confident in home: {(big_edges['edge_vs_elo'] < 0).sum()}")


if __name__ == "__main__":
    main()
