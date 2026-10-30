"""
Predicts how many significant reversals a game will have, using ONLY
pregame information (pregame_favorite_prob, series_ticker) -- so this is
genuinely usable BEFORE a game starts, not just a description of games that
already happened.

Model: Poisson regression (sklearn PoissonRegressor), the standard choice
for count data ("number of events in a fixed window"). A "predict the
historical average" baseline is reported alongside it -- if the trained
model isn't beating that baseline, that's a real, honest finding (it means
pregame favorite strength / sport alone don't carry much signal for this),
not a bug to explain away.

Usage:
    pip install scikit-learn pandas numpy
    python count_model.py --db reversals.db
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import features as feat

NUMERIC_FEATURES = ["pregame_favorite_prob", "favorite_strength"]
CATEGORICAL_FEATURES = ["series_ticker"]
TARGET = "significant_reversal_count"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", PoissonRegressor(alpha=1e-3, max_iter=500))])


def evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(float)

    n_splits = max(2, min(n_splits, len(df) // 20 or 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    maes, deviances, baseline_maes = [], [], []
    for train_idx, test_idx in kf.split(X):
        pipe = build_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.clip(pipe.predict(X.iloc[test_idx]), 1e-6, None)
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        deviances.append(mean_poisson_deviance(y.iloc[test_idx], pred))
        baseline_pred = np.full(len(test_idx), y.iloc[train_idx].mean())
        baseline_maes.append(mean_absolute_error(y.iloc[test_idx], baseline_pred))

    return {
        "n_games": len(df),
        "cv_folds": kf.get_n_splits(),
        "mae": float(np.mean(maes)),
        "mean_poisson_deviance": float(np.mean(deviances)),
        "baseline_mae_predict_mean": float(np.mean(baseline_maes)),
    }


def train_final_model(df: pd.DataFrame) -> Pipeline:
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(float)
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def predict_for_game(pipe: Pipeline, series_ticker: str, pregame_favorite_prob: float) -> float:
    row = pd.DataFrame(
        [
            {
                "series_ticker": series_ticker,
                "pregame_favorite_prob": pregame_favorite_prob,
                "favorite_strength": abs(pregame_favorite_prob - 0.5),
            }
        ]
    )
    return float(np.clip(pipe.predict(row)[0], 0, None))


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the significant-reversal-count model.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument("--out", default="count_model.pkl")
    args = p.parse_args(argv)

    games, _, _ = feat.load_tables(args.db)
    df = feat.build_pregame_features(games)
    if len(df) < 20:
        raise SystemExit(
            f"Only {len(df)} games have a usable pregame_favorite_prob -- need more data "
            "(roughly 50+) before this model's cross-validated numbers mean much."
        )

    metrics = evaluate(df)
    print("Cross-validated performance (predicting significant_reversal_count, PREGAME info only):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if metrics["mae"] >= metrics["baseline_mae_predict_mean"]:
        print(
            "  NOTE: this model is not beating 'always predict the historical average' -- "
            "pregame favorite strength / sport may not carry much signal for reversal count "
            "in your data. That's a legitimate finding, not a failure of the code."
        )

    pipe = train_final_model(df)
    with open(args.out, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nSaved trained model to {args.out}")
    print("Example: predict_for_game(pipe, 'KXNFLGAME', 0.65) -> expected significant reversal count")


if __name__ == "__main__":
    main()
