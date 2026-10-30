"""
Two analyses of WHEN significant reversals happen:

  1. Descriptive timing: what fraction of the way through a game do
     significant reversals tend to cross? Only needs games+reversals.

  2. Discrete-time hazard model: bins each game's elapsed time into equal
     buckets (default 20, i.e. 5%-of-game-time each) and trains a logistic
     regression predicting P(a significant reversal crosses in this bin |
     game hasn't had one in this bin yet, current state). This is what lets
     you ask, mid-game, "given we're 40% through and the favorite is at
     70%, what's the chance of a significant reversal in the next 5%?"
     Requires odds_timeline (see main.py's --no-timeline flag).

Usage:
    pip install scikit-learn pandas numpy
    python timing_model.py --db reversals.db --bins 20
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import features as feat

NUMERIC_FEATURES = ["bin_start", "prob_a_at_start", "favorite_strength_at_start", "reversals_so_far"]
CATEGORICAL_FEATURES = ["series_ticker"]
TARGET = "label_reversal_in_bin"


def describe_timing(games: pd.DataFrame, reversals: pd.DataFrame) -> pd.DataFrame:
    """Elapsed-fraction of each significant reversal, using only
    games.open_time/close_time -- no odds_timeline needed."""
    g = games.set_index("game_id")[["open_time", "close_time"]]
    r = reversals.join(g, on="game_id")
    r = r.dropna(subset=["open_time", "close_time", "occurred_at"])

    open_ts = pd.to_datetime(r["open_time"], utc=True).astype("int64") // 10**9
    close_ts = pd.to_datetime(r["close_time"], utc=True).astype("int64") // 10**9
    occ_ts = pd.to_datetime(r["occurred_at"], utc=True).astype("int64") // 10**9

    duration = (close_ts - open_ts).astype(float)
    duration = duration.replace(0, np.nan)
    r = r.assign(elapsed_frac=((occ_ts - open_ts) / duration).clip(0, 1))
    return r.dropna(subset=["elapsed_frac"])


def summarize_timing(r: pd.DataFrame) -> dict:
    if r.empty:
        return {"n_reversals": 0}
    return {
        "n_reversals": len(r),
        "median_elapsed_frac": float(r["elapsed_frac"].median()),
        "p25_elapsed_frac": float(r["elapsed_frac"].quantile(0.25)),
        "p75_elapsed_frac": float(r["elapsed_frac"].quantile(0.75)),
        "share_in_first_half": float((r["elapsed_frac"] < 0.5).mean()),
        "share_in_final_quarter": float((r["elapsed_frac"] >= 0.75).mean()),
    }


def build_hazard_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])


def evaluate_hazard(bins_df: pd.DataFrame, n_splits: int = 5) -> dict:
    X = bins_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = bins_df[TARGET]
    groups = bins_df["game_id"]  # keep every bin of one game on the same side of the split

    n_splits = max(2, min(n_splits, bins_df["game_id"].nunique()))
    gkf = GroupKFold(n_splits=n_splits)

    aucs, briers = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            continue
        pipe = build_hazard_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test_idx], proba))
        briers.append(brier_score_loss(y.iloc[test_idx], proba))

    return {
        "n_bins": len(bins_df),
        "n_games": bins_df["game_id"].nunique(),
        "base_rate_reversal_per_bin": float(y.mean()),
        "cv_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "cv_brier_score": float(np.mean(briers)) if briers else float("nan"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze WHEN significant reversals occur.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--out", default="hazard_model.pkl")
    args = p.parse_args(argv)

    games, reversals, timeline = feat.load_tables(args.db)

    print("=== Descriptive timing (no odds_timeline needed) ===")
    r = describe_timing(games, reversals)
    for k, v in summarize_timing(r).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if timeline.empty:
        print(
            "\nNo odds_timeline data found -- skipping the hazard model. Re-run main.py "
            "without --no-timeline (add --reprocess to backfill games already stored)."
        )
        return

    print("\n=== Hazard model: P(significant reversal in next time-bin) ===")
    bins_df = feat.build_hazard_bins(games, reversals, timeline, n_bins=args.bins)
    metrics = evaluate_hazard(bins_df)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if not np.isnan(metrics["cv_auc"]) and metrics["cv_auc"] < 0.55:
        print(
            "  NOTE: AUC close to 0.5 means the model is barely distinguishing high- from "
            "low-risk moments -- treat its bin-by-bin probabilities as a rough prior, not "
            "a precise signal."
        )

    pipe = build_hazard_pipeline()
    X = bins_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = bins_df[TARGET]
    pipe.fit(X, y)
    with open(args.out, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nSaved trained hazard model to {args.out}")


if __name__ == "__main__":
    main()
