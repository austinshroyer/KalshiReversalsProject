"""
Rigorously stress-tests ONE specific (time-in-game, entry-price) trading
rule you found via buy_strategy.py's grid search, instead of trusting a
single backtested average return at face value.

Why this exists: a grid search across many buckets will always surface a
few that look great purely by chance (multiple comparisons), and because
odds_timeline is sampled every minute, a single game can contribute dozens
of highly correlated rows to one bucket -- so the raw trade count in
buy_strategy.py's report overstates how much independent evidence you
actually have. This script:

  1. Reports the EFFECTIVE sample size (unique games), not just raw rows.
  2. Splits games chronologically into an earlier/later half and reports
     the average return in each -- a real edge should show up in both; one
     driven by a handful of games or a lucky stretch usually won't.
  3. Computes a game-level block bootstrap confidence interval, resampling
     whole games (not individual rows) so within-game autocorrelation
     doesn't understate the real uncertainty.
  4. Lets you subtract an estimated fee/slippage haircut to see if the edge
     survives realistic trading costs.

None of this makes a strategy "proven" -- the only real test is whether it
keeps working on NEW games as they settle, going forward. Treat this as a
filter for ruling out obviously-noisy buckets, not a green light.

Usage:
    python validate_strategy.py --db reversals.db \
        --time-min 0.6 --time-max 0.8 --price-min 0.10 --price-max 0.25
"""

import argparse

import numpy as np
import pandas as pd

import features as feat
from buy_strategy import build_candidate_trades


def select_trades(trades: pd.DataFrame, time_min, time_max, price_min, price_max) -> pd.DataFrame:
    mask = (
        (trades["elapsed_frac"] >= time_min)
        & (trades["elapsed_frac"] < time_max)
        & (trades["entry_price"] >= price_min)
        & (trades["entry_price"] < price_max)
    )
    return trades[mask].copy()


def apply_costs(trades: pd.DataFrame, slippage: float, fee_rate: float) -> pd.DataFrame:
    """slippage: added to your buy price to approximate a worse real fill.
    fee_rate: fraction of the (slippage-adjusted) entry price charged as a
    fee. Kalshi's real fee schedule varies by market -- set this from their
    current fee docs if you want a precise number; the default (0) just
    shows the slippage-only impact."""
    t = trades.copy()
    t["entry_price_adj"] = (t["entry_price"] + slippage).clip(upper=0.99)
    t["fee"] = t["entry_price_adj"] * fee_rate
    total_cost = t["entry_price_adj"] + t["fee"]
    t["return_pct_adj"] = (t["payoff"] - total_cost) / total_cost
    return t


def chronological_split(games: pd.DataFrame, trades: pd.DataFrame, split_frac: float = 0.5):
    g = games.dropna(subset=["open_time"]).sort_values("open_time")
    cutoff_idx = int(len(g) * split_frac)
    early_ids = set(g["game_id"].iloc[:cutoff_idx])
    late_ids = set(g["game_id"].iloc[cutoff_idx:])
    return trades[trades["game_id"].isin(early_ids)], trades[trades["game_id"].isin(late_ids)]


def summarize(t: pd.DataFrame, return_col: str = "return_pct") -> dict:
    if t.empty:
        return {"n_trades": 0, "n_games": 0}
    return {
        "n_trades": len(t),
        "n_games": t["game_id"].nunique(),
        "avg_return_pct": float(t[return_col].mean()),
        "median_return_pct": float(t[return_col].median()),
        "win_rate": float(t["won"].mean()),
    }


def block_bootstrap_ci(t: pd.DataFrame, return_col: str = "return_pct", n_boot: int = 2000, seed: int = 42):
    """Resamples whole GAMES with replacement (not individual rows), so
    within-game autocorrelation between adjacent 1-minute samples doesn't
    make the result look more certain than it really is."""
    rng = np.random.default_rng(seed)
    game_ids = t["game_id"].unique()
    if len(game_ids) < 5:
        return None
    by_game = {gid: sub[return_col].to_numpy() for gid, sub in t.groupby("game_id")}

    means = []
    for _ in range(n_boot):
        sample_ids = rng.choice(game_ids, size=len(game_ids), replace=True)
        vals = np.concatenate([by_game[g] for g in sample_ids])
        means.append(vals.mean())
    means = np.array(means)
    return {
        "boot_mean": float(means.mean()),
        "ci_low_2.5pct": float(np.percentile(means, 2.5)),
        "ci_high_97.5pct": float(np.percentile(means, 97.5)),
        "p_return_le_zero": float((means <= 0).mean()),
    }


def main():
    p = argparse.ArgumentParser(description="Stress-test one specific buy_strategy.py bucket.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument("--time-min", type=float, required=True)
    p.add_argument("--time-max", type=float, required=True)
    p.add_argument("--price-min", type=float, required=True)
    p.add_argument("--price-max", type=float, required=True)
    p.add_argument(
        "--slippage",
        type=float,
        default=0.02,
        help="Added to your fill price (0-1 scale) to approximate spread/slippage. Default 0.02 = 2 cents worse.",
    )
    p.add_argument(
        "--fee-rate",
        type=float,
        default=0.0,
        help="Fraction of entry price charged as a fee. Default 0 (slippage-only). Set from Kalshi's current fee schedule for a precise number.",
    )
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    games, _, timeline = feat.load_tables(args.db)
    if timeline.empty:
        raise SystemExit("odds_timeline is empty -- see buy_strategy.py's docstring for why this analysis needs it.")

    all_trades = build_candidate_trades(games, timeline)
    trades = select_trades(all_trades, args.time_min, args.time_max, args.price_min, args.price_max)
    if trades.empty:
        raise SystemExit("No trades match that (time, price) window -- widen it.")

    print(
        f"Selected bucket: elapsed_frac in [{args.time_min}, {args.time_max}), "
        f"entry_price in [{args.price_min}, {args.price_max})\n"
    )

    print("=== Full-sample summary (same numbers buy_strategy.py's grid would show) ===")
    for k, v in summarize(trades).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    n_games, n_trades = trades["game_id"].nunique(), len(trades)
    if n_games:
        print(
            f"\n  NOTE: {n_trades} rows come from only {n_games} distinct games "
            f"(~{n_trades / n_games:.1f} correlated rows/game) -- treat {n_games}, "
            f"not {n_trades}, as your real sample size."
        )

    print("\n=== Chronological split (does the edge hold up over time?) ===")
    early, late = chronological_split(games, trades)
    print("  Earlier half of games:")
    for k, v in summarize(early).items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print("  Later half of games:")
    for k, v in summarize(late).items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n=== Game-level block bootstrap (uncertainty on avg_return_pct) ===")
    ci = block_bootstrap_ci(trades, n_boot=args.n_boot)
    if ci is None:
        print("  Fewer than 5 distinct games in this bucket -- not enough to bootstrap meaningfully.")
    else:
        for k, v in ci.items():
            print(f"  {k}: {v:.4f}")
        if ci["ci_low_2.5pct"] <= 0:
            print(
                "  NOTE: the 95% interval includes zero (or is entirely negative) -- this "
                "bucket's edge is not statistically distinguishable from noise yet."
            )

    print("\n=== With estimated fees/slippage ===")
    adj = apply_costs(trades, slippage=args.slippage, fee_rate=args.fee_rate)
    for k, v in summarize(adj, return_col="return_pct_adj").items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
