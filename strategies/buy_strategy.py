"""
Backtests a simple rule -- "buy the underdog when their implied probability
is below X and the game is Y%-of-the-way through, hold the contract to
final settlement" -- across a grid of (X, Y) combinations, to see which
combinations paid off best historically.

WHY THIS NEEDS odds_timeline (not just the `reversals` table):
`reversals` only contains events that already cleared the significance
threshold -- i.e., only the winners. If this backtest used only those rows,
every "trade" in the sample would already be a success, which would make
any strategy look artificially good (classic survivorship bias). Instead,
this script treats EVERY point in the full odds_timeline as a candidate
trade -- including every time an underdog dipped low and never came back --
so the win rate and average return you see here are the real historical
base rates, not just the base rate among reversals that happened to occur.

CAVEATS (read before trusting any of this):
  - Historical backtest, not a live signal. Past patterns may not repeat.
  - Ignores Kalshi's trading fees, bid/ask spread, and slippage -- your real
    fill price would be worse than the raw implied probability used here.
  - Assumes you could always get filled at the exact recorded price, which
    is optimistic versus a real order book, especially in thin markets.
  - Small buckets (low n) are noisy -- the report filters out combinations
    below --min-trades, but treat anything under a few hundred skeptically.
This is a historical data analysis tool, not financial advice.

Usage:
    pip install pandas numpy
    python buy_strategy.py --db reversals.db --time-bins 5 --price-bins 5
"""

import argparse

import numpy as np
import pandas as pd

import features as feat


def build_candidate_trades(games: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, timeline point) = one candidate 'buy the current
    underdog' trade, with its eventual payoff at game settlement."""
    g = games.set_index("game_id")[["final_winner"]]
    df = timeline.join(g, on="game_id")
    df = df.dropna(subset=["final_winner"])

    df = df.copy()
    df["underdog_side"] = np.where(df["prob_a"] < 0.5, "A", "B")
    df["entry_price"] = np.where(df["prob_a"] < 0.5, df["prob_a"], 1 - df["prob_a"])
    df = df[df["entry_price"] > 0]  # skip degenerate zero-price quotes

    df["won"] = (df["underdog_side"] == df["final_winner"]).astype(int)
    df["payoff"] = df["won"].astype(float)  # Kalshi contracts settle at $1 or $0
    df["return_pct"] = (df["payoff"] - df["entry_price"]) / df["entry_price"]
    return df


def grid_search(df: pd.DataFrame, time_bins: int = 5, price_bins: int = 5, min_trades: int = 100) -> pd.DataFrame:
    df = df.copy()
    df["time_bucket"] = pd.cut(df["elapsed_frac"], bins=time_bins, labels=False, include_lowest=True)
    df["price_bucket"] = pd.cut(df["entry_price"], bins=price_bins, labels=False, include_lowest=True)
    df = df.dropna(subset=["time_bucket", "price_bucket"])

    time_cats = pd.cut(df["elapsed_frac"], bins=time_bins, include_lowest=True).cat.categories
    price_cats = pd.cut(df["entry_price"], bins=price_bins, include_lowest=True).cat.categories

    grouped = (
        df.groupby(["time_bucket", "price_bucket"])
        .agg(
            n_trades=("return_pct", "size"),
            avg_return_pct=("return_pct", "mean"),
            win_rate=("won", "mean"),
            avg_entry_price=("entry_price", "mean"),
        )
        .reset_index()
    )
    grouped["time_window"] = grouped["time_bucket"].apply(lambda i: str(time_cats[int(i)]))
    grouped["price_window"] = grouped["price_bucket"].apply(lambda i: str(price_cats[int(i)]))

    grouped = grouped[grouped["n_trades"] >= min_trades]
    return grouped.sort_values("avg_return_pct", ascending=False)


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest underdog-buying rules against historical odds paths.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument("--time-bins", type=int, default=5)
    p.add_argument("--price-bins", type=int, default=5)
    p.add_argument("--min-trades", type=int, default=100)
    p.add_argument("--out", default="buy_strategy_report.csv")
    args = p.parse_args(argv)

    games, _, timeline = feat.load_tables(args.db)
    if timeline.empty:
        raise SystemExit(
            "odds_timeline is empty -- this analysis needs the full odds path, not just the "
            "significant-reversal summary (see this file's module docstring for why). Re-run "
            "main.py without --no-timeline (add --reprocess to backfill games already stored)."
        )

    trades = build_candidate_trades(games, timeline)
    report = grid_search(trades, args.time_bins, args.price_bins, args.min_trades)

    print(f"Backtested {len(trades):,} candidate underdog entries across {trades['game_id'].nunique():,} games.\n")
    if report.empty:
        print(
            f"No (time, price) bucket reached --min-trades={args.min_trades} -- try more data, "
            "fewer bins, or a lower --min-trades."
        )
        return

    print("Top (time-in-game, entry-price) buckets by average return:")
    print(
        report[["time_window", "price_window", "n_trades", "avg_entry_price", "win_rate", "avg_return_pct"]]
        .head(15)
        .to_string(index=False)
    )

    report.to_csv(args.out, index=False)
    print(f"\nFull grid saved to {args.out} -- pivot/chart it in Excel the same way as the reversals export.")
    print(
        "\nReminder: historical backtest, no fees/slippage modeled, not a guarantee of future "
        "results. See this file's module docstring for the full caveat list."
    )


if __name__ == "__main__":
    main()
