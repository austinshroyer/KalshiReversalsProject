"""
Stress-tests one specific (time, price) bucket + cashout policy found via
cashout_strategy.py -- the cashout counterpart to validate_strategy.py.
Same checks, same reasons for them (see validate_strategy.py's docstring):
effective sample size, chronological out-of-sample split, a game-level
block bootstrap CI, and a fee/slippage sensitivity check.

Supports all three independent triggers cashout_strategy.py and
paper_trader.py do: --take-profit, --stop-loss-frac, --max-hold-frac.

Usage:
    python cashout_validate.py --db reversals.db \
        --take-profit 0.55 --max-hold-frac 0.25 \
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25

    python cashout_validate.py --db reversals.db --stop-loss-frac 0.5 \
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25
"""

import argparse

import pandas as pd

import features as feat
from cashout_strategy import build_candidate_cashout_trades
from validate_strategy import block_bootstrap_ci, chronological_split, summarize


def select_trades(trades: pd.DataFrame, time_min, time_max, price_min, price_max) -> pd.DataFrame:
    mask = (
        (trades["elapsed_frac"] >= time_min)
        & (trades["elapsed_frac"] < time_max)
        & (trades["entry_price"] >= price_min)
        & (trades["entry_price"] < price_max)
    )
    return trades[mask].copy()


def apply_costs(trades: pd.DataFrame, slippage: float, fee_rate: float) -> pd.DataFrame:
    """Cashout-specific cost model: uses exit_price (which may be an early
    cashout price OR the $1/$0 settlement), unlike validate_strategy.py's
    version which only ever deals with a terminal 1/0 payoff."""
    t = trades.copy()
    t["entry_price_adj"] = (t["entry_price"] + slippage).clip(upper=0.99)
    t["fee"] = t["entry_price_adj"] * fee_rate
    total_cost = t["entry_price_adj"] + t["fee"]
    t["return_pct_adj"] = (t["exit_price"] - total_cost) / total_cost
    return t


def main():
    p = argparse.ArgumentParser(description="Stress-test one specific cashout_strategy.py bucket.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument("--take-profit", type=float, default=None, help="Omit to disable this trigger.")
    p.add_argument("--max-hold-frac", type=float, default=None, help="Omit to disable this trigger.")
    p.add_argument(
        "--stop-loss-frac",
        type=float,
        default=None,
        help="Cash out once price falls to (1 - this) of entry price, e.g. 0.5 = down 50%%. Omit to disable.",
    )
    p.add_argument("--time-min", type=float, required=True)
    p.add_argument("--time-max", type=float, required=True)
    p.add_argument("--price-min", type=float, required=True)
    p.add_argument("--price-max", type=float, required=True)
    p.add_argument("--slippage", type=float, default=0.02)
    p.add_argument("--fee-rate", type=float, default=0.0)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    if args.take_profit is None and args.max_hold_frac is None and args.stop_loss_frac is None:
        p.error("Set at least one of --take-profit / --max-hold-frac / --stop-loss-frac.")

    games, _, timeline = feat.load_tables(args.db)
    if timeline.empty:
        raise SystemExit("odds_timeline is empty -- see buy_strategy.py's docstring for why this needs it.")

    all_trades = build_candidate_cashout_trades(
        games, timeline, args.take_profit, args.max_hold_frac, args.stop_loss_frac
    )
    trades = select_trades(all_trades, args.time_min, args.time_max, args.price_min, args.price_max)
    if trades.empty:
        raise SystemExit("No trades match that (time, price) window -- widen it.")

    print(
        f"Selected bucket: elapsed_frac in [{args.time_min}, {args.time_max}), "
        f"entry_price in [{args.price_min}, {args.price_max}), "
        f"take-profit={args.take_profit}, max-hold-frac={args.max_hold_frac}, "
        f"stop-loss-frac={args.stop_loss_frac}\n"
    )

    print("=== Full-sample summary ===")
    for k, v in summarize(trades).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  cashout_rate: {trades['cashed_out'].mean():.4f}")
    print("  exit_reason breakdown:")
    print(trades["exit_reason"].value_counts(normalize=True).to_string())
    print(f"  avg_hold_to_settlement_return_pct (for comparison): {trades['hold_to_settlement_return_pct'].mean():.4f}")

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
                "  NOTE: the 95% interval includes zero -- this bucket's edge is not "
                "statistically distinguishable from noise yet."
            )

    print("\n=== With estimated fees/slippage ===")
    adj = apply_costs(trades, slippage=args.slippage, fee_rate=args.fee_rate)
    for k, v in summarize(adj, return_col="return_pct_adj").items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
