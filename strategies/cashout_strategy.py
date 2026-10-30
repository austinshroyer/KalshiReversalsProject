"""
A cashout-aware variant of buy_strategy.py's backtest.

buy_strategy.py assumes you always hold your underdog contract all the way
to final settlement ($1 if it wins, $0 if it loses). This script adds the
option to CASH OUT early -- selling your position back to the market at
whatever price is showing at that moment -- instead of waiting for the
uncertain final outcome, in exchange for a smaller but locked-in return.

The cashout rule modeled here has three independent triggers, any
combination of which you can set -- whichever fires FIRST (soonest in
game time) wins:
  --take-profit      Sell as soon as your contract's price rises to this
                       absolute level (e.g. 0.55 = "cash out once my
                       underdog becomes at least a slight favorite").
  --stop-loss-frac    Sell once your contract's price falls to (1 - this)
                       of what you paid -- e.g. 0.5 = cash out once you're
                       down 50% on the trade.
  --max-hold-frac     Sell after holding for this fraction of the game's
                       total duration, at whatever price is showing then.
If none of your enabled triggers fire before the game actually ends, the
position is held to real settlement ($1/$0) -- identical to
buy_strategy.py for that entry. Mirrors paper_trader.py's live trigger
logic exactly, so a bucket you validate here means the same thing live.

This reads the SAME reversals.db / odds_timeline as buy_strategy.py --
nothing new to collect, just a different way of trading the same odds
paths. Output is a separate, differently-named model so you can run and
compare both independently:
    python buy_strategy.py --db reversals.db          # hold-to-settlement
    python cashout_strategy.py --db reversals.db       # with early cashout

Usage:
    pip install pandas numpy
    python cashout_strategy.py --db reversals.db --take-profit 0.55 --max-hold-frac 0.25
    python cashout_strategy.py --db reversals.db --stop-loss-frac 0.5
"""

import argparse

import numpy as np
import pandas as pd

import features as feat


def _first_ge_index(arr: np.ndarray, target: float) -> np.ndarray:
    """For each position k, the earliest position >= k where arr >= target
    (len(arr) as a sentinel meaning 'never'). Fully vectorized, O(N) per
    game -- see features.py's build_hazard_bins fix for why a per-position
    Python loop here would be a real problem at real data sizes. Usable
    here because take-profit's threshold is the SAME fixed value for every
    starting position in the game."""
    n = len(arr)
    idx = np.where(arr >= target, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _first_relative_drop_index(arr: np.ndarray, frac: float) -> np.ndarray:
    """For each position k, the earliest position j >= k where
    arr[j] <= arr[k] * (1 - frac) -- i.e. a stop-loss-style relative drop
    FROM k's OWN value, not a fixed threshold. Because the threshold
    depends on the starting position itself, this can't reuse the O(N)
    backward-scan trick _first_ge_index uses; instead it's a vectorized
    O(N^2) broadcast comparison. N here is one game's timeline length
    (~150-200 at 1-minute resolution), so N^2 is a few tens of thousands
    of simple boolean ops per game -- verified fast in practice at real
    database sizes (see this file's test suite)."""
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int)
    thresholds = arr * (1 - frac)
    below = arr[np.newaxis, :] <= thresholds[:, np.newaxis]
    valid_j = np.arange(n)[np.newaxis, :] >= np.arange(n)[:, np.newaxis]
    below = below & valid_j
    has_any = below.any(axis=1)
    return np.where(has_any, np.argmax(below, axis=1), n)


def simulate_game_cashout(
    gtime: pd.DataFrame,
    final_winner: str,
    take_profit_price: float = None,
    max_hold_frac: float = None,
    stop_loss_frac: float = None,
):
    """Simulates the cashout rule for every timeline point in one game as a
    candidate entry (buying whichever side is the underdog at that point).
    Any of take_profit_price / max_hold_frac / stop_loss_frac may be None
    to disable that trigger; with all three None, every entry holds to
    settlement. Returns a DataFrame (one row per entry) or None if
    unusable."""
    prob_a = gtime["prob_a"].to_numpy()
    elapsed = gtime["elapsed_frac"].to_numpy()
    n = len(prob_a)
    if n == 0:
        return None

    side_price = {"A": prob_a, "B": 1.0 - prob_a}

    next_tp_hit = None
    if take_profit_price is not None:
        next_tp_hit = {side: _first_ge_index(arr, take_profit_price) for side, arr in side_price.items()}

    next_sl_hit = None
    if stop_loss_frac is not None:
        next_sl_hit = {side: _first_relative_drop_index(arr, stop_loss_frac) for side, arr in side_price.items()}

    sides_all = np.where(prob_a < 0.5, "A", "B")
    entry_prices_all = np.where(prob_a < 0.5, prob_a, 1.0 - prob_a)
    positions = np.nonzero(entry_prices_all > 0)[0]
    if len(positions) == 0:
        return None

    sides = sides_all[positions]
    entry_prices = entry_prices_all[positions]

    if max_hold_frac is not None:
        cap_elapsed = elapsed[positions] + max_hold_frac
        max_hold_idx = np.searchsorted(elapsed, cap_elapsed, side="right") - 1
        max_hold_idx = np.clip(max_hold_idx, positions, n - 1)
        # A cap that extends to/past the game's last observed point isn't a
        # real early trigger -- it just means "never got there before the
        # game ended," so it doesn't count as a candidate exit.
        mh_effective = np.where(max_hold_idx < n - 1, max_hold_idx, n)
    else:
        mh_effective = np.full(len(positions), n, dtype=int)

    tp_effective = (
        np.array([next_tp_hit[s][pos] for s, pos in zip(sides, positions)])
        if next_tp_hit is not None
        else np.full(len(positions), n, dtype=int)
    )
    sl_effective = (
        np.array([next_sl_hit[s][pos] for s, pos in zip(sides, positions)])
        if next_sl_hit is not None
        else np.full(len(positions), n, dtype=int)
    )

    # Whichever enabled trigger fires soonest (smallest index) wins; ties
    # resolve take_profit > stop_loss > max_hold (rare at real timeline
    # resolution, and an arbitrary but deterministic convention either way).
    stacked = np.stack([tp_effective, sl_effective, mh_effective], axis=1)
    winner_trigger = np.argmin(stacked, axis=1)  # 0=take_profit, 1=stop_loss, 2=max_hold
    winner_idx = stacked[np.arange(len(positions)), winner_trigger]
    triggered = winner_idx < n

    exit_price = np.empty(len(positions))
    exit_reason = np.full(len(positions), "settlement", dtype=object)
    cashed_out = np.zeros(len(positions), dtype=bool)
    won_if_held = np.array([1.0 if s == final_winner else 0.0 for s in sides])

    if triggered.any():
        idxs = winner_idx[triggered]
        s_sub = sides[triggered]
        exit_price[triggered] = np.array([side_price[side][i] for side, i in zip(s_sub, idxs)])
        cashed_out[triggered] = True
        reason_names = np.array(["take_profit", "stop_loss", "max_hold"])
        exit_reason[triggered] = reason_names[winner_trigger[triggered]]
    exit_price[~triggered] = won_if_held[~triggered]

    return_pct = (exit_price - entry_prices) / entry_prices
    hold_return_pct = (won_if_held - entry_prices) / entry_prices  # as if you'd never cashed out

    return pd.DataFrame(
        {
            "game_id": gtime["game_id"].iloc[0],
            "elapsed_frac": elapsed[positions],
            "side": sides,
            "entry_price": entry_prices,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "cashed_out": cashed_out,
            "return_pct": return_pct,
            "hold_to_settlement_return_pct": hold_return_pct,
            "won": (return_pct > 0).astype(int),  # "profitable trade" -- reused by cashout_validate.py
        }
    )


def build_candidate_cashout_trades(
    games: pd.DataFrame,
    timeline: pd.DataFrame,
    take_profit_price: float = None,
    max_hold_frac: float = None,
    stop_loss_frac: float = None,
) -> pd.DataFrame:
    winners = games.set_index("game_id")["final_winner"]
    frames = []
    for gid, gtime in timeline.sort_values("ts").groupby("game_id"):
        final_winner = winners.get(gid)
        if final_winner not in ("A", "B"):
            continue
        result = simulate_game_cashout(gtime, final_winner, take_profit_price, max_hold_frac, stop_loss_frac)
        if result is not None:
            frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def grid_search(trades: pd.DataFrame, time_bins: int = 5, price_bins: int = 5, min_trades: int = 100) -> pd.DataFrame:
    df = trades.copy()
    df["time_bucket"] = pd.cut(df["elapsed_frac"], bins=time_bins, labels=False, include_lowest=True)
    df["price_bucket"] = pd.cut(df["entry_price"], bins=price_bins, labels=False, include_lowest=True)
    df = df.dropna(subset=["time_bucket", "price_bucket"])

    time_cats = pd.cut(df["elapsed_frac"], bins=time_bins, include_lowest=True).cat.categories
    price_cats = pd.cut(df["entry_price"], bins=price_bins, include_lowest=True).cat.categories

    grouped = (
        df.groupby(["time_bucket", "price_bucket"])
        .agg(
            n_trades=("return_pct", "size"),
            avg_cashout_return_pct=("return_pct", "mean"),
            avg_hold_return_pct=("hold_to_settlement_return_pct", "mean"),
            cashout_rate=("cashed_out", "mean"),
            stop_loss_rate=("exit_reason", lambda s: (s == "stop_loss").mean()),
            avg_entry_price=("entry_price", "mean"),
        )
        .reset_index()
    )
    grouped["time_window"] = grouped["time_bucket"].apply(lambda i: str(time_cats[int(i)]))
    grouped["price_window"] = grouped["price_bucket"].apply(lambda i: str(price_cats[int(i)]))
    grouped["improvement_vs_hold"] = grouped["avg_cashout_return_pct"] - grouped["avg_hold_return_pct"]

    grouped = grouped[grouped["n_trades"] >= min_trades]
    return grouped.sort_values("avg_cashout_return_pct", ascending=False)


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest a cashout (early-exit) rule against historical odds paths.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Cash out once your contract's price reaches this absolute level (0-1). Omit to disable.",
    )
    p.add_argument(
        "--max-hold-frac",
        type=float,
        default=None,
        help="Cash out after holding this fraction of total game duration. Omit to disable.",
    )
    p.add_argument(
        "--stop-loss-frac",
        type=float,
        default=None,
        help="Cash out once price falls to (1 - this) of your entry price, e.g. 0.5 = cash out down 50%%. Omit to disable.",
    )
    p.add_argument("--time-bins", type=int, default=5)
    p.add_argument("--price-bins", type=int, default=5)
    p.add_argument("--min-trades", type=int, default=100)
    p.add_argument("--out", default="cashout_strategy_report.csv")
    args = p.parse_args(argv)

    if args.take_profit is None and args.max_hold_frac is None and args.stop_loss_frac is None:
        p.error("Set at least one of --take-profit / --max-hold-frac / --stop-loss-frac (otherwise this is just buy_strategy.py).")

    games, _, timeline = feat.load_tables(args.db)
    if timeline.empty:
        raise SystemExit("odds_timeline is empty -- see buy_strategy.py's docstring for why this needs it.")

    trades = build_candidate_cashout_trades(games, timeline, args.take_profit, args.max_hold_frac, args.stop_loss_frac)
    if trades.empty:
        raise SystemExit("No candidate trades produced -- check your database has settled games with odds_timeline data.")

    print(
        f"Simulated {len(trades):,} candidate entries across {trades['game_id'].nunique():,} games "
        f"with take-profit={args.take_profit}, max-hold-frac={args.max_hold_frac}, stop-loss-frac={args.stop_loss_frac}.\n"
    )
    print(f"  Overall avg return WITH cashout:   {trades['return_pct'].mean():.4f}")
    print(f"  Overall avg return HOLD-to-settle: {trades['hold_to_settlement_return_pct'].mean():.4f}")
    print(f"  Share of entries that cashed out early: {trades['cashed_out'].mean():.4f}")
    print("  Exit reason breakdown:")
    print(trades["exit_reason"].value_counts(normalize=True).to_string())
    print()

    report = grid_search(trades, args.time_bins, args.price_bins, args.min_trades)
    if report.empty:
        print(f"No bucket reached --min-trades={args.min_trades} -- try more data or fewer bins.")
        return

    print("Top (time-in-game, entry-price) buckets by average cashout return:")
    print(
        report[
            [
                "time_window",
                "price_window",
                "n_trades",
                "avg_entry_price",
                "cashout_rate",
                "stop_loss_rate",
                "avg_cashout_return_pct",
                "avg_hold_return_pct",
                "improvement_vs_hold",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    report.to_csv(args.out, index=False)
    print(f"\nFull grid saved to {args.out}")
    print(
        "\nSame caveats as buy_strategy.py: historical backtest, no fees/slippage modeled, "
        "not a guarantee of future results."
    )


if __name__ == "__main__":
    main()
