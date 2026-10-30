"""
Live paper-trading runner: watches currently-open Kalshi sports markets and,
when a game's live state matches a bucket you found via buy_strategy.py /
cashout_strategy.py, "buys" the underdog -- either as a pure local
simulation (--mode simulate, no credentials needed) or as a real order on
Kalshi's own demo/paper-money exchange (--mode demo, needs a demo API key).
Handles entries AND exits (settlement + optional cashout) in one process.

CASHOUT + RE-ENTRY: three independent early-exit triggers, any combination
of which you can set -- whichever fires first closes the position:
  --take-profit      exit once price rises to this absolute level (0-1)
  --stop-loss-frac    exit once price falls to (1 - this) of what you paid,
                       e.g. 0.5 = cash out once you're down 50% on the trade
  --max-hold-frac     exit after holding this fraction of the game's
                       duration since entry, regardless of price
None of these are required -- with none set, positions just hold to
settlement (the original behavior). See cashout_strategy.py for the
backtest this mirrors. Once a position is cashed out early (by any of the
three triggers), the underlying game is still live, so this can buy back
into the SAME game if it re-enters your bucket later, up to
--max-entries-per-game. A game that settles is never re-entered -- it's
over. (Hold-to-settlement mode, i.e. none of the three set, never
re-enters either, since the only way a trade closes in that mode is
settlement.)

PREGAME FILTER: skips events whose game almost certainly hasn't started
yet, using expected_expiration_time (a real, documented Kalshi field: "the
time the event is likely to resolve -- for a sports game, typically a few
hours after the scheduled start"). Kalshi doesn't expose a dedicated
scheduled-kickoff-time field, so this is a heuristic, not an exact check --
tune --max-hours-before-game per sport if it's too tight or too loose.
Matters for correctness, not just speed: without it, a market that's been
open for days before the actual game (common for Kalshi sports markets)
can show a small elapsed_frac that looks in-window despite the game not
having started.

CONCURRENCY: event listing (per series), live price lookups, and exit
refreshes all run through a thread pool (--max-workers, default 8) via a
shared KalshiClient, which also rate-limits itself (see kalshi_client.py)
so concurrency doesn't outrun Kalshi's actual request limits. All database
reads/writes stay single-threaded in the main thread.

SETUP FOR --mode demo (skip this entirely for --mode simulate):
  1. Create a demo account at https://demo.kalshi.co/ (mock info is fine).
  2. Generate a key pair: python generate_kalshi_keys.py
  3. Upload the printed PUBLIC key to your demo account's Settings -> API
     page. Kalshi gives you back an API Key ID.
  4. Set environment variables (do NOT hardcode these anywhere):
       export KALSHI_API_KEY_ID=<the key id Kalshi gave you>
       export KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private.pem
       export KALSHI_BASE_URL=https://external-api.demo.kalshi.co/trade-api/v2
     This script refuses to run in --mode demo unless KALSHI_BASE_URL
     contains "demo", as a guard against accidentally pointing it at
     production.

Usage:
    # Hold-to-settlement, no credentials needed
    python paper_trader.py --db reversals.db --series KXNFLGAME \\
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \\
        --mode simulate

    # Cashout with re-entry: exit early on a price target or time cap, and
    # allow buying back into the same game if it re-enters the bucket
    python paper_trader.py --db reversals.db --series KXNFLGAME \\
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \\
        --take-profit 0.55 --max-hold-frac 0.25 --max-entries-per-game 3 \\
        --mode simulate

    # Stop-loss only: no take-profit, just cut losses at 50% down, hold
    # otherwise to settlement
    python paper_trader.py --db reversals.db --series KXNFLGAME \\
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \\
        --stop-loss-frac 0.5 --mode simulate

    # Real orders on Kalshi's own demo/paper-money exchange
    python paper_trader.py --db reversals.db --series KXNFLGAME \\
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \\
        --take-profit 0.55 --mode demo --contracts 1

Ctrl+C to stop -- open positions stay in the database as-is and pick back
up (status stays 'open') the next time you run this with the same --db.
"""

import argparse
import concurrent.futures
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import config
import db
from kalshi_client import KalshiClient, price_from_market
from pipeline import discover_series_tickers

logger = logging.getLogger("kalshi_reversals.paper_trader")


def _to_ts(iso_str):
    if not iso_str:
        return None
    try:
        return int(datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def compute_elapsed_frac(m_a: dict, m_b: dict):
    """Free -- uses open_time/close_time already present in the market data
    from get_events()/get_event(), no API call."""
    open_ts = _to_ts(m_a.get("open_time")) or _to_ts(m_b.get("open_time"))
    close_ts = _to_ts(m_a.get("close_time")) or _to_ts(m_b.get("close_time"))
    if not open_ts or not close_ts or close_ts <= open_ts:
        return None
    now_ts = int(time.time())
    return min(1.0, max(0.0, (now_ts - open_ts) / (close_ts - open_ts)))


def plausibly_started(m_a: dict, m_b: dict, max_hours_before_game: float) -> bool:
    """Free -- Kalshi doesn't expose a dedicated scheduled-kickoff-time
    field, but expected_expiration_time is real and documented: 'the time
    the event is likely to resolve (for a sports game, typically a few
    hours after the scheduled start)'. If an event isn't expected to
    resolve for a long time yet, the game almost certainly hasn't started
    -- the market is just open for early/pregame trading. This is a
    heuristic, not an exact kickoff check; tune --max-hours-before-game per
    sport if it's too tight or too loose. Missing field -> don't exclude."""
    exp_ts = _to_ts(m_a.get("expected_expiration_time")) or _to_ts(m_b.get("expected_expiration_time"))
    if exp_ts is None:
        return True
    hours_until_expiration = (exp_ts - time.time()) / 3600
    return hours_until_expiration <= max_hours_before_game


def live_price(client: KalshiClient, m: dict):
    """Costs one API call -- only call this once every free filter has
    already confirmed the event/position is worth pricing."""
    live = client.get_market(m["ticker"])
    bid = price_from_market(live, "yes_bid")
    ask = price_from_market(live, "yes_ask")
    if bid is None and ask is None:
        return None
    return (bid + ask) / 2 if (bid is not None and ask is not None) else (bid if bid is not None else ask)


# -- entry: discovery (parallel, free filters) --------------------------------

def _fetch_series_candidates(client: KalshiClient, series_ticker: str, args):
    """Runs in a worker thread: lists open events for one series and applies
    every free filter (binary-market-count, pregame heuristic, time bucket)
    before returning. Returns (raw_event_count, filtered_list)."""
    raw_count = 0
    filtered = []
    try:
        for event in client.get_events(series_ticker=series_ticker, status="open", with_nested_markets=True):
            raw_count += 1
            markets = event.get("markets") or []
            binary = [m for m in markets if m.get("market_type") == "binary"]
            if len(binary) != 2:
                continue
            m_a, m_b = binary[0], binary[1]

            if not plausibly_started(m_a, m_b, args.max_hours_before_game):
                continue

            elapsed_frac = compute_elapsed_frac(m_a, m_b)
            if elapsed_frac is None or not (args.time_min <= elapsed_frac < args.time_max):
                continue

            filtered.append((event, m_a, m_b, elapsed_frac))
    except Exception:
        logger.exception("Failed to list open events for %s -- skipping this series this cycle", series_ticker)
    return raw_count, filtered


def find_open_candidate_events(client: KalshiClient, series_tickers, args):
    """Concurrently lists open events across every watched series, applying
    free filters before returning. Returns (total_raw_events, candidates)."""
    total_raw = 0
    all_candidates = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_fetch_series_candidates, client, s, args) for s in series_tickers]
        for future in concurrent.futures.as_completed(futures):
            raw_count, filtered = future.result()
            total_raw += raw_count
            all_candidates.extend(filtered)
    return total_raw, all_candidates


def filter_blocked_games(conn, candidates, args):
    """Main-thread-only (SQLite access): drops candidates whose game already
    has an open position, has settled, or has hit the re-entry cap. Cheap
    local reads -- no network involved. A game cashed out early is NOT
    blocked here, allowing re-entry."""
    to_price = []
    for event, m_a, m_b, elapsed_frac in candidates:
        game_id = event["event_ticker"]
        status_info = db.game_trade_status(conn, game_id)
        if status_info["has_open"] or status_info["has_settled"]:
            continue
        if status_info["entry_count"] >= args.max_entries_per_game:
            continue
        to_price.append((event, m_a, m_b, elapsed_frac))
    return to_price


def price_candidates(client: KalshiClient, candidates, args):
    """Parallel network calls for live price, only for candidates that
    already survived every free filter and the DB re-entry check."""
    priced = []
    if not candidates:
        return priced
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(live_price, client, m_a): (event, m_a, m_b, elapsed_frac)
            for event, m_a, m_b, elapsed_frac in candidates
        }
        for future in concurrent.futures.as_completed(future_map):
            event, m_a, m_b, elapsed_frac = future_map[future]
            try:
                prob_a = future.result()
            except Exception:
                logger.exception("Price fetch failed for %s -- skipping this cycle", event["event_ticker"])
                continue
            if prob_a is not None:
                priced.append((event, m_a, m_b, elapsed_frac, prob_a))
    return priced


def do_enter(conn, client, trading_client, event, m_a, m_b, elapsed_frac, prob_a, args, bucket_label) -> bool:
    """Sequential, main-thread: given an already-priced candidate, applies
    the price-bucket filter and performs the actual entry (optional demo
    order + DB write). Returns True iff a position was actually opened."""
    game_id = event["event_ticker"]
    side = "A" if prob_a < 0.5 else "B"
    entry_price = prob_a if side == "A" else 1 - prob_a
    if entry_price <= 0 or not (args.price_min <= entry_price < args.price_max):
        return False

    ticker = m_a["ticker"] if side == "A" else m_b["ticker"]

    order_id = None
    if args.mode == "demo":
        live = client.get_market(ticker)
        ask = price_from_market(live, "yes_ask")
        limit_price = min(0.99, round((ask if ask is not None else entry_price) + 0.01, 2))
        try:
            resp = trading_client.create_order(
                ticker=ticker, side="bid", count=args.contracts, price=limit_price,
                time_in_force="immediate_or_cancel",
            )
        except Exception:
            logger.exception("Order placement failed for %s -- skipping this game this cycle", ticker)
            return False
        order_id = resp.get("order_id")
        fill_count = float(resp.get("fill_count") or 0)
        if fill_count <= 0:
            logger.info("IOC buy for %s at %.2f didn't fill (no liquidity) -- not recording a position", ticker, limit_price)
            return False
        entry_price = float(resp.get("average_fill_price") or entry_price)

    logger.info(
        "%s ENTER %s: bought %s (%s) at %.3f  [elapsed=%.2f]",
        "[DEMO]" if args.mode == "demo" else "[SIM]", game_id, side, ticker, entry_price, elapsed_frac,
    )
    db.insert_paper_trade(
        conn,
        {
            "game_id": game_id,
            "ticker": ticker,
            "side": side,
            "bucket_label": bucket_label,
            "entry_time": datetime.now(tz=timezone.utc).isoformat(),
            "entry_elapsed_frac": elapsed_frac,
            "entry_price": entry_price,
            "entry_order_id": order_id,
            "status": "open",
            "dry_run": 0 if args.mode == "demo" else 1,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    conn.commit()
    return True


# -- exit (parallel fetch, sequential decide/write) ----------------------------

def _cashout_enabled(args) -> bool:
    return args.take_profit is not None or args.max_hold_frac is not None or args.stop_loss_frac is not None


def _fetch_exit_data(client: KalshiClient, trade, args):
    """Runs in a worker thread: gathers everything needed to decide whether
    to close this position, without touching the database."""
    try:
        event = client.get_event(trade["game_id"])
    except Exception:
        logger.exception("Could not refresh %s -- will retry next cycle", trade["game_id"])
        return {"error": True}

    markets = {m["ticker"]: m for m in (event.get("markets") or [])}
    held = markets.get(trade["ticker"])
    if held is None:
        return {"held": None}

    if held.get("status") == "settled" or held.get("result") in ("yes", "no"):
        return {"held": held, "settled": True}

    if not _cashout_enabled(args):
        return {"held": held, "settled": False}  # hold-to-settlement mode: nothing more to fetch

    # take-profit and stop-loss both need a live price; a max-hold-frac-only
    # setup doesn't strictly need it, but fetching it anyway is one cheap
    # extra call and keeps this function simple.
    current_price = live_price(client, held)
    return {"held": held, "settled": False, "current_price": current_price}


def _process_exit(conn, trading_client, trade, fetch_result, args) -> bool:
    """Sequential, main-thread: decides whether to close, and writes it if
    so. No network calls -- fetch_result was already gathered in parallel.
    Returns True iff a close was actually written."""
    if fetch_result is None or fetch_result.get("error"):
        return False
    held = fetch_result.get("held")
    if held is None:
        return False

    if fetch_result.get("settled"):
        exit_price = 1.0 if held.get("result") == "yes" else 0.0
        return _close_trade(conn, trading_client, trade, exit_price, "settlement", args, already_settled=True)

    if not _cashout_enabled(args):
        return False  # hold-to-settlement mode: nothing to do until the market settles

    current_price = fetch_result.get("current_price")
    if current_price is None:
        return False

    hit_target = args.take_profit is not None and current_price >= args.take_profit
    hit_stop_loss = (
        args.stop_loss_frac is not None and current_price <= trade["entry_price"] * (1 - args.stop_loss_frac)
    )

    hit_time_cap = False
    if args.max_hold_frac is not None:
        open_ts = _to_ts(held.get("open_time"))
        close_ts = _to_ts(held.get("close_time"))
        if open_ts and close_ts and close_ts > open_ts:
            elapsed_frac = min(1.0, max(0.0, (int(time.time()) - open_ts) / (close_ts - open_ts)))
            hit_time_cap = (elapsed_frac - trade["entry_elapsed_frac"]) >= args.max_hold_frac

    if hit_target or hit_stop_loss or hit_time_cap:
        reason = "take_profit" if hit_target else ("stop_loss" if hit_stop_loss else "max_hold")
        return _close_trade(conn, trading_client, trade, current_price, reason, args)
    return False


def _close_trade(conn, trading_client, trade, exit_price, reason, args, already_settled=False) -> bool:
    exit_order_id = None
    if args.mode == "demo" and not already_settled:
        try:
            resp = trading_client.create_order(
                ticker=trade["ticker"], side="ask", count=args.contracts,
                price=max(0.01, round(exit_price - 0.01, 2)),
                time_in_force="immediate_or_cancel", reduce_only=True,
            )
            exit_order_id = resp.get("order_id")
            if float(resp.get("fill_count") or 0) <= 0:
                logger.warning("Exit order for %s didn't fill -- will retry closing it next cycle", trade["ticker"])
                return False
            exit_price = float(resp.get("average_fill_price") or exit_price)
        except Exception:
            logger.exception("Exit order failed for %s -- will retry next cycle", trade["ticker"])
            return False

    return_pct = (exit_price - trade["entry_price"]) / trade["entry_price"]
    logger.info(
        "%s EXIT %s (%s): %.3f -> %.3f  return=%.1f%%",
        "[DEMO]" if args.mode == "demo" else "[SIM]", trade["game_id"], reason,
        trade["entry_price"], exit_price, return_pct * 100,
    )
    db.close_paper_trade(
        conn,
        trade["trade_id"],
        {
            "status": "settled" if reason == "settlement" else "cashed_out",
            "exit_time": datetime.now(tz=timezone.utc).isoformat(),
            "exit_price": exit_price,
            "exit_order_id": exit_order_id,
            "exit_reason": reason,
            "return_pct": return_pct,
        },
    )
    conn.commit()
    return True


def main():
    p = argparse.ArgumentParser(description="Forward-test one strategy bucket live against currently-open Kalshi markets.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument(
        "--series",
        default=None,
        help="Comma-separated series tickers to watch, e.g. KXNFLGAME,KXNBAGAME. "
        "Omit to auto-discover every series in the Sports category.",
    )
    p.add_argument("--list-series", action="store_true", help="Print every series ticker + title in the Sports category and exit immediately.")
    p.add_argument("--time-min", type=float, default=None)
    p.add_argument("--time-max", type=float, default=None)
    p.add_argument("--price-min", type=float, default=None)
    p.add_argument("--price-max", type=float, default=None)
    p.add_argument("--take-profit", type=float, default=None, help="Cashout trigger price (0-1, absolute). Omit to not use a take-profit trigger.")
    p.add_argument(
        "--max-hold-frac",
        type=float,
        default=None,
        help="Time-based cashout: force an exit after holding this fraction of the game's duration "
        "since entry, regardless of price. Independent of --take-profit/--stop-loss-frac.",
    )
    p.add_argument(
        "--stop-loss-frac",
        type=float,
        default=None,
        help="Price-based cashout: exit once the position's value falls to (1 - this) of what you paid "
        "-- e.g. 0.5 = cash out once you've lost half your stake. Independent of --take-profit/--max-hold-frac.",
    )
    p.add_argument(
        "--max-entries-per-game",
        type=int,
        default=3,
        help="Cap on total entries (across re-entries after cashouts) per game. Default 3. "
        "Only relevant with --take-profit -- hold-to-settlement trades never re-enter anyway.",
    )
    p.add_argument(
        "--max-hours-before-game",
        type=float,
        default=6.0,
        help="Skip events whose expected_expiration_time is more than this many hours away "
        "(heuristic for 'the game plausibly hasn't started yet'). Default 6.0.",
    )
    p.add_argument("--max-workers", type=int, default=8, help="Thread pool size for concurrent Kalshi API calls. Default 8.")
    p.add_argument("--mode", choices=["simulate", "demo"], default="simulate")
    p.add_argument("--contracts", type=float, default=1.0)
    p.add_argument("--poll-interval", type=float, default=60.0)
    p.add_argument("--max-open-trades", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--log-file",
        default=None,
        help="Also write logs to this file. Recommended if you're running this in the background "
        "(nohup/screen/&) -- logging.basicConfig writes to stderr by default, which is easy to lose "
        "track of when a process is detached; this writes reliably to a real file regardless.",
    )
    args = p.parse_args()

    if args.list_series:
        client = KalshiClient()
        for s in client.get_series_list(category=config.SERIES_CATEGORY):
            print(f"{s.get('ticker', '?'):<20} {s.get('title', '')}")
        return

    if None in (args.time_min, args.time_max, args.price_min, args.price_max):
        p.error("--time-min/--time-max/--price-min/--price-max are required (unless using --list-series)")

    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    trading_client = None
    if args.mode == "demo":
        if "demo" not in config.KALSHI_BASE_URL:
            logger.error(
                "Refusing to run --mode demo: KALSHI_BASE_URL ('%s') doesn't look like a demo host. "
                "Set it to https://external-api.demo.kalshi.co/trade-api/v2 first.",
                config.KALSHI_BASE_URL,
            )
            raise SystemExit(1)
        from kalshi_trading_client import KalshiTradingClient  # imported lazily: simulate mode needs no credentials

        trading_client = KalshiTradingClient()

    if args.max_hold_frac is not None and args.time_max + args.max_hold_frac > 1.0:
        logger.warning(
            "--time-max (%.2f) + --max-hold-frac (%.2f) > 1.0: entries near the top of your time "
            "window may NEVER reach the time cap before the game naturally settles first -- "
            "elapsed_frac can't exceed 1.0, so entry_elapsed_frac + max_hold_frac must stay <= 1.0 "
            "for the cap to be reachable at all. Entries at elapsed_frac > %.2f structurally cannot "
            "hit max_hold; they'll only ever exit via take_profit or settlement. Lower --max-hold-frac "
            "or --time-max if you want the time cap to apply across your whole bucket.",
            args.time_max, args.max_hold_frac, 1.0 - args.max_hold_frac,
        )

    db.init_db(args.db)
    client = KalshiClient()
    explicit_series = [s.strip() for s in args.series.split(",")] if args.series else None
    series_tickers = discover_series_tickers(client, explicit_series)
    if not explicit_series:
        logger.info("No --series given -- auto-discovered %d Sports-category series.", len(series_tickers))

    bucket_label = f"time[{args.time_min},{args.time_max}) price[{args.price_min},{args.price_max})"
    if _cashout_enabled(args):
        bucket_label += (
            f" take_profit={args.take_profit} max_hold_frac={args.max_hold_frac} "
            f"stop_loss_frac={args.stop_loss_frac}"
        )

    logger.info(
        "Watching %s | bucket: %s | mode=%s | max_workers=%d | max_hours_before_game=%.1f",
        series_tickers, bucket_label, args.mode, args.max_workers, args.max_hours_before_game,
    )

    while True:
        try:
            with db.connect(args.db) as conn:
                open_trades = db.get_open_paper_trades(conn)
                closed_count = 0
                if open_trades:
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                        future_map = {
                            executor.submit(_fetch_exit_data, client, trade, args): trade for trade in open_trades
                        }
                        for future in concurrent.futures.as_completed(future_map):
                            trade = future_map[future]
                            try:
                                result = future.result()
                            except Exception:
                                logger.exception("Exit fetch failed for %s -- will retry next cycle", trade["game_id"])
                                continue
                            if _process_exit(conn, trading_client, trade, result, args):
                                closed_count += 1

                open_count = len(db.get_open_paper_trades(conn))
                at_capacity_from_start = open_count >= args.max_open_trades

                raw_count, candidates = find_open_candidate_events(client, series_tickers, args)
                to_price = filter_blocked_games(conn, candidates, args)

                entered_count = 0
                if open_count < args.max_open_trades:
                    priced = price_candidates(client, to_price, args)
                    for event, m_a, m_b, elapsed_frac, prob_a in priced:
                        if open_count >= args.max_open_trades:
                            break
                        if do_enter(conn, client, trading_client, event, m_a, m_b, elapsed_frac, prob_a, args, bucket_label):
                            open_count += 1
                            entered_count += 1

                final_open = len(db.get_open_paper_trades(conn))
                logger.info(
                    "poll cycle complete: %d closed, %d open event(s) seen, %d passed pregame/time filters, "
                    "%d entered, %d open position(s) held%s",
                    closed_count, raw_count, len(candidates), entered_count, final_open,
                    " (was already at --max-open-trades cap before this cycle)" if at_capacity_from_start else "",
                )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Unhandled error in poll loop -- continuing")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
