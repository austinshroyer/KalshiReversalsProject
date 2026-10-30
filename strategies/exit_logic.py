"""
Exit decision logic. position_monitor.py is the ONLY script that calls
this -- paper_trader.py no longer checks exits at all, specifically so
there is exactly one place exit decisions get made. Two independent loops
both deciding to close the same position could otherwise race and, in
--mode demo, fire duplicate closing orders for one real position.
"""

import logging
import time
from datetime import datetime, timezone

import db
from live_market_utils import live_price, to_ts

logger = logging.getLogger("kalshi_reversals.exit_logic")


def fetch_exit_data(client, trade, args):
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

    if args.take_profit is None:
        return {"held": held, "settled": False}  # hold-to-settlement mode: nothing more to fetch

    current_price = live_price(client, held)
    return {"held": held, "settled": False, "current_price": current_price}


def process_exit(conn, trading_client, trade, fetch_result, args) -> bool:
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

    if args.take_profit is None:
        return False  # hold-to-settlement mode: nothing to do until the market settles

    open_ts = to_ts(held.get("open_time"))
    close_ts = to_ts(held.get("close_time"))
    if not open_ts or not close_ts or close_ts <= open_ts:
        return False
    elapsed_frac = min(1.0, max(0.0, (int(time.time()) - open_ts) / (close_ts - open_ts)))

    current_price = fetch_result.get("current_price")
    if current_price is None:
        return False

    hit_target = current_price >= args.take_profit
    hit_time_cap = (
        args.max_hold_frac is not None and (elapsed_frac - trade["entry_elapsed_frac"]) >= args.max_hold_frac
    )
    if hit_target or hit_time_cap:
        return _close_trade(conn, trading_client, trade, current_price, "take_profit" if hit_target else "max_hold", args)
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
