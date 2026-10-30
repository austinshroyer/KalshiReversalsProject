"""
Scans every series in Kalshi's Sports category and prints just the ones
that currently have at least one eligible (two-team, binary-market) game
either already in progress or expected to start within --hours-ahead hours
-- formatted as a ready-to-paste --series argument for paper_trader.py.

This is different from paper_trader.py's --list-series, which lists EVERY
series in the category regardless of whether anything is actually
happening in it right now -- most won't have a live or imminent game at
any given moment. This script does the actual discovery scan (the same
eligibility + pregame logic paper_trader.py uses internally) and narrows
the list down to what's worth watching right now. Pointing paper_trader.py
at a narrowed list also cuts its own per-cycle request volume, since it
won't waste calls re-polling series that turn out to be empty anyway.

"Expected to start within --hours-ahead hours" uses the same heuristic as
paper_trader.py's pregame filter: Kalshi's expected_expiration_time field
("the time the event is likely to resolve -- for a sports game, typically
a few hours after the scheduled start"). Not an exact kickoff-time check --
see paper_trader.py's module docstring for the full rationale.

Usage:
    python find_active_series.py
    python find_active_series.py --hours-ahead 4
    python find_active_series.py --hours-ahead 8 --max-workers 12
"""

import argparse
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import config
from kalshi_client import KalshiClient


def _to_ts(iso_str):
    if not iso_str:
        return None
    try:
        return int(datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def plausibly_relevant(m_a: dict, m_b: dict, hours_ahead: float) -> bool:
    """Same heuristic as paper_trader.py's plausibly_started. Missing field
    -> assume relevant rather than silently dropping the event."""
    exp_ts = _to_ts(m_a.get("expected_expiration_time")) or _to_ts(m_b.get("expected_expiration_time"))
    if exp_ts is None:
        return True
    hours_until_expiration = (exp_ts - time.time()) / 3600
    return hours_until_expiration <= hours_ahead


def scan_series(client: KalshiClient, series_ticker: str, hours_ahead: float):
    """Runs in a worker thread. Returns (series_ticker, qualifying_event_count)."""
    count = 0
    try:
        for event in client.get_events(series_ticker=series_ticker, status="open", with_nested_markets=True):
            markets = event.get("markets") or []
            binary = [m for m in markets if m.get("market_type") == "binary"]
            if len(binary) != 2:
                continue
            m_a, m_b = binary[0], binary[1]
            if plausibly_relevant(m_a, m_b, hours_ahead):
                count += 1
    except Exception as exc:
        print(f"  (skipped {series_ticker}: {exc})")
    return series_ticker, count


def main():
    p = argparse.ArgumentParser(
        description="Find Sports-category series with a game live now or starting soon, formatted for paper_trader.py's --series."
    )
    p.add_argument(
        "--hours-ahead",
        type=float,
        default=8.0,
        help="Include games already in progress, or expected to resolve within this many hours "
        "(a proxy for 'starting soon', per the module docstring). Default 8.0.",
    )
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    client = KalshiClient()
    print("Discovering Sports-category series...")
    all_series = client.get_series_list(category=config.SERIES_CATEGORY)
    print(f"Found {len(all_series)} series total -- scanning each for live/upcoming (within {args.hours_ahead}h) games...\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(scan_series, client, s["ticker"], args.hours_ahead) for s in all_series]
        for future in concurrent.futures.as_completed(futures):
            series_ticker, count = future.result()
            if count > 0:
                results.append((series_ticker, count))

    if not results:
        print(f"No series currently have a game live or starting within {args.hours_ahead} hours.")
        return

    results.sort(key=lambda r: -r[1])

    print(f"{'Series':<20} Qualifying games")
    for series_ticker, count in results:
        print(f"{series_ticker:<20} {count}")

    series_str = ",".join(t for t, _ in results)
    print("\nPaste this into your paper_trader.py command:\n")
    print(f"  --series {series_str}")


if __name__ == "__main__":
    main()
