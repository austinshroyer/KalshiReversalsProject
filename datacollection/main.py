"""
CLI entry point.

Examples:
    # Scan all Kalshi "Sports" series for settled two-team games, no date filter
    python main.py --db reversals.db

    # Just NFL and NBA game series, only games closing after 2026-01-01
    python main.py --db reversals.db --series KXNFLGAME,KXNBAGAME --since 2026-01-01

    # Quick smoke test against a handful of games
    python main.py --db reversals.db --limit 5
"""

import argparse
import logging
from datetime import datetime, timezone

from pipeline import run


def parse_args():
    p = argparse.ArgumentParser(description="Track significant favorite reversals in Kalshi sports markets.")
    p.add_argument("--db", default="reversals.db", help="SQLite database file path (default: reversals.db)")
    p.add_argument(
        "--series",
        default=None,
        help="Comma-separated Kalshi series tickers to restrict to (e.g. KXNFLGAME,KXNBAGAME). "
        "Default: auto-discover every series in the Sports category.",
    )
    p.add_argument(
        "--since",
        default=None,
        help="Only include games with a market close time on/after this date (YYYY-MM-DD, UTC).",
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after storing this many new games.")
    p.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-fetch and overwrite games already present in the database (default: skip them).",
    )
    p.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep between games, to stay well under rate limits.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    series_tickers = [s.strip() for s in args.series.split(",")] if args.series else None

    min_close_ts = None
    if args.since:
        dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        min_close_ts = int(dt.timestamp())

    run(
        db_path=args.db,
        series_tickers=series_tickers,
        min_close_ts=min_close_ts,
        limit=args.limit,
        skip_existing=not args.reprocess,
        request_pause_seconds=args.pause,
    )


if __name__ == "__main__":
    main()
