"""
Dedicated exit-only companion to paper_trader.py.

paper_trader.py's discovery scan can take a while when watching many
series, and bundling exit-checking into the same cycle meant cashout
opportunities only got checked as often as that slow scan ran -- letting
positions ride to full settlement that could have been cashed out earlier.
This script does nothing but watch currently-open positions and check
their exit conditions (settlement or cashout), on its own tight
--poll-interval, completely independent of however long discovery takes.

Run this ALONGSIDE paper_trader.py (same --db), not instead of it:

    python paper_trader.py --db reversals.db --series KXNFLGAME \\
        --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \\
        --max-entries-per-game 3 --mode simulate --poll-interval 60 &

    python position_monitor.py --db reversals.db \\
        --take-profit 0.55 --max-hold-frac 0.25 \\
        --mode simulate --poll-interval 10 &

paper_trader.py no longer checks exits itself -- this script is the ONLY
place exit decisions happen, so the two processes never race to close the
same position (which could otherwise fire duplicate --mode demo orders for
one real position).

Omit --take-profit here and positions only ever close via settlement
(matching the old hold-to-settlement behavior) -- still worth running
alongside paper_trader.py even then, since it means settlement gets
detected on this script's tight poll interval rather than discovery's slow
one, closing out positions sooner after a game actually ends.

--mode demo setup is identical to paper_trader.py's -- see that script's
docstring.
"""

import argparse
import concurrent.futures
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import db
from exit_logic import fetch_exit_data, process_exit
from kalshi_client import KalshiClient
from live_market_utils import build_trading_client

logger = logging.getLogger("kalshi_reversals.position_monitor")


def main():
    p = argparse.ArgumentParser(description="Dedicated, tightly-polled exit checker -- run alongside paper_trader.py.")
    p.add_argument("--db", default="reversals.db")
    p.add_argument("--take-profit", type=float, default=None, help="Cashout trigger price. Omit to only ever close via settlement.")
    p.add_argument("--max-hold-frac", type=float, default=None, help="Cashout time cap (fraction of game duration since entry). Requires --take-profit.")
    p.add_argument("--mode", choices=["simulate", "demo"], default="simulate")
    p.add_argument("--contracts", type=float, default=1.0)
    p.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Default is much tighter than paper_trader.py's -- this loop is cheap (only your "
        "currently-open positions, no discovery scan), so it can afford to check often.",
    )
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    if args.max_hold_frac is not None and args.take_profit is None:
        p.error("--max-hold-frac requires --take-profit")

    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    trading_client = build_trading_client(args.mode, logger)

    db.init_db(args.db)
    client = KalshiClient()

    logger.info(
        "Monitoring open positions in %s | take_profit=%s max_hold_frac=%s | mode=%s | poll_interval=%.1fs",
        args.db, args.take_profit, args.max_hold_frac, args.mode, args.poll_interval,
    )

    while True:
        try:
            with db.connect(args.db) as conn:
                open_trades = db.get_open_paper_trades(conn)
                closed_this_cycle = 0
                if open_trades:
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                        future_map = {
                            executor.submit(fetch_exit_data, client, trade, args): trade for trade in open_trades
                        }
                        for future in concurrent.futures.as_completed(future_map):
                            trade = future_map[future]
                            try:
                                result = future.result()
                            except Exception:
                                logger.exception("Exit fetch failed for %s -- will retry next cycle", trade["game_id"])
                                continue
                            if process_exit(conn, trading_client, trade, result, args):
                                closed_this_cycle += 1

                final_open = len(db.get_open_paper_trades(conn))
                logger.info(
                    "monitor cycle complete: %d position(s) checked, %d closed, %d still open",
                    len(open_trades), closed_this_cycle, final_open,
                )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Unhandled error in monitor loop -- continuing")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
