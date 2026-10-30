"""
Runs all three predictive analyses against reversals.db in sequence.
Equivalent to running count_model.py, timing_model.py, and buy_strategy.py
individually -- use this for a full report; use the individual scripts when
tuning one piece (e.g. --bins, --time-bins) without re-running everything.

Usage:
    python train_all.py --db reversals.db
"""

import argparse

import count_model
import timing_model
import buy_strategy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="reversals.db")
    args, _ = p.parse_known_args()

    print("############################################")
    print("# 1) Significant-reversal COUNT model (pregame info only)")
    print("############################################")
    count_model.main(["--db", args.db])

    print("\n############################################")
    print("# 2) WHEN reversals occur (descriptive timing + hazard model)")
    print("############################################")
    timing_model.main(["--db", args.db])

    print("\n############################################")
    print("# 3) Optimal buy-timing backtest")
    print("############################################")
    try:
        buy_strategy.main(["--db", args.db])
    except SystemExit as e:
        print(e)


if __name__ == "__main__":
    main()
