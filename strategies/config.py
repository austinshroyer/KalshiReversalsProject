"""
Configuration for the Kalshi head-to-head reversal tracker.

Nothing here requires an API key: every endpoint this project uses
(series, events, markets, candlesticks, historical candlesticks) is
public market data and works unauthenticated. See README.md for details.
"""

import os

# --- Kalshi REST API -------------------------------------------------------
# "Recommended" production base URL per Kalshi's docs (api_environments).
# Despite the historical "elections" naming on the older shared host, this
# serves ALL Kalshi markets, not just elections.
KALSHI_BASE_URL = os.environ.get("KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2")

# Optional: only needed if you want higher rate limits / private endpoints.
# This project does not require them for anything it does.
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH")

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 1.5

# Shared client-side throttle (requests/second across ALL threads using one
# KalshiClient instance). Kalshi's Basic tier is roughly 200 tokens/sec with
# most reads costing 10 tokens, i.e. ~20 req/s sustained -- this stays under
# that with headroom, since paper_trader.py's concurrent workers all share
# one client and can otherwise burst well past the real limit. Raise this if
# you've confirmed a higher tier/limit for your account; lower it if you're
# still seeing repeated 429s in the logs even with this in place.
MAX_REQUESTS_PER_SECOND = 15

# --- Discovery scope --------------------------------------------------------
# Category used to discover sports series via GET /series?category=Sports.
SERIES_CATEGORY = "Sports"

# Candlestick resolution used for reconstructing each game's odds timeline.
# 1 = 1-minute candles (finest available), 60 = hourly, 1440 = daily.
CANDLESTICK_PERIOD_INTERVAL_MINUTES = 1

# --- Reversal detection ------------------------------------------------------
# A "significant reversal" is a change in which side is favored (crosses the
# 50% implied-probability line) where the NEW favorite's implied win
# probability climbed by at least this many percentage points, measured from
# the low point it touched while it was still the underdog up to the moment
# it overtook the old favorite. See odds.py:find_reversals for the exact
# algorithm and README.md for the reasoning/assumptions behind this
# definition (the prompt's "increasing by at least 20%" is ambiguous between
# 20 percentage points and a 20% relative change -- this project uses 20
# percentage points and exposes the threshold here so it's a one-line change
# if you want the other interpretation).
SIGNIFICANT_SWING_THRESHOLD = 0.20

# Probability source preference when reading a candlestick, in order:
# 1) midpoint of yes_bid/yes_ask close prices (best proxy for "market odds")
# 2) last traded price close
# 3) mean traded price
# A candlestick with none of these (no quotes, no trades) is skipped.
