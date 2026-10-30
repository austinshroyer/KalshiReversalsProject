"""
Minimal Kalshi REST client covering just what this project needs:

  - GET /series                                  (discover sports series)
  - GET /events                                  (list events, paginated)
  - GET /series/{series}/markets/{ticker}/candlesticks       (live/recent candles)
  - GET /historical/markets/{ticker}/candlesticks            (archived candles)

All of these are public market-data endpoints and do not require an API key.
See https://docs.kalshi.com/getting_started/quick_start_market_data for
confirmation these work unauthenticated.

If KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH are set in config.py, requests
are signed anyway (mainly useful if you want higher rate limits), but it is
never required for the endpoints above.
"""

import base64
import threading
import time
from collections import deque
from datetime import datetime, timezone

import requests

import config

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False


class KalshiAPIError(RuntimeError):
    pass


class RateLimiter:
    """Thread-safe sliding-window limiter: blocks callers so no more than
    max_per_second requests get admitted across ANY number of threads
    sharing one KalshiClient. This exists because paper_trader.py's
    concurrent workers (--max-workers) all share one client instance --
    without a shared throttle, a burst of concurrent requests can exceed
    Kalshi's actual sustained rate limit and trigger repeated 429s that
    retries alone don't reliably recover from."""

    def __init__(self, max_per_second: float):
        self.max_per_second = max_per_second
        self.lock = threading.Lock()
        self.timestamps = deque()

    def acquire(self):
        with self.lock:
            while True:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] > 1.0:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_per_second:
                    self.timestamps.append(now)
                    return
                sleep_for = 1.0 - (now - self.timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)


class KalshiClient:
    def __init__(self, base_url: str = None, api_key_id: str = None, private_key_path: str = None):
        self.base_url = (base_url or config.KALSHI_BASE_URL).rstrip("/")
        self.api_key_id = api_key_id or config.KALSHI_API_KEY_ID
        self.private_key_path = private_key_path or config.KALSHI_PRIVATE_KEY_PATH
        self._private_key = None
        if self.api_key_id and self.private_key_path:
            if not _CRYPTO_AVAILABLE:
                raise RuntimeError(
                    "KALSHI_API_KEY_ID/KALSHI_PRIVATE_KEY_PATH were set but the "
                    "'cryptography' package is not installed. Either `pip install "
                    "cryptography` or unset those env vars -- this project does not "
                    "require authentication for any of the data it fetches."
                )
            with open(self.private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = requests.Session()
        self._rate_limiter = RateLimiter(config.MAX_REQUESTS_PER_SECOND)

    # -- low level ------------------------------------------------------

    def _sign_headers(self, method: str, path: str) -> dict:
        if not self._private_key:
            return {}
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        # Signed path is the path from the API root, without query string.
        signed_path = f"/trade-api/v2{path}"
        headers = self._sign_headers("GET", signed_path)

        last_error = None
        for attempt in range(config.MAX_RETRIES):
            try:
                self._rate_limiter.acquire()
                resp = self.session.get(
                    url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT_SECONDS
                )
                if resp.status_code == 429:
                    last_error = "HTTP 429 (rate limited)"
                    time.sleep(config.RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                if resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code} (server error)"
                    time.sleep(config.RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    return {"__error__": True, "status": resp.status_code, "body": _safe_json(resp)}
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(config.RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise KalshiAPIError(f"GET {path} failed after {config.MAX_RETRIES} attempts: {last_error}")

    # -- series / events --------------------------------------------------

    def get_series_list(self, category: str = None) -> list:
        params = {}
        if category:
            params["category"] = category
        data = self._get("/series", params=params)
        if data.get("__error__"):
            raise KalshiAPIError(f"GET /series failed: {data}")
        return data.get("series", [])

    def get_events(
        self,
        series_ticker: str = None,
        status: str = None,
        with_nested_markets: bool = True,
        min_close_ts: int = None,
        max_pages: int = None,
    ):
        """Yields event dicts, paginating via cursor until exhausted."""
        cursor = None
        pages = 0
        while True:
            params = {"limit": 200, "with_nested_markets": with_nested_markets}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if status:
                params["status"] = status
            if min_close_ts:
                params["min_close_ts"] = min_close_ts
            if cursor:
                params["cursor"] = cursor
            data = self._get("/events", params=params)
            if data.get("__error__"):
                raise KalshiAPIError(f"GET /events failed: {data}")
            for event in data.get("events", []):
                yield event
            cursor = data.get("cursor")
            pages += 1
            if not cursor or (max_pages and pages >= max_pages):
                break

    # -- single market / event lookups (for live polling) -----------------

    def get_market(self, ticker: str) -> dict:
        """Live snapshot of one market (yes_bid/yes_ask etc.) -- public, no auth needed."""
        data = self._get(f"/markets/{ticker}")
        if data.get("__error__"):
            raise KalshiAPIError(f"GET /markets/{ticker} failed: {data}")
        return data.get("market", {})

    def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> dict:
        """Single event by ticker, optionally with its markets nested -- public, no auth needed."""
        data = self._get(f"/events/{event_ticker}", params={"with_nested_markets": with_nested_markets})
        if data.get("__error__"):
            raise KalshiAPIError(f"GET /events/{event_ticker} failed: {data}")
        return data.get("event", {})

    # -- candlesticks -----------------------------------------------------

    def get_market_candlesticks(
        self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_interval: int = 1
    ) -> list:
        """
        Returns a list of raw candlestick dicts covering [start_ts, end_ts].
        Tries the live/recent endpoint first; on failure (market predates the
        historical cutoff) falls back to /historical/markets/{ticker}/candlesticks,
        which uses slightly different field names (see odds.py for the shim
        that normalizes both shapes).
        """
        data = self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        if not data.get("__error__") and data.get("candlesticks"):
            return data["candlesticks"]

        # Fall back to the historical/archived endpoint.
        hist = self._get(
            f"/historical/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        if hist.get("__error__"):
            return []
        return hist.get("candlesticks", [])


def _safe_json(resp):
    try:
        return resp.json()
    except ValueError:
        return resp.text


def to_unix_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def price_from_market(market: dict, field_prefix: str):
    """Reads e.g. yes_bid_dollars/yes_bid (or yes_ask_*, no_bid_*, no_ask_*)
    from a live GET /markets/{ticker} response, preferring the dollar-string
    field and falling back to the integer-cents field. Returns None if
    neither is present/parseable."""
    dollars = market.get(f"{field_prefix}_dollars")
    if dollars not in (None, ""):
        try:
            return float(dollars)
        except (TypeError, ValueError):
            pass
    cents = market.get(field_prefix)
    if cents not in (None, ""):
        try:
            return float(cents) / 100.0
        except (TypeError, ValueError):
            pass
    return None
