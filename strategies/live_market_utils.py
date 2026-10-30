"""
Shared helpers used by both paper_trader.py (entry discovery) and
position_monitor.py (exit checking): timing/pricing math and the --mode
demo credential setup. Kept in one place on purpose -- if these lived
separately in each script, the two processes could silently drift out of
sync with each other over time.
"""

import time
from datetime import datetime

import config
from kalshi_client import KalshiClient, price_from_market


def to_ts(iso_str):
    if not iso_str:
        return None
    try:
        return int(datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def compute_elapsed_frac(m_a: dict, m_b: dict):
    """Free -- uses open_time/close_time already present in the market data
    from get_events()/get_event(), no API call."""
    open_ts = to_ts(m_a.get("open_time")) or to_ts(m_b.get("open_time"))
    close_ts = to_ts(m_a.get("close_time")) or to_ts(m_b.get("close_time"))
    if not open_ts or not close_ts or close_ts <= open_ts:
        return None
    now_ts = int(time.time())
    return min(1.0, max(0.0, (now_ts - open_ts) / (close_ts - open_ts)))


def plausibly_started(m_a: dict, m_b: dict, max_hours_before_game: float) -> bool:
    """Free -- Kalshi doesn't expose a dedicated scheduled-kickoff-time
    field, but expected_expiration_time is real and documented: 'the time
    the event is likely to resolve (for a sports game, typically a few
    hours after the scheduled start)'. Heuristic, not an exact kickoff
    check; missing field -> don't exclude. See paper_trader.py's module
    docstring for the full rationale."""
    exp_ts = to_ts(m_a.get("expected_expiration_time")) or to_ts(m_b.get("expected_expiration_time"))
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


def build_trading_client(mode: str, logger):
    """Shared --mode demo guard + lazy credential loading, so paper_trader.py
    and position_monitor.py can't diverge on this. Returns None for
    --mode simulate (no credentials touched at all in that case)."""
    if mode != "demo":
        return None
    if "demo" not in config.KALSHI_BASE_URL:
        logger.error(
            "Refusing to run --mode demo: KALSHI_BASE_URL ('%s') doesn't look like a demo host. "
            "Set it to https://external-api.demo.kalshi.co/trade-api/v2 first.",
            config.KALSHI_BASE_URL,
        )
        raise SystemExit(1)
    from kalshi_trading_client import KalshiTradingClient  # lazy: --mode simulate needs no credentials

    return KalshiTradingClient()
