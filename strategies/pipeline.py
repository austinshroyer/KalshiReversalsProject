"""
End-to-end pipeline:

  1. Discover settled, head-to-head (exactly two competitors), non-tie sports
     games from Kalshi.
  2. Pull each game's odds history as a candlestick timeline.
  3. Detect significant favorite reversals in that timeline.
  4. Persist the game summary + reversal log to SQLite.

Filtering rules for step 1 (see README.md "Assumptions" for the full
reasoning):
  - series must be in the "Sports" category (GET /series?category=Sports)
  - event must be settled
  - event must have exactly two markets attached to it. Kalshi structures a
    plain two-team matchup as one market per team (whoever wins settles
    'yes'); a three-outcome event (e.g. soccer win/lose/draw) has three
    markets, so requiring exactly two markets naturally restricts us to
    two-party games and excludes draw-capable formats.
  - both markets must have actually settled to 'yes'/'no' (not void/scalar),
    so we skip anything that was cancelled, postponed, or voided.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import config
import db
from kalshi_client import KalshiClient
from odds import candles_to_timeline, find_reversals

logger = logging.getLogger("kalshi_reversals.pipeline")


def _iso(ts_or_dt) -> Optional[str]:
    if ts_or_dt is None:
        return None
    if isinstance(ts_or_dt, (int, float)):
        return datetime.fromtimestamp(ts_or_dt, tz=timezone.utc).isoformat()
    return str(ts_or_dt)


def _parse_iso_to_ts(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return None


def discover_series_tickers(client: KalshiClient, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        return explicit
    series = client.get_series_list(category=config.SERIES_CATEGORY)
    return [s["ticker"] for s in series]


def iter_eligible_events(
    client: KalshiClient, series_tickers: Iterable[str], min_close_ts: Optional[int] = None
):
    """Yields (event_dict, team_a_market, team_b_market) for settled,
    exactly-two-market, both-resolved-yes-or-no events."""
    for series_ticker in series_tickers:
        try:
            events = client.get_events(
                series_ticker=series_ticker,
                status="settled",
                with_nested_markets=True,
                min_close_ts=min_close_ts,
            )
            for event in events:
                markets = event.get("markets") or []
                binary_markets = [m for m in markets if m.get("market_type") == "binary"]
                if len(binary_markets) != 2:
                    continue  # not a plain two-competitor matchup (or nested markets missing)

                m_a, m_b = binary_markets[0], binary_markets[1]
                if m_a.get("result") not in ("yes", "no") or m_b.get("result") not in ("yes", "no"):
                    continue  # voided / undetermined / scalar -- skip

                # A real two-team result should have exactly one winner
                # between the two complementary markets.
                a_won = m_a.get("result") == "yes"
                b_won = m_b.get("result") == "yes"
                if a_won == b_won:
                    continue  # both/neither settled yes: inconsistent, skip defensively

                yield event, m_a, m_b
        except Exception:
            logger.exception("Failed to list events for series %s; skipping series", series_ticker)
            continue


def process_event(
    client: KalshiClient, event: dict, m_a: dict, m_b: dict, store_timeline: bool = True
) -> Optional[dict]:
    game_id = event["event_ticker"]
    series_ticker = event["series_ticker"]

    open_ts = _parse_iso_to_ts(m_a.get("open_time")) or _parse_iso_to_ts(m_b.get("open_time"))
    close_ts = _parse_iso_to_ts(m_a.get("close_time")) or _parse_iso_to_ts(m_b.get("close_time"))
    if open_ts is None or close_ts is None or close_ts <= open_ts:
        logger.warning("Skipping %s: missing/invalid open or close time", game_id)
        return None

    candles = client.get_market_candlesticks(
        series_ticker=series_ticker,
        ticker=m_a["ticker"],
        start_ts=open_ts,
        end_ts=close_ts,
        period_interval=config.CANDLESTICK_PERIOD_INTERVAL_MINUTES,
    )
    timeline = candles_to_timeline(candles)
    if not timeline:
        logger.warning("Skipping %s: no usable odds history for %s", game_id, m_a["ticker"])
        return None

    reversals, pregame_favorite, pregame_favorite_prob = find_reversals(timeline)
    if pregame_favorite is None:
        logger.warning("Skipping %s: could not establish a pregame favorite (all 50/50)", game_id)
        return None

    team_a_name = m_a.get("yes_sub_title") or m_a.get("ticker")
    team_b_name = m_b.get("yes_sub_title") or m_b.get("ticker")
    names = {"A": team_a_name, "B": team_b_name}

    final_winner = "A" if m_a.get("result") == "yes" else "B"

    game_row = {
        "game_id": game_id,
        "series_ticker": series_ticker,
        "title": event.get("title"),
        "team_a_ticker": m_a["ticker"],
        "team_a_name": team_a_name,
        "team_b_ticker": m_b["ticker"],
        "team_b_name": team_b_name,
        "open_time": _iso(open_ts),
        "close_time": _iso(close_ts),
        "pregame_favorite": pregame_favorite,
        "pregame_favorite_name": names[pregame_favorite],
        "pregame_favorite_prob": pregame_favorite_prob,
        "final_winner": final_winner,
        "final_winner_name": names[final_winner],
        "significant_reversal_count": len(reversals),
        "candlestick_points_used": len(timeline),
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    reversal_rows = [
        {
            "game_id": game_id,
            "sequence_num": r.sequence_num,
            "occurred_at": _iso(r.timestamp),
            "previous_favorite": r.previous_favorite,
            "new_favorite": r.new_favorite,
            "new_favorite_name": names[r.new_favorite],
            "new_favorite_prob_before": r.new_favorite_prob_before,
            "new_favorite_prob_after": r.new_favorite_prob_after,
            "swing_magnitude": r.swing_magnitude,
        }
        for r in reversals
    ]

    timeline_rows = []
    if store_timeline:
        duration = close_ts - open_ts
        timeline_rows = [
            {
                "game_id": game_id,
                "ts": ts,
                "elapsed_frac": min(1.0, max(0.0, (ts - open_ts) / duration)) if duration else 0.0,
                "prob_a": p_a,
            }
            for ts, p_a in timeline
        ]

    return {"game": game_row, "reversals": reversal_rows, "timeline": timeline_rows}


def run(
    db_path: str,
    series_tickers: Optional[List[str]] = None,
    min_close_ts: Optional[int] = None,
    limit: Optional[int] = None,
    skip_existing: bool = True,
    request_pause_seconds: float = 0.0,
    store_timeline: bool = True,
):
    db.init_db(db_path)
    client = KalshiClient()

    resolved_series = discover_series_tickers(client, series_tickers)
    logger.info("Scanning %d sports series for eligible games", len(resolved_series))

    processed = 0
    with db.connect(db_path) as conn:
        for event, m_a, m_b in iter_eligible_events(client, resolved_series, min_close_ts=min_close_ts):
            game_id = event["event_ticker"]
            if skip_existing and db.game_already_stored(conn, game_id):
                continue

            result = process_event(client, event, m_a, m_b, store_timeline=store_timeline)
            if result is None:
                continue

            db.upsert_game(conn, result["game"])
            db.replace_reversals(conn, game_id, result["reversals"])
            if store_timeline:
                db.replace_timeline(conn, game_id, result["timeline"])
            conn.commit()

            processed += 1
            logger.info(
                "Stored %s (%s vs %s): %d significant reversal(s), pregame favorite %s",
                game_id,
                result["game"]["team_a_name"],
                result["game"]["team_b_name"],
                result["game"]["significant_reversal_count"],
                result["game"]["pregame_favorite_name"],
            )

            if request_pause_seconds:
                time.sleep(request_pause_seconds)
            if limit and processed >= limit:
                break

    logger.info("Done. Processed %d game(s).", processed)
    return processed
