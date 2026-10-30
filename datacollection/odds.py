"""
Turns raw Kalshi candlesticks into a clean (timestamp, implied_probability)
timeline for a single market, and finds "significant" favorite reversals in
that timeline.

Two candlestick shapes exist in Kalshi's API and both are handled here:
  - live/recent candles:  yes_bid.close_dollars / yes_ask.close_dollars / price.close_dollars
  - historical (archived) candles: yes_bid.close / yes_ask.close / price.close  (no _dollars suffix)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import config


def _f(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_price(candle: dict) -> Optional[float]:
    """Best-effort implied probability (0-1) for one candlestick, trying
    bid/ask midpoint first, then last trade, then mean trade price."""
    yes_bid = candle.get("yes_bid") or {}
    yes_ask = candle.get("yes_ask") or {}
    price = candle.get("price") or {}

    bid_close = _f(yes_bid.get("close_dollars", yes_bid.get("close")))
    ask_close = _f(yes_ask.get("close_dollars", yes_ask.get("close")))
    if bid_close is not None and ask_close is not None:
        return (bid_close + ask_close) / 2.0
    if bid_close is not None:
        return bid_close
    if ask_close is not None:
        return ask_close

    trade_close = _f(price.get("close_dollars", price.get("close")))
    if trade_close is not None:
        return trade_close

    trade_mean = _f(price.get("mean_dollars", price.get("mean")))
    if trade_mean is not None:
        return trade_mean

    return None


def candles_to_timeline(candles: List[dict]) -> List[Tuple[int, float]]:
    """Returns sorted [(end_period_ts, implied_probability_0_to_1), ...],
    dropping candlesticks with no usable price data at all."""
    out = []
    for c in candles:
        p = _extract_price(c)
        if p is None:
            continue
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        out.append((int(ts), max(0.0, min(1.0, p))))
    out.sort(key=lambda t: t[0])
    return out


@dataclass
class Reversal:
    sequence_num: int
    timestamp: int
    previous_favorite: str  # 'A' or 'B'
    new_favorite: str       # 'A' or 'B'
    new_favorite_prob_before: float  # new favorite's prob at their weakest point of this run
    new_favorite_prob_after: float   # new favorite's prob at the moment they overtook
    swing_magnitude: float           # new_favorite_prob_after - new_favorite_prob_before


def find_reversals(
    timeline: List[Tuple[int, float]], threshold: float = None
) -> Tuple[List[Reversal], Optional[str], Optional[float]]:
    """
    Walks a team-A implied-probability timeline (values > 0.5 favor A, < 0.5
    favor B) and returns:
      - the list of *significant* reversals (favorite changes where the new
        favorite's win probability climbed at least `threshold` from its low
        point as the underdog to the moment it took over as favorite)
      - the pregame favorite ('A' or 'B'), i.e. the favorite at the first
        data point in the timeline
      - the pregame favorite's implied probability at that first data point

    Every crossing of the 50% line is a change of favorite; only crossings
    that clear `threshold` are counted/logged as "significant" per this
    project's definition (see config.SIGNIFICANT_SWING_THRESHOLD).
    """
    threshold = config.SIGNIFICANT_SWING_THRESHOLD if threshold is None else threshold
    if not timeline:
        return [], None, None

    def side_of(p_a: float) -> Optional[str]:
        if p_a > 0.5:
            return "A"
        if p_a < 0.5:
            return "B"
        return None  # exactly 0.5: no favorite yet, skip until it breaks a tie

    reversals: List[Reversal] = []

    # Establish the pregame favorite from the first data point that isn't
    # exactly a 50/50 tie.
    start_idx = 0
    while start_idx < len(timeline) and side_of(timeline[start_idx][1]) is None:
        start_idx += 1
    if start_idx >= len(timeline):
        return [], None, None

    pregame_favorite = side_of(timeline[start_idx][1])
    pregame_favorite_prob = timeline[start_idx][1] if pregame_favorite == "A" else 1 - timeline[start_idx][1]

    current_favorite = pregame_favorite
    extreme_p_a = timeline[start_idx][1]  # tracks the most-extreme P(A) seen during the current run
    seq = 0

    for i in range(start_idx + 1, len(timeline)):
        ts, p_a = timeline[i]
        side = side_of(p_a)
        if side is None:
            continue  # exact 0.5, ambiguous instant, wait for the next real reading

        if side == current_favorite:
            if side == "A":
                extreme_p_a = max(extreme_p_a, p_a)
            else:
                extreme_p_a = min(extreme_p_a, p_a)
            continue

        # Favorite flipped from current_favorite -> side.
        if side == "A":
            new_fav_before = extreme_p_a          # A's prob at B's peak dominance (A's low point)
            new_fav_after = p_a                   # A's prob right after taking over
        else:
            new_fav_before = 1 - extreme_p_a       # B's prob at A's peak dominance (B's low point)
            new_fav_after = 1 - p_a                # B's prob right after taking over

        swing = new_fav_after - new_fav_before
        if swing >= threshold:
            seq += 1
            reversals.append(
                Reversal(
                    sequence_num=seq,
                    timestamp=ts,
                    previous_favorite=current_favorite,
                    new_favorite=side,
                    new_favorite_prob_before=round(new_fav_before, 6),
                    new_favorite_prob_after=round(new_fav_after, 6),
                    swing_magnitude=round(swing, 6),
                )
            )

        # Whether or not it cleared the significance bar, the favorite has
        # changed -- start tracking a new run from this point.
        current_favorite = side
        extreme_p_a = p_a

    return reversals, pregame_favorite, pregame_favorite_prob
