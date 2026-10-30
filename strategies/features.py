"""
Data assembly for the predictive models in count_model.py, timing_model.py,
and buy_strategy.py. This module doesn't train anything -- it just loads
the shared reversals.db tables and shapes them into the feature frames each
model needs, so the three scripts don't duplicate this logic.
"""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd


def load_tables(db_path: str):
    """Returns (games, reversals, timeline) DataFrames. `timeline` is empty
    (with the right columns) if odds_timeline hasn't been populated yet --
    see main.py's --no-timeline flag / README for how to backfill it."""
    conn = sqlite3.connect(db_path)
    try:
        games = pd.read_sql_query("SELECT * FROM games", conn)
        reversals = pd.read_sql_query("SELECT * FROM reversals", conn)
        try:
            timeline = pd.read_sql_query("SELECT * FROM odds_timeline", conn)
        except pd.errors.DatabaseError:
            timeline = pd.DataFrame(columns=["game_id", "ts", "elapsed_frac", "prob_a"])
    finally:
        conn.close()
    return games, reversals, timeline


def build_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game, using ONLY information known at/before market open
    (pregame_favorite_prob, series_ticker). This is what makes the count
    model in count_model.py legitimately "predictive" rather than a
    description of games that already happened -- every feature here would
    have been available before the game started."""
    df = games.copy()
    df = df[df["pregame_favorite_prob"].notna()].copy()
    df["favorite_strength"] = (df["pregame_favorite_prob"] - 0.5).abs()
    df["series_ticker"] = df["series_ticker"].fillna("UNKNOWN")
    return df


def _to_unix(iso_str) -> int:
    return int(datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).timestamp())


def _reversal_elapsed_fracs(grev: pd.DataFrame, gtime: pd.DataFrame):
    """Maps each reversal's occurred_at timestamp onto the elapsed_frac
    already computed for that game's timeline (nearest match if the exact
    candle timestamp isn't present, e.g. due to a resolution change)."""
    ts_to_frac = dict(zip(gtime["ts"], gtime["elapsed_frac"]))
    fracs = []
    for occurred_at in grev["occurred_at"]:
        ts = _to_unix(occurred_at)
        frac = ts_to_frac.get(ts)
        if frac is None:
            idx = (gtime["ts"] - ts).abs().idxmin()
            frac = gtime.loc[idx, "elapsed_frac"]
        fracs.append(float(frac))
    return sorted(fracs)


def build_hazard_bins(
    games: pd.DataFrame, reversals: pd.DataFrame, timeline: pd.DataFrame, n_bins: int = 20
) -> pd.DataFrame:
    """
    Discretizes every game's [0,1] elapsed-time axis into n_bins buckets and
    returns one row per (game, bin) with:
      - label_reversal_in_bin: did a significant reversal cross during this bin
      - prob_a_at_start / favorite_strength_at_start: state as of the START
        of the bin only (no lookahead into the bin's own outcome)
      - reversals_so_far: count of significant reversals strictly before
        this bin

    This is the dataset the discrete-time hazard model in timing_model.py
    trains on. Requires odds_timeline to be populated.
    """
    if timeline.empty:
        raise ValueError(
            "odds_timeline is empty -- re-run main.py without --no-timeline "
            "(add --reprocess to backfill games already in the database)."
        )

    rows = []
    reversals_by_game = {gid: sub.sort_values("occurred_at") for gid, sub in reversals.groupby("game_id")}
    # Group once up front instead of re-filtering the (potentially huge)
    # timeline table inside the per-game loop below -- filtering it per-game
    # is O(n_games * n_timeline_rows) and was the actual cause of very long
    # runtimes on real-sized databases.
    timeline_by_game = {gid: sub.sort_values("ts") for gid, sub in timeline.groupby("game_id")}
    bin_edges = np.array([i / n_bins for i in range(n_bins + 1)])
    bin_starts = bin_edges[:-1]

    for game in games.itertuples():
        gid = game.game_id
        gtime = timeline_by_game.get(gid)
        if gtime is None or gtime.empty:
            continue
        elapsed = gtime["elapsed_frac"].to_numpy()
        prob_a_vals = gtime["prob_a"].to_numpy()

        grev = reversals_by_game.get(gid)
        rev_fracs = np.array(
            _reversal_elapsed_fracs(grev, gtime) if grev is not None and len(grev) else []
        )

        # For each bin start, the index of the last timeline point at/before
        # it (i.e. "most recent known state as of the start of this bin").
        idxs = np.clip(np.searchsorted(elapsed, bin_starts, side="right") - 1, 0, len(elapsed) - 1)

        reversals_so_far = 0
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_bin = rev_fracs[(rev_fracs >= lo) & (rev_fracs < hi)] if len(rev_fracs) else rev_fracs
            label = 1 if len(in_bin) else 0
            prob_a_at_start = float(prob_a_vals[idxs[b]])

            rows.append(
                {
                    "game_id": gid,
                    "series_ticker": game.series_ticker,
                    "bin_index": b,
                    "bin_start": lo,
                    "prob_a_at_start": prob_a_at_start,
                    "favorite_strength_at_start": abs(prob_a_at_start - 0.5),
                    "reversals_so_far": reversals_so_far,
                    "label_reversal_in_bin": label,
                }
            )
            reversals_so_far += len(in_bin)

    return pd.DataFrame(rows)
