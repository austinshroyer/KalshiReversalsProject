"""SQLite storage for tracked games and their significant reversals."""

import sqlite3
from contextlib import contextmanager
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id                     TEXT PRIMARY KEY,   -- Kalshi event_ticker
    series_ticker               TEXT NOT NULL,
    title                       TEXT,
    team_a_ticker                TEXT NOT NULL,
    team_a_name                  TEXT,
    team_b_ticker                TEXT NOT NULL,
    team_b_name                  TEXT,
    open_time                   TEXT,               -- ISO8601, market open (pregame snapshot time)
    close_time                  TEXT,               -- ISO8601, market close (game/settlement time)
    pregame_favorite            TEXT,               -- 'A' or 'B'
    pregame_favorite_name       TEXT,
    pregame_favorite_prob       REAL,                -- 0-1 implied probability at open
    final_winner                TEXT,                -- 'A' or 'B'
    final_winner_name           TEXT,
    significant_reversal_count  INTEGER NOT NULL DEFAULT 0,
    candlestick_points_used     INTEGER,             -- size of the odds timeline analyzed
    fetched_at                  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reversals (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id                  TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    sequence_num             INTEGER NOT NULL,        -- 1st, 2nd, ... significant reversal in the game
    occurred_at              TEXT NOT NULL,           -- ISO8601 timestamp of the candlestick where it flipped
    previous_favorite        TEXT NOT NULL,           -- 'A' or 'B'
    new_favorite              TEXT NOT NULL,           -- 'A' or 'B'
    new_favorite_name         TEXT,
    new_favorite_prob_before  REAL NOT NULL,           -- new favorite's win prob at their low point as underdog
    new_favorite_prob_after   REAL NOT NULL,           -- new favorite's win prob right after taking over
    swing_magnitude           REAL NOT NULL            -- new_favorite_prob_after - new_favorite_prob_before
);

CREATE INDEX IF NOT EXISTS idx_reversals_game_id ON reversals(game_id);

-- Full odds path per game (not just the significant-reversal points).
-- Needed for any model that has to know the true base rate of an event
-- (e.g. "how often does the underdog actually come back from X%?"), since
-- `reversals` only contains the events that DID clear the significance bar
-- and would badly overstate any prediction built from it alone.
CREATE TABLE IF NOT EXISTS odds_timeline (
    game_id       TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    ts            INTEGER NOT NULL,   -- unix timestamp of the candlestick
    elapsed_frac  REAL NOT NULL,      -- 0.0 at market open, 1.0 at market close
    prob_a        REAL NOT NULL,      -- implied P(team A wins), 0-1
    PRIMARY KEY (game_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_timeline_game_id ON odds_timeline(game_id);

-- Live forward-test log: one row per simulated or real (demo-account)
-- paper trade placed by paper_trader.py, so you can compare how a bucket
-- performs on games the backtest never saw against what the backtest
-- predicted for it.
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id               TEXT NOT NULL,
    ticker                 TEXT NOT NULL,       -- the specific team-market ticker bought
    side                   TEXT NOT NULL,       -- 'A' or 'B'
    bucket_label            TEXT,                -- human-readable strategy/bucket description
    entry_time               TEXT NOT NULL,
    entry_elapsed_frac       REAL,
    entry_price               REAL NOT NULL,
    entry_order_id            TEXT,               -- Kalshi order id (NULL in --mode simulate)
    status                   TEXT NOT NULL DEFAULT 'open',  -- open | cashed_out | settled
    exit_time                 TEXT,
    exit_price                 REAL,
    exit_order_id               TEXT,
    exit_reason                  TEXT,             -- take_profit | max_hold | settlement
    return_pct                    REAL,
    dry_run                      INTEGER NOT NULL DEFAULT 1,  -- 1 = --mode simulate, 0 = --mode demo (real demo-account order)
    created_at                    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_game_id ON paper_trades(game_id);
"""


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    # WAL mode lets one writer and any number of readers proceed concurrently
    # instead of blocking each other -- needed now that paper_trader.py and
    # position_monitor.py are two independent processes writing to the same
    # file. busy_timeout makes SQLite wait and retry briefly on a transient
    # lock instead of raising immediately. Both are persistent, database-level
    # settings; safe (and cheap) to set on every connect() call.
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str):
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def game_already_stored(conn, game_id: str) -> bool:
    cur = conn.execute("SELECT 1 FROM games WHERE game_id = ?", (game_id,))
    return cur.fetchone() is not None


def upsert_game(conn, game_row: dict):
    conn.execute(
        """
        INSERT INTO games (
            game_id, series_ticker, title, team_a_ticker, team_a_name,
            team_b_ticker, team_b_name, open_time, close_time,
            pregame_favorite, pregame_favorite_name, pregame_favorite_prob,
            final_winner, final_winner_name, significant_reversal_count,
            candlestick_points_used, fetched_at
        ) VALUES (
            :game_id, :series_ticker, :title, :team_a_ticker, :team_a_name,
            :team_b_ticker, :team_b_name, :open_time, :close_time,
            :pregame_favorite, :pregame_favorite_name, :pregame_favorite_prob,
            :final_winner, :final_winner_name, :significant_reversal_count,
            :candlestick_points_used, :fetched_at
        )
        ON CONFLICT(game_id) DO UPDATE SET
            series_ticker=excluded.series_ticker,
            title=excluded.title,
            team_a_ticker=excluded.team_a_ticker,
            team_a_name=excluded.team_a_name,
            team_b_ticker=excluded.team_b_ticker,
            team_b_name=excluded.team_b_name,
            open_time=excluded.open_time,
            close_time=excluded.close_time,
            pregame_favorite=excluded.pregame_favorite,
            pregame_favorite_name=excluded.pregame_favorite_name,
            pregame_favorite_prob=excluded.pregame_favorite_prob,
            final_winner=excluded.final_winner,
            final_winner_name=excluded.final_winner_name,
            significant_reversal_count=excluded.significant_reversal_count,
            candlestick_points_used=excluded.candlestick_points_used,
            fetched_at=excluded.fetched_at
        """,
        game_row,
    )


def replace_reversals(conn, game_id: str, reversal_rows: Iterable[dict]):
    conn.execute("DELETE FROM reversals WHERE game_id = ?", (game_id,))
    conn.executemany(
        """
        INSERT INTO reversals (
            game_id, sequence_num, occurred_at, previous_favorite, new_favorite,
            new_favorite_name, new_favorite_prob_before, new_favorite_prob_after,
            swing_magnitude
        ) VALUES (
            :game_id, :sequence_num, :occurred_at, :previous_favorite, :new_favorite,
            :new_favorite_name, :new_favorite_prob_before, :new_favorite_prob_after,
            :swing_magnitude
        )
        """,
        list(reversal_rows),
    )


def replace_timeline(conn, game_id: str, timeline_rows: Iterable[dict]):
    conn.execute("DELETE FROM odds_timeline WHERE game_id = ?", (game_id,))
    conn.executemany(
        """
        INSERT INTO odds_timeline (game_id, ts, elapsed_frac, prob_a)
        VALUES (:game_id, :ts, :elapsed_frac, :prob_a)
        """,
        list(timeline_rows),
    )


def paper_trade_exists_for_game(conn, game_id: str) -> bool:
    """Any trade at all (open, cashed_out, or settled) exists for this game.
    Kept for simple existence checks; maybe_enter's blocking logic uses the
    richer game_trade_status() below, which distinguishes cashed_out
    (re-entry allowed) from open/settled (re-entry blocked)."""
    cur = conn.execute("SELECT 1 FROM paper_trades WHERE game_id = ? LIMIT 1", (game_id,))
    return cur.fetchone() is not None


def game_trade_status(conn, game_id: str) -> dict:
    """Used to decide whether re-entering a game is allowed: blocked while a
    position is currently open (no double-entry) or once the game has
    settled (it's over) -- but NOT blocked by a prior cashed_out close,
    since the game is still live and could re-enter the bucket later."""
    rows = conn.execute("SELECT status FROM paper_trades WHERE game_id = ?", (game_id,)).fetchall()
    statuses = [r["status"] for r in rows]
    return {
        "has_open": "open" in statuses,
        "has_settled": "settled" in statuses,
        "entry_count": len(statuses),
    }


def insert_paper_trade(conn, row: dict) -> int:
    cur = conn.execute(
        """
        INSERT INTO paper_trades (
            game_id, ticker, side, bucket_label, entry_time, entry_elapsed_frac,
            entry_price, entry_order_id, status, dry_run, created_at
        ) VALUES (
            :game_id, :ticker, :side, :bucket_label, :entry_time, :entry_elapsed_frac,
            :entry_price, :entry_order_id, :status, :dry_run, :created_at
        )
        """,
        row,
    )
    return cur.lastrowid


def get_open_paper_trades(conn):
    return conn.execute("SELECT * FROM paper_trades WHERE status = 'open'").fetchall()


def close_paper_trade(conn, trade_id: int, updates: dict):
    updates = dict(updates, trade_id=trade_id)
    conn.execute(
        """
        UPDATE paper_trades SET
            status = :status, exit_time = :exit_time, exit_price = :exit_price,
            exit_order_id = :exit_order_id, exit_reason = :exit_reason, return_pct = :return_pct
        WHERE trade_id = :trade_id
        """,
        updates,
    )
