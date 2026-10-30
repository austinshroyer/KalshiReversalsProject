import random
from datetime import datetime, timedelta, timezone

import db
import odds

random.seed(7)

DB_PATH = "test_synth.db"
N_GAMES = 150
SERIES = ["KXNFLGAME", "KXNBAGAME", "KXNHLGAME"]


def simulate_game(game_idx):
    series = random.choice(SERIES)
    duration_min = {"KXNFLGAME": 195, "KXNBAGAME": 150, "KXNHLGAME": 165}[series]
    open_dt = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=game_idx)
    close_dt = open_dt + timedelta(minutes=duration_min)
    open_ts = int(open_dt.timestamp())
    close_ts = int(close_dt.timestamp())

    # pregame favorite strength varies
    p = min(0.95, max(0.05, random.gauss(0.62, 0.15)))
    timeline = [(open_ts, p)]
    # random walk with occasional bigger jumps (to create real reversals)
    for t in range(open_ts + 60, close_ts, 60):
        drift = random.gauss(0, 0.02)
        if random.random() < 0.01:
            drift += random.choice([-1, 1]) * random.uniform(0.15, 0.4)
        p = min(0.98, max(0.02, p + drift))
        timeline.append((t, p))

    final_winner = "A" if random.random() < timeline[-1][1] else "B"

    reversals, pregame_favorite, pregame_favorite_prob = odds.find_reversals(timeline)

    game_id = f"SYNTH-{game_idx:04d}"
    team_a_name, team_b_name = f"TeamA{game_idx}", f"TeamB{game_idx}"
    names = {"A": team_a_name, "B": team_b_name}

    game_row = {
        "game_id": game_id,
        "series_ticker": series,
        "title": f"{team_a_name} vs {team_b_name}",
        "team_a_ticker": f"{game_id}-A",
        "team_a_name": team_a_name,
        "team_b_ticker": f"{game_id}-B",
        "team_b_name": team_b_name,
        "open_time": open_dt.isoformat(),
        "close_time": close_dt.isoformat(),
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
            "occurred_at": datetime.fromtimestamp(r.timestamp, tz=timezone.utc).isoformat(),
            "previous_favorite": r.previous_favorite,
            "new_favorite": r.new_favorite,
            "new_favorite_name": names[r.new_favorite],
            "new_favorite_prob_before": r.new_favorite_prob_before,
            "new_favorite_prob_after": r.new_favorite_prob_after,
            "swing_magnitude": r.swing_magnitude,
        }
        for r in reversals
    ]

    duration = close_ts - open_ts
    timeline_rows = [
        {"game_id": game_id, "ts": ts, "elapsed_frac": (ts - open_ts) / duration, "prob_a": p_a}
        for ts, p_a in timeline
    ]

    return game_row, reversal_rows, timeline_rows


db.init_db(DB_PATH)
with db.connect(DB_PATH) as conn:
    for i in range(N_GAMES):
        g, revs, tl = simulate_game(i)
        db.upsert_game(conn, g)
        db.replace_reversals(conn, g["game_id"], revs)
        db.replace_timeline(conn, g["game_id"], tl)
    conn.commit()

print(f"Built {DB_PATH} with {N_GAMES} synthetic games.")
