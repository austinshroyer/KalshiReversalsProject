# Kalshi Head-to-Head Reversal Tracker

Scans Kalshi's public sports markets for **settled, two-competitor games**
(no ties/draws possible), reconstructs each game's odds timeline from
candlestick data, detects **significant favorite reversals**, and stores
everything in a local SQLite database.

No API key is required — every endpoint used here (series, events, market
candlesticks, historical candlesticks) is public Kalshi market data.

## Setup

```bash
pip install -r requirements.txt
python main.py --db reversals.db --limit 5 -v   # smoke test on a handful of games
python main.py --db reversals.db                # full run over all Sports-category series
```

Useful flags:

```bash
# Restrict to specific series (comma-separated tickers), e.g. NFL and NBA only
python main.py --db reversals.db --series KXNFLGAME,KXNBAGAME

# Only games that closed on/after a given date
python main.py --db reversals.db --since 2026-01-01

# Re-fetch games already in the DB instead of skipping them
python main.py --db reversals.db --reprocess

# Be gentle on rate limits
python main.py --db reversals.db --pause 0.25
```

Run `python main.py --help` for the full list.

## What gets stored

`games` table — one row per qualifying game:

| column | meaning |
|---|---|
| `game_id` | Kalshi event ticker |
| `team_a_ticker`/`team_a_name`, `team_b_ticker`/`team_b_name` | the two competitors |
| `open_time` / `close_time` | market open (used as the "pregame" snapshot) / close (settlement) |
| `pregame_favorite` / `pregame_favorite_name` / `pregame_favorite_prob` | who was favored at market open, and their implied win probability |
| `final_winner` / `final_winner_name` | who actually won |
| `significant_reversal_count` | how many *significant* reversals occurred (see below) |
| `candlestick_points_used` | size of the odds timeline that was analyzed (for sanity-checking data quality) |

`reversals` table — one row per significant reversal, linked to `games.game_id`:

| column | meaning |
|---|---|
| `sequence_num` | 1st, 2nd, ... reversal within the game |
| `occurred_at` | timestamp of the candlestick where the favorite flipped |
| `previous_favorite` / `new_favorite` | `A` or `B` |
| `new_favorite_prob_before` | the new favorite's win probability at their lowest point while still the underdog |
| `new_favorite_prob_after` | the new favorite's win probability right after taking over |
| `swing_magnitude` | `new_favorite_prob_after - new_favorite_prob_before` (always ≥ the significance threshold) |

## Assumptions made (the prompt left these underspecified)

The request had some genuinely ambiguous points. Here's exactly how this
project resolves each one — every knob below lives in `config.py` so you can
change the behavior without touching the rest of the code:

1. **"Only two parties playing against each other, excluding all ties."**
   Interpreted as: (a) restrict to games structured as a plain two-competitor
   matchup, and (b) exclude any event whose format allows a draw. Kalshi
   represents a two-team matchup as exactly two binary markets (one per
   team, mutually exclusive); a draw-capable event (e.g. soccer) has a third
   market for the draw outcome. Requiring **exactly two markets on the
   event** satisfies both (a) and (b) simultaneously. Events where the two
   markets didn't settle to a clean one-winner outcome (voided, postponed,
   scalar, etc.) are also skipped.

2. **"Significant reversal ... favorite shifts, and the new favorite's odds
   increase by at least 20%."** This is ambiguous between "20 percentage
   points" and "a 20% relative change." This project uses **20 percentage
   points** (`config.SIGNIFICANT_SWING_THRESHOLD = 0.20`), measured from the
   new favorite's low point as the underdog to the moment they overtake —
   i.e., the size of the swing that caused the flip, not just the size of
   one candlestick's jump. Change that one constant if you want a different
   threshold or a relative-percentage definition instead.

3. **"Pregame favorite."** Kalshi doesn't expose a separate "scheduled game
   start time" distinct from the market's own `open_time`/`close_time`.
   This project treats the market's `open_time` (when the market first
   opened for trading) as the pregame snapshot. If you have a more precise
   game-start time per sport, swap in that timestamp when calling
   `odds.find_reversals` — the function is agnostic to how you slice the
   timeline.

4. **Favorite determination.** Only team A's market is fetched (its implied
   win probability is `mid(yes_bid, yes_ask)` close price per candlestick,
   falling back to last-trade/mean-trade price). Team B's probability is
   assumed to be `1 - P(A)`, which is normally true by construction since the
   two markets are complementary outcomes of one mutually-exclusive event.

5. **Candlestick resolution.** 1-minute candles by default
   (`config.CANDLESTICK_PERIOD_INTERVAL_MINUTES`), the finest Kalshi offers.
   Coarser candles will smooth over some short-lived reversals.

## Project layout

- `config.py` — all tunables (API base URL, thresholds, resolution)
- `kalshi_client.py` — thin REST wrapper (series/events/candlesticks, with
  automatic fallback from the live candlestick endpoint to the historical
  one for older/archived markets)
- `odds.py` — candlestick → probability timeline, and the reversal-detection
  algorithm (see `find_reversals`, which has a runnable docstring-level
  explanation of the swing calculation)
- `db.py` — SQLite schema + upserts
- `pipeline.py` — discovery + filtering + orchestration
- `main.py` — CLI

## Predictive modeling

Three scripts, one per question, all reading the same `reversals.db`:

| Script | Question | Needs `odds_timeline`? |
|---|---|---|
| `count_model.py` | How many significant reversals will this game have? | No -- pregame info only |
| `timing_model.py` | When (as % of game elapsed) do reversals tend to happen? | Descriptive part: no. Hazard model: yes |
| `buy_strategy.py` | What's the historical EV of buying the underdog, by (time, price)? | Yes |

Run all three at once with `python train_all.py --db reversals.db`, or run each individually (each has its own `--help`).

### Why `odds_timeline` exists

`reversals` only contains events that already cleared the 20-point significance bar -- i.e., only the "winners." A model trained on that alone would never see the cases where an underdog dipped low and *never* came back, which would make any buy-timing strategy look far better than it really is (classic survivorship bias). `odds_timeline` stores the **full** per-game odds path (every candlestick, not just the significant-reversal points) specifically so `buy_strategy.py` can treat every historical moment as a candidate trade -- wins and losses both -- and compute a real, unbiased base rate.

This table is populated automatically by `main.py` going forward. **If you already have games in your database, back-fill them once with:**

```bash
python main.py --db reversals.db --reprocess
```

(This re-fetches candlesticks for every stored game, so it takes about as long as your original run did. `--no-timeline` opts back out if you'd rather save the disk space and only use `count_model.py`.)

### `count_model.py` -- reversal count

Poisson regression (the standard model for count data) trained on `pregame_favorite_prob` and `series_ticker` -- both known before the game starts, so this is a genuine prediction, not a description of history. It's evaluated with 5-fold cross-validation against a "just predict the historical average" baseline, and **it says so explicitly if it isn't beating that baseline** -- pregame favorite strength alone may just not carry much signal for reversal count in your data, and the script won't paper over that.

### `timing_model.py` -- when reversals happen

Two parts:
1. **Descriptive**: median/quartile elapsed-time-fraction of significant reversals, no odds_timeline needed.
2. **Hazard model**: bins each game into N equal time chunks (default 20 = 5% each) and trains logistic regression to predict P(a significant reversal crosses in this chunk | hasn't happened yet in this chunk, current favorite/probability/reversals-so-far). This is the piece you'd query mid-game: "we're 40% through, favorite's at 70% -- what's the chance of a reversal in the next 5%?"

### `buy_strategy.py` -- optimal buy timing

Treats every `odds_timeline` point as a candidate "buy the current underdog, hold to settlement" trade, buckets by (time-in-game, entry price), and reports average return and win rate per bucket -- an actual historical EV table, not a hindsight-only view of realized reversals.

**Read the caveats in the script's docstring before acting on this.** In particular: no fees/slippage/spread are modeled, fills are assumed at the exact recorded price, and this is backtested history, not a live edge. This is a data analysis tool, not financial advice.

### Testing without waiting on real data

`generate_synthetic_test_db.py` builds a small fake `reversals.db` (random-walk odds paths, realistic table shapes) so you can run all three scripts immediately and see what the output looks like before your real backfill finishes:

```bash
python generate_synthetic_test_db.py   # writes test_synth.db
python train_all.py --db test_synth.db
```

## Live paper trading

Backtests and their validators only ever look backward. `paper_trader.py` is the one thing in this project that looks forward: it watches currently-open Kalshi markets and, when a live game matches a bucket you found, "trades" it in real time -- either purely in your own database (`--mode simulate`, needs no Kalshi credentials at all) or as a real order on Kalshi's own demo/paper-money exchange (`--mode demo`).

### `--mode simulate` (no setup required)

```bash
python paper_trader.py --db reversals.db --series KXNFLGAME \
    --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \
    --mode simulate
```

This polls public market data only, computes the same `elapsed_frac`/`entry_price` your backtest used, and logs a hypothetical trade to a new `paper_trades` table (same `reversals.db`) whenever a live game enters your bucket. No account, no API key, nothing that touches Kalshi beyond reading public prices.

### `--mode demo` (real orders, fake money)

Places actual orders on Kalshi's sandbox exchange -- same order-matching and execution behavior as production, zero real financial risk. One-time setup:

```bash
python generate_kalshi_keys.py                 # writes kalshi_private.pem, prints the public key
# upload the printed public key at https://demo.kalshi.co/ -> Settings -> API
# Kalshi gives you back an API Key ID

export KALSHI_API_KEY_ID=<the id Kalshi gave you>
export KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private.pem
export KALSHI_BASE_URL=https://external-api.demo.kalshi.co/trade-api/v2
```

Then:

```bash
python paper_trader.py --db reversals.db --series KXNFLGAME \
    --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \
    --mode demo --contracts 1
```

`paper_trader.py` refuses to run in `--mode demo` unless `KALSHI_BASE_URL` contains `"demo"` -- a guard against accidentally pointing it at your real account. Demo and production credentials are entirely separate (separate account, separate key upload), so a demo key simply won't authenticate against production either way.

### Adding the cashout rule

Add `--take-profit` (and optionally `--max-hold-frac`) to trade the `cashout_strategy.py` policy instead of holding every position to settlement:

```bash
python paper_trader.py --db reversals.db --series KXNFLGAME \
    --time-min 0.4 --time-max 0.6 --price-min 0.10 --price-max 0.25 \
    --take-profit 0.55 --max-hold-frac 0.25 --mode simulate
```

### Reading the results

Every trade (simulated or real) lands in `paper_trades` -- `status` (`open`/`cashed_out`/`settled`), `entry_price`/`exit_price`, `return_pct`, and `dry_run` (1 = simulate, 0 = real demo order) so you can filter appropriately. Compare its average `return_pct` against what `buy_strategy.py`/`cashout_strategy.py` predicted for the same bucket -- that comparison, on games the backtest never saw, is the real test.

Run it with `nohup`/`screen`/a systemd unit/etc. if you want it running unattended across multiple games -- it polls every `--poll-interval` seconds (default 60) and picks back up open positions on restart as long as you point it at the same `--db`.

**This is a research and forward-testing tool, not investment advice**, and `--mode demo` still means Kalshi's *sandbox* -- don't point `KALSHI_BASE_URL` at production without independently deciding you want to risk real money on this.

## Limitations

- Thinly-traded markets can have gaps or stale quotes in their candlesticks;
  `odds.candles_to_timeline` silently skips candlesticks with no usable
  price at all, which can occasionally merge over a brief real reversal.
- Kalshi's public endpoints are rate-limited per IP; use `--pause` for large
  backfills if you hit 429s (the client already retries those with backoff).
- This was built and reviewed against Kalshi's documented API shapes as of
  September 2026, but wasn't run against live Kalshi traffic from this
  environment (network egress here doesn't allow reaching kalshi.com) — run
  the `--limit 5 -v` smoke test first on your machine before a full backfill.
