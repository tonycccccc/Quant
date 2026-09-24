"""
Step 1 — Fetch historical 30-min OHLCV bars from Alpaca (IEX feed).

Two entry points:
  fetch_bars(months_back)   — full backfill, chunked in 12-month windows
                              with exponential-backoff retry
  fetch_incremental()       — append-only: fetches from the last cached
                              bar's timestamp to now; ~30 seconds vs 15+ min

Cache: Models/raw_bars.parquet, MultiIndex (symbol, timestamp).

Robustness: large single-request fetches sometimes hit proxy timeouts
mid-pagination. Chunked mode splits into 12-month requests with retry;
incremental mode only ever fetches a small window.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, WATCHLIST, MODELS_DIR,
    ML_RAW_BARS_PATH, ML_HISTORY_MONTHS, ML_EXTRA_TRAINING_SYMBOLS,
)

# QQQ is used for relative-strength features; SPY for macro regime context
_INDEX_TICKERS = ['QQQ', 'SPY']

# Chunking / retry parameters (production-safe)
_CHUNK_MONTHS = 12         # months per Alpaca request — small enough to complete reliably
_MAX_RETRIES  = 4
_BASE_BACKOFF = 5.0        # seconds; doubles each retry


def _fetch_chunk(client, symbols, start, end, chunk_i, chunk_total):
    """Fetch one chunk with exponential-backoff retry on network errors."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            print(f'  [chunk {chunk_i}/{chunk_total}] {start.date()} -> {end.date()}  '
                  f'(attempt {attempt}/{_MAX_RETRIES})...')
            bars = client.get_stock_bars(request)
            return bars.df
        except Exception as e:
            last_err = e
            wait = _BASE_BACKOFF * (2 ** (attempt - 1))
            print(f'    ! error: {type(e).__name__}: {str(e)[:100]}  '
                  f'-> retrying in {wait:.1f}s')
            time.sleep(wait)
    raise RuntimeError(f'[collect] Chunk {chunk_i} failed after {_MAX_RETRIES} attempts: {last_err}')


def fetch_bars(months_back: int = ML_HISTORY_MONTHS, force: bool = False):
    """
    Fetch 30-min bars for all watchlist stocks + ML_EXTRA_TRAINING_SYMBOLS + QQQ/SPY.

    Uses 12-month chunked pagination with exponential-backoff retry so a single
    proxy hiccup doesn't abort a 60-month fetch. Chunks are concat'd + deduped
    on (symbol, timestamp) before write.

    Returns a pd.DataFrame with MultiIndex (symbol, timestamp) and
    columns [open, high, low, close, volume].

    If raw_bars.parquet already exists and force=False, loads from disk.
    """
    import pandas as pd
    from alpaca.data.historical import StockHistoricalDataClient

    if not force and ML_RAW_BARS_PATH.exists():
        print(f'[collect] Loading cached bars from {ML_RAW_BARS_PATH}')
        return pd.read_parquet(ML_RAW_BARS_PATH)

    end   = datetime.now()
    start = end - timedelta(days=months_back * 31)   # 31 days/month buffer

    # Deduplicate in case of overlap between watchlist and extra symbols
    all_trade_symbols = list(WATCHLIST.keys())
    extra = [s for s in ML_EXTRA_TRAINING_SYMBOLS if s not in all_trade_symbols]
    symbols = all_trade_symbols + extra + _INDEX_TICKERS
    print(f'[collect] Fetching {months_back}mo of 30-min bars for {len(symbols)} symbols '
          f'({len(all_trade_symbols)} watchlist + {len(extra)} training-only + {len(_INDEX_TICKERS)} indexes)')

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # Split into 12-month chunks, oldest first
    chunk_starts = []
    cursor = end
    while cursor > start:
        chunk_start = cursor - timedelta(days=_CHUNK_MONTHS * 31)
        if chunk_start < start:
            chunk_start = start
        chunk_starts.append((chunk_start, cursor))
        cursor = chunk_start
    chunk_starts.reverse()   # oldest first for logging clarity

    print(f'[collect] Split into {len(chunk_starts)} chunk(s) of ~{_CHUNK_MONTHS}mo each')

    chunk_dfs = []
    for i, (chunk_start, chunk_end) in enumerate(chunk_starts, 1):
        df_chunk = _fetch_chunk(client, symbols, chunk_start, chunk_end, i, len(chunk_starts))
        if df_chunk is not None and len(df_chunk):
            chunk_dfs.append(df_chunk)

    if not chunk_dfs:
        raise RuntimeError('[collect] All chunks failed — no data fetched')

    # Concat + dedupe (chunks may overlap at boundaries)
    df = pd.concat(chunk_dfs)
    df = df[~df.index.duplicated(keep='first')]

    # Keep only OHLCV
    keep = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df   = df[keep]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ML_RAW_BARS_PATH)
    fetched = df.index.get_level_values(0).nunique()
    print(f'[collect] Saved {len(df):,} bars across {fetched} symbols -> {ML_RAW_BARS_PATH}')
    return df


def fetch_recent(days_back: int = 90):
    """Live fetch of the full universe, uncached. Same shape as raw_bars.parquet."""
    from alpaca.data.historical import StockHistoricalDataClient

    all_trade_symbols = list(WATCHLIST.keys())
    extra = [s for s in ML_EXTRA_TRAINING_SYMBOLS if s not in all_trade_symbols]
    symbols = all_trade_symbols + extra + _INDEX_TICKERS
    end = datetime.now()
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    bars = _fetch_chunk(client, symbols, end - timedelta(days=days_back), end, 1, 1)
    return bars[[c for c in ['open', 'high', 'low', 'close', 'volume'] if c in bars.columns]]


def fetch_incremental(lookback_days: int = 2):
    """
    Incremental fetch: appends bars from (last_cached_ts - lookback_days) to now.

    Designed for daily update — runs in ~30 seconds. Only fetches the small
    window of new bars, then dedupes into the existing raw_bars.parquet.

    Parameters
    ----------
    lookback_days : how many days of overlap to fetch before the last cached
                    timestamp. Overlap protects against timezone edge cases
                    and any partial/missing bars from the previous fetch.
                    Default 2 days is safe.

    Returns the updated raw_bars DataFrame. If no cache exists, falls back
    to a full 60-month fetch.
    """
    import pandas as pd
    from alpaca.data.historical import StockHistoricalDataClient

    if not ML_RAW_BARS_PATH.exists():
        print('[collect-incremental] No cache found — running full 60mo backfill')
        return fetch_bars(months_back=60, force=True)

    existing = pd.read_parquet(ML_RAW_BARS_PATH)
    last_ts = existing.index.get_level_values(1).max()
    end     = datetime.now()

    # Convert last_ts to naive UTC datetime for arithmetic
    if hasattr(last_ts, 'tz_convert'):
        last_ts_utc = last_ts.tz_convert('UTC').tz_localize(None)
    else:
        last_ts_utc = last_ts

    # Overlap the last few days to catch any missed bars
    start = last_ts_utc - timedelta(days=lookback_days)

    age_days = (end - last_ts_utc).days
    if age_days < 0:
        print(f'[collect-incremental] Cache is already up to date (last bar: {last_ts})')
        return existing
    print(f'[collect-incremental] Cache last bar: {last_ts}  '
          f'({age_days} days stale). Fetching from {start.date()} to {end.date()}...')

    all_trade_symbols = list(WATCHLIST.keys())
    extra = [s for s in ML_EXTRA_TRAINING_SYMBOLS if s not in all_trade_symbols]
    symbols = all_trade_symbols + extra + _INDEX_TICKERS

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    try:
        new_bars = _fetch_chunk(client, symbols, start, end, 1, 1)
    except Exception as e:
        print(f'[collect-incremental] Fetch failed: {e}')
        return existing

    if new_bars is None or len(new_bars) == 0:
        print('[collect-incremental] No new bars returned')
        return existing

    keep = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in new_bars.columns]
    new_bars = new_bars[keep]

    # Concat + dedupe on (symbol, timestamp) — keep newest values in case of revisions
    combined = pd.concat([existing, new_bars])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    added = len(combined) - len(existing)
    combined.to_parquet(ML_RAW_BARS_PATH)
    new_last = combined.index.get_level_values(1).max()
    print(f'[collect-incremental] Added {added:,} new bars  |  '
          f'total {len(combined):,}  |  new last: {new_last}')
    return combined
