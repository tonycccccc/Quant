"""
Macro / options-adjacent features from CBOE indices via yfinance.

Adds four features that reflect market-wide fear, complacency, and
options-implied risk pricing — signals institutional traders use every day.

Features:
  vix_9d          — 9-day expected S&P 500 volatility (CBOE ^VIX9D)
  vix_3m          — 3-month expected S&P 500 volatility (CBOE ^VIX3M)
  vix_term_ratio  — vix_9d / vix_3m
                    > 1.0 = backwardation = fear peak, often bottoms
                    < 1.0 = contango = complacency, often tops
  put_call_ratio  — CBOE equity put/call ratio (^CPCE) — sentiment gauge

All values are daily; they're aligned to intraday bars via shift(1) + ffill
so each intraday bar sees only the PREVIOUS trading day's macro data
(no lookahead).

Fetched once per day and cached to Models/macro_features.parquet so we
don't hammer yfinance on every training run.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR

# Cache path — cleared automatically if older than 24 hours
MACRO_CACHE_PATH = MODELS_DIR / 'macro_features.parquet'

# yfinance tickers (^CPCE 404s on yfinance — put/call fetched via stooq below)
_YF_TICKERS = {
    'vix_9d':          '^VIX9D',
    'vix':             '^VIX',       # already fetched elsewhere, useful for spot cross-check
    'vix_3m':          '^VIX3M',
}


def _fetch_yf_series(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch a single yfinance daily Close series over [start, end]."""
    import yfinance as yf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hist = yf.Ticker(ticker).history(
                start=start.strftime('%Y-%m-%d'),
                end=(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                auto_adjust=False,
            )
        if hist.empty or 'Close' not in hist.columns:
            return pd.Series(dtype=float, name=ticker)
        s = hist['Close'].copy()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        s.name = ticker
        return s
    except Exception as e:
        print(f'  [macro] fetch failed for {ticker}: {e}')
        return pd.Series(dtype=float, name=ticker)


def _fetch_put_call_from_stooq(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Fetch CBOE equity put/call ratio from stooq.com (free daily CSV).
    Fallback chain: stooq direct CSV -> pandas-datareader -> None.

    stooq URL pattern: https://stooq.com/q/d/?s=^cpce&d1=YYYYMMDD&d2=YYYYMMDD&i=d&f=csv
    Returns tz-naive daily Series indexed by date, name='put_call_ratio'.
    """
    import io
    import requests

    # Try stooq direct CSV — a few alternate URL patterns
    for ticker_url in ('%5Ecpce', 'cpce.us', '%5Ecpc'):
        url = (f'https://stooq.com/q/d/l/?s={ticker_url}'
               f'&d1={start.strftime("%Y%m%d")}'
               f'&d2={end.strftime("%Y%m%d")}'
               f'&i=d')
        try:
            r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty or 'Close' not in df.columns:
                continue
            s = df.set_index(pd.to_datetime(df['Date']))['Close']
            s.name = 'put_call_ratio'
            s.index.name = None
            print(f'  put_call_ratio  (stooq {ticker_url}): {len(s)} daily observations')
            return s
        except Exception:
            continue

    # Fallback: pandas-datareader
    try:
        import pandas_datareader.data as web
        s = web.DataReader('^CPCE', 'stooq', start, end)['Close']
        s.name = 'put_call_ratio'
        s.index = pd.DatetimeIndex(s.index).tz_localize(None) if s.index.tz else s.index
        print(f'  put_call_ratio  (pdr stooq): {len(s)} daily observations')
        return s
    except Exception as e:
        print(f'  [macro] pandas-datareader put/call failed: {e}')

    print('  [macro] all put/call sources exhausted — will default to 0.7')
    return pd.Series(dtype=float, name='put_call_ratio')


def fetch_macro_features(start: pd.Timestamp, end: pd.Timestamp,
                           force: bool = False) -> pd.DataFrame:
    """
    Fetch all macro daily series between start and end.

    Returns a DataFrame with columns:
      vix_9d, vix_3m, vix_term_ratio, put_call_ratio
    indexed by daily timestamps (tz-naive, ET-aligned dates).

    Caches to Models/macro_features.parquet; refreshes if older than 24h
    or if force=True.
    """
    # Cache freshness check
    if not force and MACRO_CACHE_PATH.exists():
        age_hours = (datetime.now().timestamp() - MACRO_CACHE_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            cached = pd.read_parquet(MACRO_CACHE_PATH)
            if len(cached) > 0 and cached.index.min() <= start and cached.index.max() >= end - pd.Timedelta(days=1):
                print(f'[macro] Loading cached macro features from {MACRO_CACHE_PATH} '
                      f'(age {age_hours:.1f}h)')
                return cached.loc[start:end].copy()

    print(f'[macro] Fetching CBOE indices ({start.date()} -> {end.date()})...')
    parts = {}
    # VIX suite from yfinance (works)
    for key, ticker in _YF_TICKERS.items():
        s = _fetch_yf_series(ticker, start, end)
        if len(s):
            parts[key] = s
            print(f'  {key:15s} ({ticker}): {len(s):4d} daily observations')
        else:
            print(f'  {key:15s} ({ticker}): FAILED — will fill with defaults')

    # Put/call ratio from stooq (yfinance ^CPCE 404s)
    pc_series = _fetch_put_call_from_stooq(start, end)
    if len(pc_series):
        parts['put_call_ratio'] = pc_series
    else:
        print(f'  {"put_call_ratio":15s} (stooq ^CPCE): FAILED — will fill with defaults')

    if not parts:
        print('[macro] All macro fetches failed — returning empty frame')
        return pd.DataFrame(columns=['vix_9d', 'vix_3m', 'vix_term_ratio', 'put_call_ratio'])

    df = pd.concat(parts, axis=1)

    # Compute term ratio (short VIX / long VIX)
    if 'vix_9d' in df.columns and 'vix_3m' in df.columns:
        df['vix_term_ratio'] = df['vix_9d'] / df['vix_3m'].replace(0, np.nan)
    else:
        df['vix_term_ratio'] = 1.0

    # Keep only feature columns
    keep = ['vix_9d', 'vix_3m', 'vix_term_ratio', 'put_call_ratio']
    for col in keep:
        if col not in df.columns:
            # Missing series — fill with neutral defaults so downstream doesn't break
            df[col] = {'vix_9d': 20.0, 'vix_3m': 22.0,
                       'vix_term_ratio': 1.0, 'put_call_ratio': 0.7}[col]
    df = df[keep].copy()

    # Forward-fill weekends/holidays then backfill any leading NaN
    df = df.sort_index().ffill().bfill()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MACRO_CACHE_PATH)
    print(f'[macro] Cached {len(df):,} daily rows -> {MACRO_CACHE_PATH}')
    return df


def align_to_intraday(macro_daily: pd.DataFrame,
                       intraday_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Broadcast daily macro values to a 30-min intraday index.

    Uses shift(1) so each intraday bar sees only the PREVIOUS trading day's
    macro close — no lookahead.

    Returns a DataFrame with the same columns as macro_daily, indexed by
    intraday_index. Missing values filled with neutral defaults.
    """
    if macro_daily is None or macro_daily.empty:
        return pd.DataFrame(index=intraday_index, columns=[
            'vix_9d', 'vix_3m', 'vix_term_ratio', 'put_call_ratio',
        ]).fillna({'vix_9d': 20.0, 'vix_3m': 22.0,
                    'vix_term_ratio': 1.0, 'put_call_ratio': 0.7})

    # Shift 1 day so intraday bars see YESTERDAY's macro close
    shifted = macro_daily.shift(1).ffill()

    # Reindex to intraday timestamps
    if intraday_index.tz is not None:
        # macro_daily is tz-naive; convert to same tz as intraday for merge_asof
        shifted.index = shifted.index.tz_localize(intraday_index.tz).tz_convert(intraday_index.tz)

    # Normalize both indexes to ns-precision datetime64 (merge_asof needs
    # identical dtypes; parquet/pandas can produce us or s precision).
    shifted.index = pd.DatetimeIndex(shifted.index).astype('datetime64[ns, UTC]') \
        if intraday_index.tz is None or str(intraday_index.tz) == 'UTC' \
        else pd.DatetimeIndex(shifted.index)

    # Use merge_asof for efficient temporal join
    idx_df = pd.DataFrame(index=intraday_index).reset_index()
    idx_df.columns = ['ts']
    # Cast to a common precision (nanoseconds) — merge_asof requires it
    idx_df['ts'] = pd.to_datetime(idx_df['ts'], utc=True).astype('datetime64[ns, UTC]')

    shifted_df = shifted.reset_index()
    shifted_df.columns = ['ts'] + list(shifted.columns)
    shifted_df['ts'] = pd.to_datetime(shifted_df['ts'], utc=True).astype('datetime64[ns, UTC]')

    merged = pd.merge_asof(
        idx_df.sort_values('ts'),
        shifted_df.sort_values('ts'),
        on='ts',
        direction='backward',
    )
    merged.set_index('ts', inplace=True)
    # Restore original tz on the index to match caller's expectation
    if intraday_index.tz is not None and str(intraday_index.tz) != 'UTC':
        merged.index = merged.index.tz_convert(intraday_index.tz)
    merged.index.name = intraday_index.name

    # Fill any remaining NaN with sensible defaults (e.g., start-of-series bars)
    merged = merged.fillna({
        'vix_9d':          20.0,
        'vix_3m':          22.0,
        'vix_term_ratio':  1.0,
        'put_call_ratio':  0.7,
    })
    return merged


def get_latest_macro() -> dict:
    """
    Return the most recent daily macro values as a dict for live inference.
    Fetches on demand if cache is stale.
    """
    end   = pd.Timestamp.now().tz_localize(None)
    start = end - pd.Timedelta(days=30)
    df    = fetch_macro_features(start, end)
    if df.empty:
        return {'vix_9d': 20.0, 'vix_3m': 22.0,
                'vix_term_ratio': 1.0, 'put_call_ratio': 0.7}
    # Use previous day's close (same shift(1) logic as align_to_intraday)
    if len(df) < 2:
        latest = df.iloc[-1]
    else:
        latest = df.iloc[-2]
    return {k: float(latest[k]) for k in
            ('vix_9d', 'vix_3m', 'vix_term_ratio', 'put_call_ratio')}
