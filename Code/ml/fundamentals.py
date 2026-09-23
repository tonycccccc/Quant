"""
Fundamentals module — yfinance-based company fundamentals for advisor context.

Purpose: give the advisor a "second opinion" alongside the technical signal.
Fundamentals are ADVISORY ONLY — they don't trigger trades, don't change
stops, don't override the technical filters. They INFORM decisions when
the technical signal already fires.

Metrics fetched (per ticker, cached daily):
  - Valuation:      forward_pe, trailing_pe, peg_ratio
  - Growth:         revenue_growth (YoY), earnings_growth (YoY)
  - Profitability:  profit_margin, roe (return on equity)
  - Balance sheet:  debt_to_equity, free_cashflow
  - Analyst:        rec_mean (1=strong buy, 5=strong sell), target_price
  - Meta:           market_cap, sector, fetched_at

Quality score (0-100) aggregates these into a single number using
threshold-based buckets calibrated to a tech-heavy universe.

Cache: Models/fundamentals.parquet, refreshed daily (24h TTL).
Auto-refreshed by daily-update.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, WATCHLIST, ML_EXTRA_TRAINING_SYMBOLS

FUNDAMENTALS_CACHE_PATH = MODELS_DIR / 'fundamentals.parquet'

# All symbols we might want fundamentals for
_ALL_SYMBOLS = list(WATCHLIST.keys()) + list(ML_EXTRA_TRAINING_SYMBOLS)


# ── Fetch from yfinance ───────────────────────────────────────────────────

def _fetch_one_ticker(ticker: str) -> dict:
    """
    Fetch fundamental fields for one ticker via yfinance.

    Returns dict with fields; None/np.nan for missing values.
    yfinance is inconsistent — most fields will be there, some won't.
    """
    import yfinance as yf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = yf.Ticker(ticker)
            info = t.info or {}
    except Exception as e:
        print(f'  [fundamentals] {ticker}: fetch failed: {e}')
        return {'ticker': ticker, 'fetched_at': datetime.now().isoformat(),
                 'fetch_ok': False}

    if not info or info.get('quoteType') is None:
        print(f'  [fundamentals] {ticker}: empty info')
        return {'ticker': ticker, 'fetched_at': datetime.now().isoformat(),
                 'fetch_ok': False}

    def _get(key, default=np.nan):
        v = info.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    return {
        'ticker':             ticker,
        'fetched_at':         datetime.now().isoformat(),
        'fetch_ok':           True,
        'sector':             info.get('sector', ''),
        'market_cap':         _get('marketCap'),
        'current_price':      _get('currentPrice') or _get('regularMarketPrice'),
        # Valuation
        'forward_pe':         _get('forwardPE'),
        'trailing_pe':        _get('trailingPE'),
        'peg_ratio':          _get('trailingPegRatio') or _get('pegRatio'),
        'price_to_book':      _get('priceToBook'),
        'price_to_sales':     _get('priceToSalesTrailing12Months'),
        # Growth (yfinance returns fractions, e.g. 0.43 = 43%)
        'revenue_growth':     _get('revenueGrowth'),
        'earnings_growth':    _get('earningsGrowth'),
        # Profitability
        'profit_margin':      _get('profitMargins'),
        'roe':                _get('returnOnEquity'),
        'roa':                _get('returnOnAssets'),
        # Balance sheet
        'debt_to_equity':     _get('debtToEquity'),   # often reported as *100
        'current_ratio':      _get('currentRatio'),
        'free_cashflow':      _get('freeCashflow'),
        # Analyst sentiment
        'analyst_rec_mean':   _get('recommendationMean'),
        'analyst_target':     _get('targetMeanPrice'),
        'num_analysts':       _get('numberOfAnalystOpinions'),
        # Beta (for context)
        'beta':               _get('beta'),
    }


def refresh_all(force: bool = False) -> pd.DataFrame:
    """
    Fetch fundamentals for all symbols (WATCHLIST + ML_EXTRA_TRAINING_SYMBOLS).

    Cache freshness: 24-hour TTL. Pass force=True to override.
    Returns the cached DataFrame.
    """
    # Cache-hit path
    if not force and FUNDAMENTALS_CACHE_PATH.exists():
        age_hours = (datetime.now().timestamp() -
                     FUNDAMENTALS_CACHE_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            cached = pd.read_parquet(FUNDAMENTALS_CACHE_PATH)
            if len(cached) >= len(_ALL_SYMBOLS) * 0.8:  # 80% coverage acceptable
                print(f'[fundamentals] Cached (age {age_hours:.1f}h) — '
                      f'{len(cached)} tickers')
                return cached

    print(f'[fundamentals] Fetching for {len(_ALL_SYMBOLS)} tickers via yfinance...')
    rows = []
    for i, ticker in enumerate(_ALL_SYMBOLS, 1):
        row = _fetch_one_ticker(ticker)
        rows.append(row)
        if i % 10 == 0:
            print(f'  [{i}/{len(_ALL_SYMBOLS)}] fetched')

    df = pd.DataFrame(rows).set_index('ticker')

    # Add computed quality score
    df['quality_score'] = df.apply(_compute_quality_score, axis=1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUNDAMENTALS_CACHE_PATH)
    ok_count = int(df['fetch_ok'].sum()) if 'fetch_ok' in df.columns else len(df)
    print(f'[fundamentals] Cached {len(df)} tickers ({ok_count} with data) -> '
          f'{FUNDAMENTALS_CACHE_PATH}')
    return df


def get_ticker(ticker: str, force_refresh: bool = False) -> dict:
    """
    Return fundamentals for one ticker, using cache if fresh.

    If ticker not in cache, fetches just that one.
    """
    ticker = ticker.upper()
    if force_refresh or not FUNDAMENTALS_CACHE_PATH.exists():
        refresh_all(force=force_refresh)

    if FUNDAMENTALS_CACHE_PATH.exists():
        df = pd.read_parquet(FUNDAMENTALS_CACHE_PATH)
        if ticker in df.index:
            row = df.loc[ticker].to_dict()
            row['ticker'] = ticker
            return row

    # Not in cache — fetch just this one
    row = _fetch_one_ticker(ticker)
    row['quality_score'] = _compute_quality_score(pd.Series(row))
    return row


# ── Quality score computation ────────────────────────────────────────────

def _compute_quality_score(row: pd.Series) -> float:
    """
    Aggregate fundamentals into a 0-100 quality score.

    Breakdown (max points):
      Growth        20   (revenue + earnings YoY)
      Valuation     20   (forward PE + PEG)
      Profitability 20   (ROE + profit margin)
      Balance       15   (debt/equity + FCF)
      Analyst       25   (recommendation + target upside)

    Missing values contribute 0 to their category (not penalized) so
    stocks with sparse yfinance data aren't unfairly zeroed.
    """
    if not row.get('fetch_ok', True):
        return np.nan

    score = 0.0

    # ── Growth (20 pts) ──────────────────────────────────────────────────
    rev_growth = row.get('revenue_growth', np.nan)
    if not pd.isna(rev_growth):
        if   rev_growth > 0.30: score += 12
        elif rev_growth > 0.15: score += 8
        elif rev_growth > 0.05: score += 4
    earn_growth = row.get('earnings_growth', np.nan)
    if not pd.isna(earn_growth):
        if   earn_growth > 0.20: score += 8
        elif earn_growth > 0.05: score += 5

    # ── Valuation (20 pts) ───────────────────────────────────────────────
    fwd_pe = row.get('forward_pe', np.nan)
    if not pd.isna(fwd_pe) and fwd_pe > 0:
        if   fwd_pe < 15: score += 15
        elif fwd_pe < 25: score += 10
        elif fwd_pe < 40: score += 5
    peg = row.get('peg_ratio', np.nan)
    if not pd.isna(peg) and peg > 0:
        if   peg < 1:  score += 5    # cheap for growth
        elif peg > 3:  score -= 5    # expensive for growth
    # Clip valuation contribution to [0, 20]
    score = max(0, score)

    # ── Profitability (20 pts) ───────────────────────────────────────────
    roe = row.get('roe', np.nan)
    if not pd.isna(roe):
        if   roe > 0.25: score += 10
        elif roe > 0.15: score += 7
        elif roe > 0.08: score += 3
    pm = row.get('profit_margin', np.nan)
    if not pd.isna(pm):
        if   pm > 0.25: score += 10
        elif pm > 0.15: score += 7
        elif pm > 0.05: score += 3

    # ── Balance sheet (15 pts) ───────────────────────────────────────────
    de = row.get('debt_to_equity', np.nan)
    # yfinance often reports D/E as a percentage (e.g., 85 for 85%) — normalize
    if not pd.isna(de):
        if de > 5: de = de / 100   # heuristic: any value > 5 is likely percentage
        if   de < 0.5: score += 10
        elif de < 1.0: score += 7
        elif de < 2.0: score += 3
    fcf = row.get('free_cashflow', np.nan)
    if not pd.isna(fcf) and fcf > 0:
        score += 5

    # ── Analyst sentiment (25 pts) ───────────────────────────────────────
    rec = row.get('analyst_rec_mean', np.nan)
    if not pd.isna(rec):
        if   rec < 1.5: score += 15
        elif rec < 2.0: score += 10
        elif rec < 2.5: score += 5
    target = row.get('analyst_target', np.nan)
    price  = row.get('current_price', np.nan)
    if not pd.isna(target) and not pd.isna(price) and price > 0:
        upside = target / price - 1
        if   upside > 0.20: score += 10
        elif upside > 0.10: score += 5
        elif upside > 0.00: score += 2

    return min(100, max(0, round(score, 1)))


# ── Verdict label ────────────────────────────────────────────────────────

def leaderboard(filter_mode: str = 'all') -> pd.DataFrame:
    """
    Print a sorted leaderboard of all tickers by fundamental quality score.

    filter_mode:
      'all'          — every ticker
      'strong'       — quality score >= 70 only
      'quality-dips' — quality >= 70 AND technical signal weak (advisor SKIP)
                       These are watchlist-priority: quality names at oversold levels.

    Returns the leaderboard DataFrame (also prints).
    """
    df = refresh_all(force=False)
    if 'fetch_ok' in df.columns:
        df = df[df['fetch_ok'] == True]

    # Filter
    if filter_mode == 'strong':
        df = df[df['quality_score'] >= 70]
    elif filter_mode == 'quality-dips':
        # Needs technical context — compute in caller or delegate
        df = df[df['quality_score'] >= 70]  # start with strong, filter for dips downstream

    df = df.sort_values('quality_score', ascending=False)

    print(f'\n{"="*82}')
    print(f'  FUNDAMENTALS LEADERBOARD — {len(df)} tickers  (filter: {filter_mode})')
    print(f'{"="*82}')
    header = f'  {"#":<3} {"Ticker":<6} {"Score":>6} {"Rev%":>7} {"Earn%":>7} {"FwdPE":>7} {"PEG":>5} {"ROE":>7} {"Target%":>8} {"Rec":>5} {"Verdict":>8}'
    print(header)
    print(f'  {"-"*80}')
    for i, (ticker, r) in enumerate(df.iterrows(), 1):
        emoji, verdict, _ = quality_label(r['quality_score'])
        rev  = f'{r["revenue_growth"]*100:+6.1f}%' if not pd.isna(r["revenue_growth"]) else '   N/A'
        earn = f'{r["earnings_growth"]*100:+6.1f}%' if not pd.isna(r["earnings_growth"]) else '   N/A'
        pe   = f'{r["forward_pe"]:>6.1f}' if not pd.isna(r["forward_pe"]) else '   N/A'
        peg  = f'{r["peg_ratio"]:>4.2f}' if not pd.isna(r["peg_ratio"]) else ' N/A'
        roe  = f'{r["roe"]*100:+6.1f}%' if not pd.isna(r["roe"]) else '   N/A'
        upside = '   N/A'
        if not pd.isna(r["analyst_target"]) and not pd.isna(r["current_price"]) and r["current_price"] > 0:
            up = r["analyst_target"] / r["current_price"] - 1
            upside = f'{up*100:+6.1f}%'
        rec = f'{r["analyst_rec_mean"]:>4.2f}' if not pd.isna(r["analyst_rec_mean"]) else ' N/A'
        print(f'  {i:<3} {ticker:<6} {r["quality_score"]:>5.0f}  {rev} {earn} {pe} {peg} {roe} {upside} {rec}  {emoji} {verdict}')

    return df


def quality_label(score: float) -> tuple:
    """
    Convert a quality score into (emoji, verdict, description).
    """
    if pd.isna(score):
        return ('⚪', 'UNKNOWN', 'fundamentals data unavailable')
    if score >= 70:
        return ('🟢', 'STRONG',
                 'high-quality name — growth + profitability + analyst confidence')
    if score >= 50:
        return ('🟡', 'DECENT',
                 'mixed quality — some strong metrics, some weak')
    if score >= 30:
        return ('🟠', 'WEAK',
                 'below-average fundamentals — trade with caution')
    return ('🔴', 'POOR',
             'weak fundamentals — likely a low-quality or distressed name')
