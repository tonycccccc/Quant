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
        'operating_margin':   _get('operatingMargins'),
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

def compute_fair_value(row: pd.Series) -> dict:
    """
    Compute a growth-adjusted fair value estimate from reported fundamentals.

    Method (independent of analyst targets):
      forward_eps         = current_price / forward_pe
      sustainable_growth  = weighted avg of revenue + earnings growth
                            (revenue weighted higher — more sustainable)
      fair_pe             = growth-tier lookup (5-tier ladder)
      fair_value          = forward_eps * fair_pe
      discount_to_fair    = (fair_value - price) / price

    Returns dict with all computed values, or None if insufficient data.
    """
    fwd_pe = row.get('forward_pe', np.nan)
    price  = row.get('current_price', np.nan)
    if pd.isna(fwd_pe) or fwd_pe <= 0 or pd.isna(price) or price <= 0:
        return {}

    forward_eps = price / fwd_pe

    # Sustainable growth: revenue is more reliable than earnings (which can spike
    # from one-time gains, tax effects, or coming off a low base). Weight
    # revenue at 60%, earnings at 40%. Cap at 40% overall — no company
    # sustains growth above that long enough for a multi-year fair-value.
    rev_g  = row.get('revenue_growth', np.nan)
    earn_g = row.get('earnings_growth', np.nan)
    _CAP = 0.40   # long-term sustainable growth ceiling
    if not pd.isna(rev_g) and not pd.isna(earn_g):
        # Cap earnings at 2x revenue (still generous)
        earn_g_capped = min(earn_g, 2 * abs(rev_g) if rev_g > 0 else 0.30)
        sustainable_growth = 0.6 * min(rev_g, _CAP) + 0.4 * min(earn_g_capped, _CAP)
    elif not pd.isna(rev_g):
        sustainable_growth = min(rev_g, _CAP)
    elif not pd.isna(earn_g):
        sustainable_growth = min(earn_g, _CAP)
    else:
        return {}

    # Fair PE ladder based on sustainable growth tier
    if   sustainable_growth > 0.30: fair_pe = 35    # hyper-growth premium
    elif sustainable_growth > 0.15: fair_pe = 25    # strong growth
    elif sustainable_growth > 0.05: fair_pe = 18    # mature
    elif sustainable_growth > 0:    fair_pe = 12    # slow
    else:                            fair_pe = 8    # declining

    fair_value = forward_eps * fair_pe
    discount_to_fair = (fair_value - price) / price   # positive = undervalued

    return {
        'forward_eps':        forward_eps,
        'sustainable_growth': sustainable_growth,
        'fair_pe':            fair_pe,
        'fair_value':         fair_value,
        'discount_to_fair':   discount_to_fair,
    }


def _compute_quality_score(row: pd.Series) -> float:
    """
    Aggregate fundamentals into a 0-100 quality score.

    Emphasizes HARD FINANCIALS reported in earnings filings; de-weights
    analyst opinions (targets are unreliable and often lagging).

    Breakdown (max points):
      Growth        25   (revenue + earnings YoY, from reported 10-Q/10-K)
      Valuation     25   (forward PE + PEG + P/S — mathematical, based on filings)
      Profitability 25   (ROE + profit margin + operating margin — reported)
      Balance sheet 15   (debt/equity + FCF — reported)
      Analyst rec   10   (buy/hold/sell only — NO target price)

    Extension penalty: if analyst target ≤ current price (analysts see no
    upside), score is capped at 65 (DECENT tier) regardless of underlying
    strength — the market has already priced in the fundamentals.

    Missing values contribute 0 to their category (not penalized).
    """
    if not row.get('fetch_ok', True):
        return np.nan

    score = 0.0

    # ── Growth (25 pts, from reported filings) ────────────────────────────
    rev_growth = row.get('revenue_growth', np.nan)
    if not pd.isna(rev_growth):
        if   rev_growth > 0.30: score += 15    # top-tier growth
        elif rev_growth > 0.15: score += 10
        elif rev_growth > 0.05: score += 5
    earn_growth = row.get('earnings_growth', np.nan)
    if not pd.isna(earn_growth):
        if   earn_growth > 0.20: score += 10
        elif earn_growth > 0.05: score += 5

    # ── Valuation (25 pts, mathematical from reported earnings) ───────────
    fwd_pe = row.get('forward_pe', np.nan)
    if not pd.isna(fwd_pe) and fwd_pe > 0:
        if   fwd_pe < 15: score += 12
        elif fwd_pe < 25: score += 8
        elif fwd_pe < 40: score += 4
    trailing_pe = row.get('trailing_pe', np.nan)
    if not pd.isna(trailing_pe) and trailing_pe > 0:
        # Trailing PE cross-check — punish if wildly higher than forward
        # (indicates earnings deteriorating fast)
        if not pd.isna(fwd_pe) and trailing_pe > fwd_pe * 2.5:
            score -= 3
    peg = row.get('peg_ratio', np.nan)
    if not pd.isna(peg) and peg > 0:
        if   peg < 1:  score += 8       # cheap for growth
        elif peg < 2:  score += 3
        elif peg > 3:  score -= 3       # expensive for growth
    ps = row.get('price_to_sales', np.nan)
    if not pd.isna(ps) and ps > 0 and ps < 5:
        score += 5   # reasonable P/S ratio bonus
    score = max(0, score)   # never negative from valuation alone

    # ── Profitability (25 pts, from reported financials) ─────────────────
    roe = row.get('roe', np.nan)
    if not pd.isna(roe):
        if   roe > 0.30: score += 12   # exceptional (NVDA, TSM territory)
        elif roe > 0.20: score += 9
        elif roe > 0.12: score += 5
        elif roe > 0.05: score += 2
    pm = row.get('profit_margin', np.nan)
    if not pd.isna(pm):
        if   pm > 0.25: score += 8
        elif pm > 0.15: score += 5
        elif pm > 0.05: score += 2
    # Operating margin (if available in row — extend to fetch later)
    om = row.get('operating_margin', np.nan)
    if not pd.isna(om):
        if om > 0.20: score += 5
        elif om > 0.10: score += 2

    # ── Balance sheet (15 pts) ───────────────────────────────────────────
    de = row.get('debt_to_equity', np.nan)
    if not pd.isna(de):
        if de > 5: de = de / 100   # yfinance sometimes reports as percent
        if   de < 0.5: score += 10
        elif de < 1.0: score += 7
        elif de < 2.0: score += 3
    fcf = row.get('free_cashflow', np.nan)
    if not pd.isna(fcf) and fcf > 0:
        score += 5

    # ── Analyst recommendation only (10 pts — target price ignored) ──────
    # Analyst rec is a consensus of many analysts, generally more reliable
    # than any single target price. We take the buy/hold/sell signal but
    # NOT the target upside (targets are often revised after the move).
    rec = row.get('analyst_rec_mean', np.nan)
    if not pd.isna(rec):
        if   rec < 1.5: score += 10      # strong buy consensus
        elif rec < 2.0: score += 7       # buy
        elif rec < 2.5: score += 3       # hold-buy

    raw_score = min(100, max(0, round(score, 1)))

    # ── Fair-value check (independent of analysts, derived from growth) ──
    # Compute what the stock SHOULD be worth given its growth rate, and
    # compare to current price. This is a stronger signal than analyst
    # target because it's derived directly from reported financials.
    fv = compute_fair_value(row)
    if fv:
        discount = fv['discount_to_fair']    # positive = undervalued
        if   discount > 0.20:  raw_score += 5    # deeply undervalued (>20% discount)
        elif discount > 0.05:  raw_score += 3    # undervalued (5-20% discount)
        elif discount < -0.30: raw_score -= 10   # deeply overvalued
        elif discount < -0.15: raw_score -= 5    # overvalued
        # Hard cap when significantly overvalued
        if discount < -0.20: raw_score = min(raw_score, 60)  # DECENT max
        if discount < -0.35: raw_score = min(raw_score, 40)  # WEAK max

    # ── Analyst-target cross-check (secondary signal) ────────────────────
    # Use analyst target ONLY as a sanity check on the fair-value calc.
    target = row.get('analyst_target', np.nan)
    price  = row.get('current_price', np.nan)
    if not pd.isna(target) and not pd.isna(price) and price > 0:
        upside = target / price - 1
        # Only apply extension cap if BOTH signals agree it's overvalued
        if upside <= 0.00 and fv and fv.get('discount_to_fair', 0) < 0:
            raw_score = min(raw_score, 65)   # capped at DECENT

    return min(100, max(0, round(raw_score, 1)))


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

    print(f'\n{"="*94}')
    print(f'  FUNDAMENTALS LEADERBOARD — {len(df)} tickers  (filter: {filter_mode})')
    print(f'{"="*94}')
    header = (f'  {"#":<3} {"Ticker":<6} {"Score":>5} {"Rev%":>7} {"Earn%":>7} '
              f'{"FwdPE":>6} {"PEG":>5} {"ROE":>7} {"FairVal":>8} {"vs Fair":>8} {"Rec":>5} {"Verdict":>9}')
    print(header)
    print(f'  {"-"*92}')
    for i, (ticker, r) in enumerate(df.iterrows(), 1):
        emoji, verdict, _ = quality_label(r['quality_score'])
        rev  = f'{r["revenue_growth"]*100:+6.1f}%' if not pd.isna(r["revenue_growth"]) else '   N/A'
        earn = f'{r["earnings_growth"]*100:+6.1f}%' if not pd.isna(r["earnings_growth"]) else '   N/A'
        pe   = f'{r["forward_pe"]:>5.1f}' if not pd.isna(r["forward_pe"]) else '  N/A'
        peg  = f'{r["peg_ratio"]:>4.2f}' if not pd.isna(r["peg_ratio"]) else ' N/A'
        roe  = f'{r["roe"]*100:+6.1f}%' if not pd.isna(r["roe"]) else '   N/A'
        rec  = f'{r["analyst_rec_mean"]:>4.2f}' if not pd.isna(r["analyst_rec_mean"]) else ' N/A'
        # Fair value from growth-adjusted computation
        fv = compute_fair_value(r)
        fair_val_str = f'${fv["fair_value"]:>6.0f}' if fv else '     N/A'
        vs_fair_str  = f'{fv["discount_to_fair"]*100:+6.1f}%' if fv else '     N/A'
        print(f'  {i:<3} {ticker:<6} {r["quality_score"]:>4.0f}  {rev} {earn} '
              f'{pe} {peg} {roe} {fair_val_str} {vs_fair_str}  {rec}  {emoji} {verdict}')

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
