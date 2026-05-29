"""
Phase 1: Intraday 30-Minute Polling & Analytics Engine

One polling cycle:
  1. Fetch 30-min bars from Alpaca for all 8 stocks + SPY/QQQ (15 days back)
  2. Determine market regime from deterministic EMA/VWAP/structure rules
  3. Check VIX — hard block at >= 30
  4. Compute RATMB indicators + signal score for each ticker
  5. Pass confirmed signals (score >= threshold + clearance == 1) to execution_callback
"""
import pytz
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    WATCHLIST, MARKET_INDEXES, SIGNAL_BUY_THRESHOLD, SIGNAL_WATCH_THRESHOLD,
    VIX_HARD_BLOCK,
)
import database as db
import technicals as ta

ET = pytz.timezone('America/New_York')


# ── Market-hours helpers ───────────────────────────────────────────────────

def is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mo = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    mc = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return mo <= now <= mc


def is_tradeable_window() -> bool:
    """Enforce: no trades first 10 min or last 20 min of session."""
    if not is_market_hours():
        return False
    now = datetime.now(ET)
    earliest = now.replace(hour=9,  minute=40, second=0, microsecond=0)
    latest   = now.replace(hour=15, minute=40, second=0, microsecond=0)
    return earliest <= now <= latest


# ── Alpaca data fetch ──────────────────────────────────────────────────────

def fetch_bars(tickers: list, days_back: int = 7) -> dict:
    """
    Fetch 30-min OHLCV bars from Alpaca Historical Data API.
    Returns {ticker: DataFrame} with ET-localized DatetimeIndex.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    end    = datetime.now(ET)
    start  = end - timedelta(days=days_back)

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,   # free-tier feed; change to DataFeed.SIP with paid subscription
    )
    raw_bars = client.get_stock_bars(req)
    bars_df  = raw_bars.df   # MultiIndex (symbol, timestamp)

    result = {}
    for ticker in tickers:
        try:
            df = bars_df.loc[ticker].copy()
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
            df.columns = [c.lower() for c in df.columns]
            result[ticker] = df
        except (KeyError, AttributeError):
            print(f'  [Alpaca] No data for {ticker}')
    return result


# ── Deterministic regime classifier ───────────────────────────────────────

def get_regime_bias(spy_df: pd.DataFrame, qqq_df: pd.DataFrame) -> dict:
    """
    Classify market regime from SPY/QQQ structural indicators (no LLM).

    Scoring (7 points total):
      +2  SPY EMA20 > EMA50
      +2  QQQ EMA20 > EMA50
      +1  SPY close holds VWAP
      +1  SPY higher-highs
      +1  SPY higher-lows

    Returns {"regime_bias": "bullish"|"neutral"|"bearish", "confidence": 0-1}.
    """
    default = {'regime_bias': 'neutral', 'confidence': 0.5}

    ind_spy = ta.compute_indicators(spy_df)
    ind_qqq = ta.compute_indicators(qqq_df)
    if not ind_spy or not ind_qqq:
        print('  [Regime] Insufficient SPY/QQQ data — defaulting to neutral')
        return default

    bullish = 0
    if ind_spy['ema20'] > ind_spy['ema50']:  bullish += 2
    if ind_qqq['ema20'] > ind_qqq['ema50']:  bullish += 2
    if ind_spy.get('vwap_hold'):              bullish += 1
    if ind_spy.get('higher_highs'):           bullish += 1
    if ind_spy.get('higher_lows'):            bullish += 1

    bearish = 7 - bullish
    if bullish >= 5:
        return {'regime_bias': 'bullish', 'confidence': round(bullish / 7, 2)}
    if bearish >= 5:
        return {'regime_bias': 'bearish', 'confidence': round(bearish / 7, 2)}
    return default


# ── Main polling cycle ─────────────────────────────────────────────────────

def run_polling_cycle(
    execution_callback: Optional[Callable] = None,
    test_mode: bool = False,
    force_run: bool = False,
) -> list:
    """
    Execute one 30-minute polling cycle.

    execution_callback(ticker, indicators, signal_score, regime_bias):
        Called for each signal that clears both the score threshold and DB clearance.

    test_mode=True  → skips market-hours gate; uses 7 days of historical data.
    force_run=True  → also skips tradeable-window gate.

    Returns list of scored signal dicts (all tickers, sorted by score desc).
    """
    now_str = datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
    print(f"\n{'='*60}")
    print(f'  PHASE 1 — Polling Cycle  [{now_str}]')
    print(f"{'='*60}")

    if not test_mode and not force_run:
        if not is_market_hours():
            print('[Phase 1] Market is closed. Use --test or --force to override.')
            return []
        if not is_tradeable_window():
            print('[Phase 1] Outside tradeable window (first 10 / last 20 min of session).')
            return []

    all_tickers = list(WATCHLIST.keys()) + MARKET_INDEXES
    days_back   = 15   # needs 55+ bars for EMA-50 warmup; 15 days covers ~100 bars
    print(f'[Alpaca] Fetching {days_back}-day 30-min bars for: {", ".join(all_tickers)}')

    all_bars = fetch_bars(all_tickers, days_back=days_back)

    spy_df = all_bars.get('SPY', pd.DataFrame())
    qqq_df = all_bars.get('QQQ', pd.DataFrame())

    # ── VIX hard block ─────────────────────────────────────────────────────
    try:
        import yfinance as yf
        vix_hist = yf.Ticker('^VIX').history(period='2d')
        vix = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 0.0
    except Exception as e:
        print(f'  [VIX] Could not fetch VIX: {e} — proceeding without block')
        vix = 0.0

    print(f'  VIX = {vix:.1f}')
    if vix >= VIX_HARD_BLOCK:
        print(f'[Phase 1] VIX={vix:.1f} >= {VIX_HARD_BLOCK} — hard risk-off, no new trades.')
        return []

    # ── Regime check (deterministic, no LLM) ──────────────────────────────
    print('[Regime] Computing market regime...')
    if len(spy_df) >= 55 and len(qqq_df) >= 55:
        regime = get_regime_bias(spy_df, qqq_df)
    else:
        regime = {'regime_bias': 'neutral', 'confidence': 0.5}
        print('  [Regime] Insufficient SPY/QQQ data — defaulted to neutral')

    regime_bias = regime['regime_bias']
    regime_conf = regime['confidence']
    print(f'  Regime: {regime_bias.upper()} (confidence {regime_conf:.0%})')

    # Hard risk-off gate: bearish regime with high confidence → no trades
    if regime_bias == 'bearish' and regime_conf >= 0.80:
        print('[Phase 1] Risk-off regime detected — no trades this cycle.')
        return []

    # ── EMA alignment flags for ML feature row ────────────────────────────
    _ind_spy = ta.compute_indicators(spy_df) if len(spy_df) >= 55 else None
    _ind_qqq = ta.compute_indicators(qqq_df) if len(qqq_df) >= 55 else None
    spy_ema_aligned = int(_ind_spy['ema20'] > _ind_spy['ema50']) if _ind_spy else 0
    qqq_ema_aligned = int(_ind_qqq['ema20'] > _ind_qqq['ema50']) if _ind_qqq else 0

    # ── Score each watchlist ticker ────────────────────────────────────────
    scored = []
    for ticker in WATCHLIST:
        df = all_bars.get(ticker)
        if df is None or len(df) < 55:
            bars_count = len(df) if df is not None else 0
            print(f'  [{ticker}] Skipped — only {bars_count} bars (need 55)')
            continue

        indicators = ta.compute_indicators(df)
        if indicators is None:
            continue

        rs = ta.compute_rs_vs_qqq(df, qqq_df) if len(qqq_df) >= 10 else 0.0
        base_score, components = ta.score_signal(indicators, rs)
        final_score = ta.apply_regime_multiplier(base_score, regime_bias, regime_conf)

        indicators.update({
            'rs_vs_qqq':        round(rs, 4),
            'regime_bias':      regime_bias,
            'signal_score':     round(final_score, 1),
            'spy_ema_aligned':  spy_ema_aligned,
            'qqq_ema_aligned':  qqq_ema_aligned,
        })

        label = '🟢 BUY SIGNAL' if final_score >= SIGNAL_BUY_THRESHOLD \
            else '🟡 WATCH' if final_score >= SIGNAL_WATCH_THRESHOLD \
            else '⚪ BELOW'

        print(f'  [{ticker}] Score={final_score:.1f}/100  '
              f'(base={base_score} x regime={regime_bias})  -> {label}')

        scored.append({
            'ticker':      ticker,
            'final_score': round(final_score, 1),
            'base_score':  base_score,
            'components':  components,
            'indicators':  indicators,
            'bar_df':      df,
            'regime_bias': regime_bias,
            'regime_conf': regime_conf,
        })

    scored.sort(key=lambda x: x['final_score'], reverse=True)

    # ── Route confirmed signals to execution ───────────────────────────────
    signals_fired = 0
    for signal in scored:
        if signal['final_score'] < SIGNAL_BUY_THRESHOLD:
            continue

        ticker     = signal['ticker']
        indicators = signal['indicators']
        bar_df     = signal.get('bar_df')
        timestamp  = bar_df.index[-1] if bar_df is not None and len(bar_df) > 0 else None
        clearance  = db.get_today_clearance(ticker)
        clr_int    = clearance['clearance'] if clearance else 0

        if not clr_int:
            reason = 'no daily clearance' if clearance else 'Phase 0 not yet run today'
            print(f'  [{ticker}] Blocked — {reason}')
            db.save_signal(ticker, signal['final_score'], 0, False, reason, indicators)
            continue

        print(f'  [{ticker}] Signal confirmed + cleared! Forwarding to execution...')
        db.save_signal(ticker, signal['final_score'], 1, True, None, indicators)
        signals_fired += 1

        if execution_callback:
            execution_callback(ticker, indicators, signal['final_score'], regime_bias,
                               bar_df=bar_df, timestamp=timestamp)

    print(f'\n[Phase 1] Done — {len(scored)} scored, '
          f'{signals_fired} signals executed\n')
    return scored
