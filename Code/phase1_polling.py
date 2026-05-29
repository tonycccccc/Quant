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
from ml.alert_log import log_alert
from ml.features import indicators_to_feature_row

try:
    from ml.predict import predict_success_prob
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
    def predict_success_prob(feat):
        return float('nan')

ET = pytz.timezone('America/New_York')


# ── Cross-sectional rank helpers ──────────────────────────────────────────

def _compute_live_daily_feature(df: pd.DataFrame, kind: str) -> float:
    """
    Compute one daily-timeframe value for ranking purposes from the live 30-min
    bar_df. Mirrors the resample logic in ml/features._compute_daily_features
    but returns a single scalar (latest available daily bar).

    Used to populate ranks; the actual feature_row daily values come from
    indicators_to_feature_row which uses the same resample.
    """
    if df is None or len(df) < 100:
        return 0.0
    idx_et = df.index.tz_convert('America/New_York') if df.index.tz else df.index
    daily = df.copy()
    daily.index = idx_et
    by_day = daily.resample('B').agg(close=('close', 'last')).dropna()
    if len(by_day) < 21:
        return 0.0
    if kind == 'd_return_20d':
        return float(by_day['close'].iloc[-1] / by_day['close'].iloc[-21] - 1)
    return 0.0


def _inject_cross_sectional_ranks(raw_indicators: dict) -> None:
    """
    Compute pct-ranks across all in-flight watchlist tickers and write them
    back into each ticker's indicators dict.

    Mirrors training-time _add_cross_sectional_ranks: pandas rank(pct=True)
    across symbols at the same bar. Mutates raw_indicators in place.

    Ranked fields:
      - prior_5d_return  -> rs_rank_5d         (proxy from 30-min bar history)
      - rsi              -> rsi_rank
      - d_return_20d     -> momentum_rank_20d
      - vol_ratio        -> vol_ratio_rank
    """
    import pandas as pd
    if not raw_indicators:
        return

    rows = []
    tickers = []
    for ticker, (indicators, df, rs) in raw_indicators.items():
        bar_close = df['close']
        prior_5d = float(bar_close.iloc[-1] / bar_close.iloc[-66] - 1) \
                   if len(bar_close) >= 66 else 0.0
        vol_ratio = (indicators.get('volume', 0.0) /
                     indicators['volume_avg']) if indicators.get('volume_avg', 0) > 0 else 1.0
        d_return_20d = _compute_live_daily_feature(df, 'd_return_20d')
        rows.append({
            'prior_5d_return': prior_5d,
            'rsi':             float(indicators.get('rsi', 50.0)),
            'd_return_20d':    d_return_20d,
            'vol_ratio':       float(vol_ratio),
        })
        tickers.append(ticker)

    if len(rows) < 2:
        # With <2 tickers, ranks aren't meaningful. Leave defaults (0.5).
        for ticker in tickers:
            indicators, _, _ = raw_indicators[ticker]
            indicators.setdefault('rs_rank_5d', 0.5)
            indicators.setdefault('rsi_rank', 0.5)
            indicators.setdefault('momentum_rank_20d', 0.5)
            indicators.setdefault('vol_ratio_rank', 0.5)
        return

    cross = pd.DataFrame(rows, index=tickers)
    rank_map = {
        'prior_5d_return': 'rs_rank_5d',
        'rsi':             'rsi_rank',
        'd_return_20d':    'momentum_rank_20d',
        'vol_ratio':       'vol_ratio_rank',
    }
    for src, dst in rank_map.items():
        ranks = cross[src].rank(pct=True, method='average').fillna(0.5)
        for ticker, val in ranks.items():
            raw_indicators[ticker][0][dst] = float(val)


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
    # 90 calendar days ~= 63 business days. Required to populate ALL daily
    # features in indicators_to_feature_row():
    #   - d_ema_aligned needs ~50 daily bars (EMA-50)
    #   - d_return_20d / d_vol_ratio need 20 daily bars
    #   - 30-min EMA-50 needs ~55 bars (covered comfortably by 63 days * 13)
    # Without this, daily features (top 5 by importance) silently default to 0
    # at inference, creating a major train/inference mismatch.
    days_back = 90
    print(f'[Alpaca] Fetching {days_back}-day 30-min bars for: {", ".join(all_tickers)}')

    all_bars = fetch_bars(all_tickers, days_back=days_back)

    spy_df = all_bars.get('SPY', pd.DataFrame())
    qqq_df = all_bars.get('QQQ', pd.DataFrame())

    # ── VIX hard block (FAIL-SAFE: unknown VIX = no trade) ─────────────────
    vix = None
    try:
        import yfinance as yf
        vix_hist = yf.Ticker('^VIX').history(period='2d')
        if not vix_hist.empty:
            vix = float(vix_hist['Close'].iloc[-1])
    except Exception as e:
        print(f'  [VIX] Fetch failed: {e}')

    if vix is None:
        print('[Phase 1] VIX UNKNOWN — refusing to trade (fail-safe). '
              'Check yfinance / network before live trading.')
        return []

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
    # Pass 1: compute indicators + base score for every ticker
    scored = []
    raw_indicators = {}   # ticker -> indicators dict (mutable, ranks added in pass 2)
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
        indicators.update({
            'rs_vs_qqq':        round(rs, 4),
            'regime_bias':      regime_bias,
            'spy_ema_aligned':  spy_ema_aligned,
            'qqq_ema_aligned':  qqq_ema_aligned,
        })
        raw_indicators[ticker] = (indicators, df, rs)

    # ── Cross-sectional ranks across all in-flight watchlist tickers ──────
    # Mirrors the training-time rank computation in features.py so the ML model
    # sees real rank values, not the 0.5 default. Without this, momentum_rank_20d
    # (#6 feature by importance) is dead at inference.
    _inject_cross_sectional_ranks(raw_indicators)

    # Pass 2: finalize scores and ML-feature-row construction
    for ticker, (indicators, df, rs) in raw_indicators.items():
        base_score, components = ta.score_signal(indicators, rs)
        final_score = ta.apply_regime_multiplier(base_score, regime_bias, regime_conf)
        indicators['signal_score']       = round(final_score, 1)
        indicators['primary_base_score'] = round(base_score, 1)   # train/inference parity for ML primary_score feature

        label = '🟢 BUY SIGNAL' if final_score >= SIGNAL_BUY_THRESHOLD \
            else '🟡 WATCH' if final_score >= SIGNAL_WATCH_THRESHOLD \
            else '⚪ BELOW'

        print(f'  [{ticker}] Score={final_score:.1f}/100  '
              f'(base={base_score} x regime={regime_bias})  -> {label}')

        # ── Alert logging: every score >= WATCH gets a training-data row ──
        if final_score >= SIGNAL_WATCH_THRESHOLD:
            try:
                feat_row = indicators_to_feature_row(
                    indicators, df, df.index[-1], rs_vs_qqq=rs,
                )
                ml_prob = predict_success_prob(feat_row) if _ML_AVAILABLE else None
                clearance_today = db.get_today_clearance(ticker)
                cleared_flag = bool(clearance_today and clearance_today.get('clearance'))
                regime_mult = final_score / base_score if base_score > 0 else 1.0
                log_alert(
                    symbol=ticker,
                    timestamp=df.index[-1],
                    base_score=base_score,
                    regime_multiplier=regime_mult,
                    final_score=final_score,
                    regime_bias=regime_bias,
                    regime_confidence=regime_conf,
                    ml_probability=ml_prob,
                    cleared=cleared_flag,
                    feature_row=feat_row,
                    entry_price=indicators['close'],
                    vix=vix,
                )
            except Exception as e:
                print(f'  [{ticker}] alert_log failed: {e}')

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
