"""
Bottom & Reversal Analysis — where does the model think a downtrend ends?

Answers three questions for a ticker in a downtrend:
  1. Where are the meaningful support levels? (Fib, EMAs, prior lows, BB)
  2. Are any reversal signals firing right now?
  3. What's the model's best-case / worst-case bottom range?

This is DESCRIPTIVE, not a trading signal. The model doesn't buy on this
output — the technical rule score still needs to fire independently.

The value: for a "quality dip" candidate (fundamentals strong, technical
broken), this tells you WHERE to watch for a reversal setup to develop.

CLI: python Code/main.py analyze-bottom --ticker AVGO
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import ML_RAW_BARS_PATH, EMA_SHORT_PERIOD


# ── Support level identification ─────────────────────────────────────────

def _compute_support_levels(daily_df: pd.DataFrame, current_price: float) -> list:
    """
    Return sorted list of (level, source, distance_pct) tuples BELOW current price.
    Sources: 'bollinger', 'fib_618', 'fib_500', 'fib_382', 'fib_236',
             'ema_50', 'ema_100', 'ema_200', 'swing_low_60d', 'swing_low_180d'.
    """
    if daily_df is None or len(daily_df) < 30:
        return []

    close = daily_df['close']
    high  = daily_df['high']
    low   = daily_df['low']

    levels = []

    # Bollinger lower (20-day)
    if len(daily_df) >= 20:
        bb_ma  = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        if not pd.isna(bb_ma) and not pd.isna(bb_std):
            levels.append((float(bb_ma - 2*bb_std), 'Bollinger lower (20d)'))

    # EMAs (50, 100, 200 day)
    for period, label in [(50, '50-day EMA'), (100, '100-day EMA'), (200, '200-day EMA')]:
        if len(daily_df) >= period:
            ema = close.ewm(span=period, adjust=False).mean().iloc[-1]
            if not pd.isna(ema):
                levels.append((float(ema), label))

    # Fibonacci retracements from most recent significant swing high/low
    # Use 90-day window for reasonable swing
    window = min(90, len(daily_df))
    swing_high = float(high.iloc[-window:].max())
    swing_low  = float(low.iloc[-window:].min())
    swing_range = swing_high - swing_low
    if swing_range > 0:
        for ratio, label in [(0.236, 'Fib 23.6% retrace'),
                              (0.382, 'Fib 38.2% retrace'),
                              (0.500, 'Fib 50%   retrace'),
                              (0.618, 'Fib 61.8% retrace (golden)'),
                              (0.786, 'Fib 78.6% retrace')]:
            # Retracement from swing high (support levels below current)
            level = swing_high - ratio * swing_range
            levels.append((float(level), label))

    # Recent swing lows
    if len(daily_df) >= 60:
        levels.append((float(low.iloc[-60:].min()), '60-day swing low'))
    if len(daily_df) >= 180:
        levels.append((float(low.iloc[-180:].min()), '180-day swing low'))

    # Filter to levels BELOW current price (support), sort by distance
    supports = [(lvl, src, (lvl - current_price) / current_price)
                for lvl, src in levels if lvl < current_price]
    supports.sort(key=lambda x: -x[0])   # closest first (highest price)
    return supports


# ── Reversal signal detection ────────────────────────────────────────────

def _compute_reversal_signals(daily_df: pd.DataFrame) -> dict:
    """
    Check for classical reversal patterns in the latest daily bars.
    Returns dict of {signal_name: (fired: bool, note: str)}.
    """
    signals = {}
    if daily_df is None or len(daily_df) < 30:
        return signals

    close = daily_df['close']
    high  = daily_df['high']
    low   = daily_df['low']
    vol   = daily_df['volume'] if 'volume' in daily_df.columns else pd.Series([0]*len(daily_df), index=daily_df.index)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).fillna(50)

    latest_rsi = float(rsi.iloc[-1])
    signals['rsi_oversold'] = (
        latest_rsi < 30,
        f'RSI {latest_rsi:.1f} (< 30 = oversold)' if latest_rsi < 30
        else f'RSI {latest_rsi:.1f} (need < 30)'
    )

    # RSI bullish divergence: price makes new 20-day low, but RSI doesn't
    if len(close) >= 20:
        recent_price_low_idx = close.iloc[-20:].idxmin()
        recent_rsi_at_low    = rsi.loc[recent_price_low_idx]
        current_price        = close.iloc[-1]
        current_rsi          = rsi.iloc[-1]
        divergence = (current_price <= close.iloc[-20:].min() * 1.01 and
                      current_rsi > recent_rsi_at_low + 2)
        signals['rsi_divergence'] = (
            divergence,
            f'Price near 20d-low + RSI higher than at prior low' if divergence
            else 'no clear divergence'
        )

    # MACD histogram turning up (last 3 bars trending)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    if len(macd_hist) >= 3:
        h1, h2, h3 = float(macd_hist.iloc[-3]), float(macd_hist.iloc[-2]), float(macd_hist.iloc[-1])
        turning_up = h1 < h2 < h3 and h3 > h1   # 3 bars up (whether still negative or not)
        signals['macd_turning_up'] = (
            turning_up,
            f'MACD hist trending up: {h1:.3f} -> {h2:.3f} -> {h3:.3f}' if turning_up
            else f'MACD hist not confirming: {h1:.3f} -> {h2:.3f} -> {h3:.3f}'
        )

    # Volume capitulation (recent 5-day volume spike)
    if len(vol) >= 30:
        vol_ma = vol.rolling(20).mean()
        recent5_max_vol = float(vol.iloc[-5:].max())
        vol_ma_now      = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else 1
        spike_ratio = recent5_max_vol / vol_ma_now if vol_ma_now > 0 else 0
        # AND: that spike day was a red day (close < open)
        recent5 = daily_df.iloc[-5:]
        spike_idx = recent5['volume'].idxmax()
        was_red   = (recent5.loc[spike_idx, 'close'] < recent5.loc[spike_idx, 'open'])
        capitulation = spike_ratio > 2.0 and was_red
        signals['volume_capitulation'] = (
            capitulation,
            f'Vol spike {spike_ratio:.1f}x on red day' if capitulation
            else f'Recent max vol spike {spike_ratio:.1f}x (need > 2x on red day)'
        )

    # Higher low (last 3 lows show upward-trending lows)
    if len(low) >= 15:
        recent_lows = low.iloc[-15:].rolling(3).min().dropna()
        higher_low = bool(recent_lows.iloc[-1] > recent_lows.iloc[-6] > recent_lows.iloc[-11]) \
            if len(recent_lows) >= 11 else False
        signals['higher_low_pattern'] = (
            higher_low,
            'Higher lows forming' if higher_low else 'No higher low pattern yet'
        )

    # Reclaim above 20-day EMA
    ema20 = close.ewm(span=20, adjust=False).mean()
    if len(ema20) > 5:
        was_below = float(close.iloc[-5]) < float(ema20.iloc[-5])
        now_above = float(close.iloc[-1]) > float(ema20.iloc[-1])
        reclaim = was_below and now_above
        signals['ema20_reclaim'] = (
            reclaim,
            'Reclaimed 20-day EMA' if reclaim else 'Below 20-day EMA'
        )

    return signals


# ── Main analysis function ───────────────────────────────────────────────

def analyze_bottom(ticker: str) -> dict:
    """
    Full bottom & reversal analysis for a ticker.
    Prints report and returns dict with all computed values.
    """
    ticker = ticker.upper()

    # Load cached bars
    if not ML_RAW_BARS_PATH.exists():
        raise FileNotFoundError('No cached bars — run daily-update first')
    raw = pd.read_parquet(ML_RAW_BARS_PATH)
    if ticker not in raw.index.get_level_values(0):
        raise ValueError(f'{ticker} not in cache')

    df_30m = raw.loc[ticker]
    current_price = float(df_30m['close'].iloc[-1])

    # Resample to daily
    idx_et = df_30m.index.tz_convert('America/New_York') if df_30m.index.tz else df_30m.index
    d = df_30m.copy(); d.index = idx_et
    daily = d.resample('B').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),  close=('close', 'last'),
        volume=('volume', 'sum'),
    ).dropna(subset=['close'])

    if len(daily) < 30:
        raise ValueError(f'{ticker} has only {len(daily)} daily bars — need at least 30')

    # 90-day and all-time high/low context
    high_90d = float(daily['high'].iloc[-min(90, len(daily)):].max())
    low_90d  = float(daily['low'].iloc[-min(90, len(daily)):].min())
    pct_from_high = (current_price - high_90d) / high_90d

    supports = _compute_support_levels(daily, current_price)
    signals  = _compute_reversal_signals(daily)

    # Composite reversal score (0-100)
    n_fired = sum(1 for k, (fired, _) in signals.items() if fired)
    n_total = len(signals)
    reversal_score = (n_fired / n_total * 100) if n_total else 0

    # Verdict
    if reversal_score >= 70:
        verdict = ('🟢 STRONG REVERSAL SIGNAL', 'Multiple bottoming signals firing — high-probability turn')
    elif reversal_score >= 40:
        verdict = ('🟡 EARLY SIGNS OF REVERSAL',
                    'Some signals firing but need more confirmation before entry')
    elif reversal_score >= 20:
        verdict = ('🟠 WEAK — WAIT', 'Isolated signals; downtrend still intact')
    else:
        verdict = ('🔴 NO REVERSAL YET', 'Downtrend appears to be continuing — potential further downside')

    result = {
        'ticker':           ticker,
        'current_price':    current_price,
        'high_90d':         high_90d,
        'low_90d':          low_90d,
        'pct_from_high':    pct_from_high,
        'supports':         supports,
        'signals':          signals,
        'reversal_score':   reversal_score,
        'verdict':          verdict,
    }
    _print_bottom_report(result, daily)
    return result


def _print_bottom_report(r: dict, daily_df: pd.DataFrame) -> None:
    print(f'\n{"="*72}')
    print(f'  BOTTOM & REVERSAL ANALYSIS — {r["ticker"]}')
    print(f'{"="*72}')

    print(f'\n  Current price:     ${r["current_price"]:.2f}')
    print(f'  90-day high:       ${r["high_90d"]:.2f}  '
          f'({r["pct_from_high"]:+.2%} from current — '
          f'{"in correction" if r["pct_from_high"] < -0.10 else "close to highs"})')
    print(f'  90-day low:        ${r["low_90d"]:.2f}')

    # Support levels
    print(f'\n  KEY SUPPORT LEVELS (nearest first):')
    if not r['supports']:
        print(f'    (no computed support levels below current price)')
    else:
        for i, (level, source, dist_pct) in enumerate(r['supports'][:8], 1):
            marker = ' ⭐' if 'golden' in source or '200-day' in source or 'swing_low' in source.lower() else ''
            print(f'    {i}. ${level:.2f}   {source:<28}  ({dist_pct:+.2%}){marker}')

    # Reversal signals
    print(f'\n  REVERSAL SIGNALS (checking for bottoming):')
    for name, (fired, note) in r['signals'].items():
        mark = '✓' if fired else '✗'
        print(f'    {mark} {name.replace("_", " ").title():<26} {note}')

    print(f'\n  Reversal score: {r["reversal_score"]:.0f}/100')
    emoji_verdict, desc = r['verdict']
    print(f'  {emoji_verdict}')
    print(f'  {desc}')

    # Best-case / worst-case interpretation
    print(f'\n  MODEL INTERPRETATION:')
    if not r['supports']:
        print(f'    Insufficient historical data for a bottom estimate')
        return
    nearest = r['supports'][0]
    strongest = None
    for lvl, src, dist in r['supports']:
        if 'golden' in src or '200-day' in src or '61.8' in src:
            strongest = (lvl, src, dist)
            break
    if not strongest and len(r['supports']) > 2:
        strongest = r['supports'][2]

    print(f'    Nearest support:   ${nearest[0]:.2f}  ({nearest[1]}, {nearest[2]:+.2%})')
    if strongest:
        print(f'    Strongest support: ${strongest[0]:.2f}  ({strongest[1]}, {strongest[2]:+.2%})')

    if r['reversal_score'] >= 40:
        print(f'\n    Best case: bottom appears to be forming near ${nearest[0]:.2f}. '
              f'Watch for confirmed reversal signal before entry.')
    else:
        print(f'\n    Model does NOT see a confirmed bottom yet.')
        print(f'    Likely near-term support: ${nearest[0]:.2f} (first defensive level).')
        if strongest and strongest[0] < nearest[0]:
            print(f'    If ${nearest[0]:.2f} fails, next major support: ${strongest[0]:.2f}.')
        print(f'    Watch for reversal signals to develop over next 5-15 trading days.')
