"""
Pure-pandas technical indicator computation and signal scoring.
No TA-Lib dependency required.
"""
import pandas as pd
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD, ATR_PERIOD,
    VOLUME_MA_PERIOD, RESISTANCE_LOOKBACK, VWAP_HOLD_BARS,
    VOLUME_SPIKE_MULTIPLIER, BREAKOUT_VOLUME_MULTIPLIER,
    RSI_PERIOD, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    RSI_OVERBOUGHT,
)


# ── Low-level indicators ───────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI via EWM (alpha = 1/period)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    return 100 - (100 / (1 + rs)).fillna(100)


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST_PERIOD,
    slow: int = MACD_SLOW_PERIOD,
    signal: int = MACD_SIGNAL_PERIOD,
) -> dict:
    """Returns dict of Series: macd_line, signal_line, histogram."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {
        'macd_line':   macd_line,
        'signal_line': signal_line,
        'histogram':   macd_line - signal_line,
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP for the provided slice (should be today's bars only)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()


# ── Full indicator bundle ──────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> Optional[dict]:
    """
    Compute all RATMB indicators from a 30-min OHLCV DataFrame.
    Returns None when there are fewer than 55 bars (EMA-50 + warmup).
    Expects a DatetimeTZIndex so VWAP can be reset per day.
    """
    if len(df) < 55:
        return None

    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    ema20    = compute_ema(close, EMA_SHORT_PERIOD)
    ema50    = compute_ema(close, EMA_LONG_PERIOD)
    atr      = compute_atr(df, ATR_PERIOD)
    vol_ma   = volume.rolling(VOLUME_MA_PERIOD).mean()
    rsi_ser  = compute_rsi(close)
    macd_out = compute_macd(close)

    # VWAP — reset at the start of each calendar day
    if hasattr(df.index, 'date'):
        today    = df.index[-1].date()
        today_df = df[pd.DatetimeIndex(df.index).date == today]
    else:
        today_df = df.tail(13)   # fallback: ~1 session of 30-min bars

    vwap_series   = compute_vwap(today_df)
    current_vwap  = float(vwap_series.iloc[-1]) if len(vwap_series) else float(close.iloc[-1])

    # ATR contraction: current ATR vs value 10 bars ago
    atr_ref             = float(atr.iloc[-11]) if len(atr) > 10 else float(atr.iloc[0])
    atr_contraction_ratio = float(atr.iloc[-1]) / atr_ref if atr_ref > 0 else 1.0

    # Higher-highs / higher-lows: compare last 13 bars vs prior 13 bars (~1 session each)
    hh = (float(high.iloc[-13:].max()) > float(high.iloc[-26:-13].max())) if len(high) >= 26 else False
    hl = (float(low.iloc[-13:].min())  > float(low.iloc[-26:-13].min()))  if len(low)  >= 26 else False

    # Resistance: highest close in the RESISTANCE_LOOKBACK bars before the current bar
    resistance = float(close.iloc[-RESISTANCE_LOOKBACK - 1:-1].max()) if len(close) > RESISTANCE_LOOKBACK else float(close.max())

    # VWAP hold: last N bars all closed above VWAP
    vwap_hold = False
    if len(today_df) >= VWAP_HOLD_BARS and len(vwap_series) >= VWAP_HOLD_BARS:
        vwap_hold = bool(
            (today_df['close'].iloc[-VWAP_HOLD_BARS:].values >=
             vwap_series.iloc[-VWAP_HOLD_BARS:].values).all()
        )

    vol_avg = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else float(volume.mean())

    return {
        'close':                float(close.iloc[-1]),
        'high':                 float(high.iloc[-1]),
        'low':                  float(low.iloc[-1]),
        'volume':               float(volume.iloc[-1]),
        'ema20':                float(ema20.iloc[-1]),
        'ema50':                float(ema50.iloc[-1]),
        'atr':                  float(atr.iloc[-1]),
        'vwap':                 current_vwap,
        'volume_avg':           vol_avg,
        'atr_contraction_ratio': atr_contraction_ratio,
        'higher_highs':         hh,
        'higher_lows':          hl,
        'resistance':           resistance,
        'vwap_hold':            vwap_hold,
        'rsi':                  float(rsi_ser.iloc[-1]),
        'macd_line':            float(macd_out['macd_line'].iloc[-1]),
        'macd_histogram':       float(macd_out['histogram'].iloc[-1]),
    }


# ── Signal scoring ─────────────────────────────────────────────────────────

def score_signal(indicators: dict, rs_vs_qqq: float) -> tuple:
    """
    Score the setup on a 0–135 base scale.
    Component breakdown:
      trend_score       max 25   (EMA alignment + HH/HL structure)
      breakout_strength max 20   (close > resistance)
      volume_quality    max 20   (volume vs average)
      vwap_support      max 20   (price above VWAP + hold)
      relative_strength max 15   (5-day RS vs QQQ)
      rsi_quality       max 20   (RSI momentum zone; >80 = hard block → 0)
      macd_momentum     max 15   (histogram > 0 and/or macd_line > 0)
    Returns (base_score, components_dict).
    base_score is 0 when RSI >= RSI_OVERBOUGHT (overbought hard block).
    """
    c = indicators

    # ── RSI hard block ─────────────────────────────────────────────────────
    rsi = c.get('rsi', 50.0)
    if rsi >= RSI_OVERBOUGHT:
        return 0, {
            'trend_score': 0, 'breakout_strength': 0, 'volume_quality': 0,
            'vwap_support': 0, 'relative_strength': 0, 'rsi_quality': 0,
            'macd_momentum': 0, 'base_score': 0,
            'rsi_blocked': True, 'rsi': round(rsi, 1),
            'vol_ratio': 0.0, 'atr_contraction': round(c.get('atr_contraction_ratio', 1.0), 3),
        }

    # ── Trend (max 25) ─────────────────────────────────────────────────────
    trend = 0
    if c['close'] > c['ema20']:                        trend += 10
    if c['ema20'] > c['ema50']:                        trend += 10
    if c['higher_highs'] and c['higher_lows']:         trend += 5
    elif c['higher_highs'] or c['higher_lows']:        trend += 2

    # ── Breakout (max 20) ──────────────────────────────────────────────────
    breakout = 0
    if c['close'] > c['resistance']:
        breakout += 15
        if c['close'] > c['resistance'] * 1.005:      breakout += 5

    # ── Volume quality (max 20) ────────────────────────────────────────────
    vol_ratio    = c['volume'] / c['volume_avg'] if c['volume_avg'] > 0 else 1.0
    volume_score = 0
    if vol_ratio >= VOLUME_SPIKE_MULTIPLIER:           volume_score += 10
    if vol_ratio >= BREAKOUT_VOLUME_MULTIPLIER:        volume_score += 10

    # ── VWAP support (max 20) ──────────────────────────────────────────────
    vwap_score = 0
    if c['close'] > c['vwap']:                         vwap_score += 10
    if c['vwap_hold']:                                 vwap_score += 10

    # ── Relative strength vs QQQ (max 15) ─────────────────────────────────
    rs_score = 0
    if rs_vs_qqq > 0.00:                               rs_score += 7
    if rs_vs_qqq > 0.02:                               rs_score += 8

    # ── RSI quality (max 20) ──────────────────────────────────────────────
    # Sweet spot 55–74: confirms trend without being extended.
    # 75–79: caution zone, reduced score.  <50: weak momentum.
    rsi_score = 0
    if 55 <= rsi < 75:    rsi_score = 20
    elif 50 <= rsi < 55:  rsi_score = 12
    elif 75 <= rsi < RSI_OVERBOUGHT:  rsi_score = 8

    # ── MACD momentum (max 15) ────────────────────────────────────────────
    macd_hist = c.get('macd_histogram', 0.0)
    macd_line = c.get('macd_line', 0.0)
    macd_score = 0
    if macd_hist > 0:    macd_score += 8   # momentum accelerating (line > signal)
    if macd_line > 0:    macd_score += 7   # overall bullish trend (above zero)

    base_score = trend + breakout + volume_score + vwap_score + rs_score + rsi_score + macd_score

    return base_score, {
        'trend_score':       trend,
        'breakout_strength': breakout,
        'volume_quality':    volume_score,
        'vwap_support':      vwap_score,
        'relative_strength': rs_score,
        'rsi_quality':       rsi_score,
        'macd_momentum':     macd_score,
        'base_score':        base_score,
        'rsi_blocked':       False,
        'rsi':               round(rsi, 1),
        'vol_ratio':         round(vol_ratio, 3),
        'atr_contraction':   round(c['atr_contraction_ratio'], 3),
    }


def apply_regime_multiplier(base_score: float, regime_bias: str, confidence: float) -> float:
    """Scale base score by 0.90–1.10 depending on LLM regime output."""
    conf = min(max(confidence, 0.0), 1.0)
    if regime_bias == 'bullish':
        mult = 1.0 + (0.10 * conf)
    elif regime_bias == 'bearish':
        mult = 1.0 - (0.10 * conf)
    else:
        mult = 1.0
    return base_score * mult


# ── Relative strength ──────────────────────────────────────────────────────

def compute_rs_vs_qqq(stock_df: pd.DataFrame, qqq_df: pd.DataFrame,
                      days: int = 5) -> float:
    """5-day return of the stock minus the 5-day return of QQQ."""
    def _period_return(df: pd.DataFrame) -> float:
        closes = df['close']
        if len(closes) < 2:
            return 0.0
        if hasattr(df.index, 'date'):
            by_day = closes.resample('D').last().dropna()
            if len(by_day) >= 2:
                start = float(by_day.iloc[max(-days - 1, -len(by_day))])
                end   = float(by_day.iloc[-1])
                return (end / start - 1) if start > 0 else 0.0
        return float(closes.iloc[-1] / closes.iloc[0] - 1) if float(closes.iloc[0]) > 0 else 0.0

    return _period_return(stock_df) - _period_return(qqq_df)
