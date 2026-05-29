"""
Step 2 — Compute the ML feature matrix from raw OHLCV bars.

Multi-timeframe approach: every 30-min bar carries features from three
timeframes so the model can see both the intraday setup AND the broader
trend context that drives 3-5 day outcomes.

  30-min  — intraday momentum, VWAP structure, volume profile
  4-hour  — session-level trend, mid-term RSI and momentum (8-bar proxy)
  Daily   — macro trend alignment, monthly momentum, daily volatility regime

All features are price-scale invariant (ratios / percentages / booleans)
so the model generalises across stocks at different price levels.

FEATURE_COLS is the canonical list shared by training and inference.
Any change here must be reflected in indicators_to_feature_row() below.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMA_SHORT_PERIOD, EMA_LONG_PERIOD, ATR_PERIOD,
    VOLUME_MA_PERIOD, RESISTANCE_LOOKBACK, VWAP_HOLD_BARS,
    RSI_PERIOD, MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    ML_FEATURES_PATH, ML_RAW_BARS_PATH, MODELS_DIR,
    WATCHLIST, ML_EXTRA_TRAINING_SYMBOLS,
)

# ── Canonical feature list ─────────────────────────────────────────────────
# Must match what indicators_to_feature_row() produces for live inference.
FEATURE_COLS = [
    # ── 30-min timeframe ──────────────────────────────────────────────────
    'close_ema20_ratio',    # (close/EMA20) - 1     : intraday trend distance
    'ema20_ema50_ratio',    # (EMA20/EMA50) - 1     : 30-min trend structure
    'close_vwap_ratio',     # (close-VWAP)/VWAP     : intraday price vs VWAP
    'close_resist_ratio',   # (close-resist)/resist : breakout distance
    'atr_pct',              # ATR/close             : intraday volatility
    'atr_contraction',      # ATR/ATR[-10]          : compression ratio
    'rsi',                  # RSI-14 (0-100)
    'macd_hist_pct',        # MACD histogram/close  : momentum acceleration
    'macd_line_pct',        # MACD line/close       : trend bias
    'vwap_hold',            # 1 if last N bars all above VWAP
    'prior_1d_return',      # ~1 trading day return (13 30-min bars)
    'prior_5d_return',      # ~5 trading day return (65 30-min bars)
    'rs_vs_qqq',            # 5d stock return minus 5d QQQ return
    'hour',                 # decimal hour ET (9.5-16.0)
    'day_of_week',          # 0=Mon ... 4=Fri
    'spy_ema_aligned',      # 1 if SPY EMA20 > EMA50
    'qqq_ema_aligned',      # 1 if QQQ EMA20 > EMA50
    # ── 4-hour timeframe (8-bar proxy on 30-min) ──────────────────────────
    'h4_close_ema_ratio',   # (close/EMA8) - 1      : 4H trend position
    'h4_rsi',               # RSI-8 on 30-min       : 4H momentum
    'h4_return',            # pct_change(8)         : last 4 hours return
    # ── Daily timeframe (resampled from 30-min, prior day's data) ─────────
    'd_close_ema20_ratio',  # (daily_close/daily_EMA20) - 1 : daily trend
    'd_ema_aligned',        # 1 if daily EMA20 > daily EMA50
    'd_rsi',                # daily RSI-14
    'd_atr_pct',            # daily ATR/close       : daily volatility regime
    'd_vol_ratio',          # daily volume/daily volMA20
    'd_return_20d',         # 20-trading-day return : monthly momentum
]

_WARMUP_BARS = 70          # 30-min bars to drop at start (indicator warmup)
_INDEX_TICKERS = {'QQQ', 'SPY'}


# ── Signal-quality score (mirrors technicals.score_signal) ────────────────

def compute_signal_score_col(feat_df: pd.DataFrame) -> pd.Series:
    """
    Reconstruct the rule-based signal score vectorially from feature columns.
    Mirrors technicals.score_signal() + apply_regime_multiplier() exactly.

    Uses vol_ratio / higher_highs / higher_lows from the parquet (they are
    computed in build_feature_matrix but excluded from FEATURE_COLS).

    Returns a float Series with the same index as feat_df.
    Max possible: 135 x 1.10 ~= 148.
    """
    df = feat_df

    score = pd.Series(0.0, index=df.index)

    # Trend (max 25)
    score += (df['close_ema20_ratio'] > 0).astype(float) * 10
    score += (df['ema20_ema50_ratio'] > 0).astype(float) * 10
    hh = df['higher_highs'].fillna(0) if 'higher_highs' in df.columns else pd.Series(0.0, index=df.index)
    hl = df['higher_lows'].fillna(0)  if 'higher_lows'  in df.columns else pd.Series(0.0, index=df.index)
    score += ((hh > 0) & (hl > 0)).astype(float) * 5
    score += (((hh > 0) | (hl > 0)) & ~((hh > 0) & (hl > 0))).astype(float) * 2

    # Breakout (max 20)
    score += (df['close_resist_ratio'] > 0.000).astype(float) * 15
    score += (df['close_resist_ratio'] > 0.005).astype(float) * 5

    # Volume quality (max 20)
    vol = df['vol_ratio'].fillna(1.0) if 'vol_ratio' in df.columns else pd.Series(1.0, index=df.index)
    score += (vol >= 1.5).astype(float) * 10
    score += (vol >= 1.8).astype(float) * 10

    # VWAP support (max 20)
    score += (df['close_vwap_ratio'] > 0).astype(float) * 10
    score += (df['vwap_hold'] > 0).astype(float) * 10

    # Relative strength vs QQQ (max 15)
    score += (df['rs_vs_qqq'] > 0.00).astype(float) * 7
    score += (df['rs_vs_qqq'] > 0.02).astype(float) * 8

    # RSI quality (max 20)
    rsi = df['rsi']
    rsi_pts = pd.Series(0.0, index=df.index)
    rsi_pts = rsi_pts.where(~((rsi >= 55) & (rsi < 75)), 20.0)
    rsi_pts = rsi_pts.where(~((rsi >= 50) & (rsi < 55)), 12.0)
    rsi_pts = rsi_pts.where(~((rsi >= 75) & (rsi < 80)), 8.0)
    score += rsi_pts
    score = score.where(rsi < 80, 0.0)

    # MACD momentum (max 15)
    score += (df['macd_hist_pct'] > 0).astype(float) * 8
    score += (df['macd_line_pct'] > 0).astype(float) * 7

    # Regime multiplier
    spy = df['spy_ema_aligned'].fillna(0) if 'spy_ema_aligned' in df.columns else pd.Series(0.0, index=df.index)
    qqq = df['qqq_ema_aligned'].fillna(0) if 'qqq_ema_aligned' in df.columns else pd.Series(0.0, index=df.index)
    mult = pd.Series(1.0, index=df.index)
    mult = mult.where(~((spy > 0) & (qqq > 0)), 1.10)
    mult = mult.where(~((spy == 0) & (qqq == 0)), 0.90)

    return (score * mult).round(1)


# ── VWAP with daily ET reset ───────────────────────────────────────────────

def _vwap_daily_reset(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP resetting each calendar day in ET."""
    idx = df.index
    if idx.tz is None:
        idx_et = idx.tz_localize('America/New_York')
    else:
        idx_et = idx.tz_convert('America/New_York')

    tp    = (df['high'] + df['low'] + df['close']) / 3
    tpv   = tp * df['volume']
    dates = pd.Series(idx_et.date, index=df.index)

    cumtpv = tpv.groupby(dates).cumsum()
    cumvol  = df['volume'].groupby(dates).cumsum()
    return cumtpv / cumvol.where(cumvol > 0, other=float('nan'))


# ── 4-hour proxy features (rolling-window on 30-min bars) ─────────────────

def _compute_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4-hour proxy features using 8-bar rolling windows on 30-min bars.
    8 x 30-min bars = 4 trading hours.  No lookahead: only past bars used.
    """
    close = df['close']

    # 4H trend: EMA-8 on 30-min (responds to ~4 hours of price action)
    h4_ema = close.ewm(span=8, adjust=False).mean()
    h4_close_ema_ratio = close / h4_ema - 1

    # 4H RSI: RSI with period=8
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/8, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/8, adjust=False).mean()
    rs    = gain / loss.replace(0, float('nan'))
    h4_rsi = (100 - 100 / (1 + rs)).fillna(100)

    # 4H return: % change over last 8 bars (~4 hours)
    h4_return = close.pct_change(8)

    return pd.DataFrame({
        'h4_close_ema_ratio': h4_close_ema_ratio,
        'h4_rsi':             h4_rsi,
        'h4_return':          h4_return,
    }, index=df.index)


# ── Daily timeframe features (resampled from 30-min) ──────────────────────

def _compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 30-min OHLCV to daily bars and compute daily-scale indicators.

    Uses the PREVIOUS day's close (shift=1) so intraday bars never see
    same-day daily data — no lookahead bias.

    Returns a DataFrame aligned to df.index, filled with 0 where unavailable.
    """
    # Convert to ET for proper trading-day boundaries
    idx = df.index
    if idx.tz is None:
        idx_et = idx.tz_localize('America/New_York')
    else:
        idx_et = idx.tz_convert('America/New_York')

    df_et = df.copy()
    df_et.index = idx_et

    # Resample to business-day bars
    daily = df_et.resample('B').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),  close=('close', 'last'),
        volume=('volume', 'sum'),
    ).dropna(subset=['close'])

    _zero = pd.DataFrame(
        0.0, index=df.index,
        columns=['d_close_ema20_ratio', 'd_ema_aligned', 'd_rsi',
                 'd_atr_pct', 'd_vol_ratio', 'd_return_20d'],
    )
    if len(daily) < 5:
        return _zero

    close_d = daily['close']

    # Daily EMA20 / EMA50
    d_ema20 = close_d.ewm(span=EMA_SHORT_PERIOD, adjust=False).mean()
    d_ema50 = close_d.ewm(span=EMA_LONG_PERIOD,  adjust=False).mean()

    # Daily RSI-14
    delta_d  = close_d.diff()
    gain_d   = delta_d.clip(lower=0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss_d   = (-delta_d).clip(lower=0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs_d     = gain_d / loss_d.replace(0, float('nan'))
    d_rsi    = (100 - 100 / (1 + rs_d)).fillna(100)

    # Daily ATR
    tr_d = pd.concat([
        daily['high'] - daily['low'],
        (daily['high'] - close_d.shift()).abs(),
        (daily['low']  - close_d.shift()).abs(),
    ], axis=1).max(axis=1)
    d_atr = tr_d.ewm(span=ATR_PERIOD, adjust=False).mean()

    # Daily volume ratio
    d_vol_ma  = daily['volume'].rolling(VOLUME_MA_PERIOD).mean()
    d_vol_ratio = daily['volume'] / d_vol_ma.where(d_vol_ma > 0, other=daily['volume'])

    # 20-trading-day return (monthly momentum)
    d_return_20d = close_d.pct_change(20)

    daily_feat = pd.DataFrame({
        'd_close_ema20_ratio': close_d / d_ema20 - 1,
        'd_ema_aligned':       (d_ema20 > d_ema50).astype(float),
        'd_rsi':               d_rsi,
        'd_atr_pct':           d_atr / close_d,
        'd_vol_ratio':         d_vol_ratio,
        'd_return_20d':        d_return_20d,
    })

    # Shift by 1 day: each intraday bar uses YESTERDAY's daily data
    daily_feat = daily_feat.shift(1)

    # Align timezone back to df.index for reindexing
    if df.index.tz is not None:
        daily_feat.index = daily_feat.index.tz_convert(df.index.tz)
    else:
        daily_feat.index = daily_feat.index.tz_localize(None)

    result = daily_feat.reindex(df.index, method='ffill').fillna(0)
    result = result.replace([np.inf, -np.inf], 0)
    return result


# ── Per-stock feature matrix ───────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame,
                         qqq_df: pd.DataFrame = None,
                         spy_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Vectorized multi-timeframe indicator computation for one stock's history.

    Parameters
    ----------
    df      : OHLCV DataFrame, DatetimeIndex (UTC or ET)
    qqq_df  : QQQ bars (rs_vs_qqq + qqq_ema_aligned)
    spy_df  : SPY bars (spy_ema_aligned)

    Returns pd.DataFrame with FEATURE_COLS plus metadata columns.
    First _WARMUP_BARS rows are dropped; remaining NaNs forward-filled then 0.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    # ── 30-min EMAs ───────────────────────────────────────────────────────
    ema20 = close.ewm(span=EMA_SHORT_PERIOD, adjust=False).mean()
    ema50 = close.ewm(span=EMA_LONG_PERIOD,  adjust=False).mean()

    # ── 30-min ATR ────────────────────────────────────────────────────────
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # ── 30-min RSI ────────────────────────────────────────────────────────
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, float('nan'))
    rsi       = (100 - 100 / (1 + rs)).fillna(100)

    # ── 30-min MACD ───────────────────────────────────────────────────────
    ema_fast    = close.ewm(span=MACD_FAST_PERIOD,   adjust=False).mean()
    ema_slow    = close.ewm(span=MACD_SLOW_PERIOD,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    macd_hist   = macd_line - signal_line

    # ── Volume MA ─────────────────────────────────────────────────────────
    vol_ma = volume.rolling(VOLUME_MA_PERIOD).mean()

    # ── VWAP (daily ET reset) ─────────────────────────────────────────────
    vwap = _vwap_daily_reset(df)

    # ── ATR contraction ───────────────────────────────────────────────────
    atr_ref       = atr.shift(10)
    atr_contraction = atr / atr_ref.where(atr_ref > 0, other=atr)

    # ── Higher-highs / higher-lows ────────────────────────────────────────
    hh = (high.rolling(13).max() > high.shift(13).rolling(13).max()).astype(float)
    hl = (low.rolling(13).min()  > low.shift(13).rolling(13).min()).astype(float)

    # ── Resistance ────────────────────────────────────────────────────────
    resistance = close.shift(1).rolling(RESISTANCE_LOOKBACK).max()

    # ── VWAP hold ─────────────────────────────────────────────────────────
    above_vwap = (close >= vwap).astype(float)
    vwap_hold  = above_vwap.rolling(VWAP_HOLD_BARS).min().fillna(0)

    # ── 30-min momentum ───────────────────────────────────────────────────
    prior_1d = close.pct_change(13)
    prior_5d = close.pct_change(65)

    # ── RS vs QQQ ─────────────────────────────────────────────────────────
    if qqq_df is not None and len(qqq_df) > 0:
        qqq_close = qqq_df['close'].reindex(df.index, method='ffill')
        rs_qqq = prior_5d - qqq_close.pct_change(65)
    else:
        qqq_close = pd.Series(1.0, index=df.index)
        rs_qqq    = pd.Series(0.0, index=df.index)

    # ── SPY / QQQ macro EMA alignment ────────────────────────────────────
    if spy_df is not None and len(spy_df) > EMA_LONG_PERIOD:
        spy_c       = spy_df['close'].reindex(df.index, method='ffill')
        spy_ema_aligned = (spy_c.ewm(span=EMA_SHORT_PERIOD, adjust=False).mean() >
                           spy_c.ewm(span=EMA_LONG_PERIOD,  adjust=False).mean()).astype(float)
    else:
        spy_ema_aligned = pd.Series(0.0, index=df.index)

    if qqq_df is not None and len(qqq_df) > EMA_LONG_PERIOD:
        qqq_ema_aligned = (qqq_close.ewm(span=EMA_SHORT_PERIOD, adjust=False).mean() >
                           qqq_close.ewm(span=EMA_LONG_PERIOD,  adjust=False).mean()).astype(float)
    else:
        qqq_ema_aligned = pd.Series(0.0, index=df.index)

    # ── Time features ─────────────────────────────────────────────────────
    idx = df.index
    if idx.tz is None:
        idx_et = idx.tz_localize('America/New_York')
    else:
        idx_et = idx.tz_convert('America/New_York')
    hour        = pd.Series(idx_et.hour + idx_et.minute / 60, index=df.index)
    day_of_week = pd.Series(idx_et.dayofweek.astype(float),   index=df.index)

    # ── 4H proxy features ─────────────────────────────────────────────────
    feat_4h = _compute_4h_features(df)

    # ── Daily features ────────────────────────────────────────────────────
    feat_daily = _compute_daily_features(df)

    # ── Assemble feature DataFrame ────────────────────────────────────────
    feat = pd.DataFrame({
        # 30-min features
        'close_ema20_ratio':  close / ema20 - 1,
        'ema20_ema50_ratio':  ema20 / ema50 - 1,
        'close_vwap_ratio':   (close - vwap) / vwap.where(vwap > 0, other=close),
        'close_resist_ratio': (close - resistance) / resistance.where(resistance > 0, other=close),
        'atr_pct':            atr / close,
        'atr_contraction':    atr_contraction,
        'vol_ratio':          volume / vol_ma.where(vol_ma > 0, other=volume),
        'rsi':                rsi,
        'macd_hist_pct':      macd_hist / close,
        'macd_line_pct':      macd_line / close,
        'higher_highs':       hh,
        'higher_lows':        hl,
        'vwap_hold':          vwap_hold,
        'prior_1d_return':    prior_1d,
        'prior_5d_return':    prior_5d,
        'rs_vs_qqq':          rs_qqq,
        'hour':               hour,
        'day_of_week':        day_of_week,
        'spy_ema_aligned':    spy_ema_aligned,
        'qqq_ema_aligned':    qqq_ema_aligned,
        # 4H proxy features
        'h4_close_ema_ratio': feat_4h['h4_close_ema_ratio'],
        'h4_rsi':             feat_4h['h4_rsi'],
        'h4_return':          feat_4h['h4_return'],
        # Daily features
        'd_close_ema20_ratio': feat_daily['d_close_ema20_ratio'],
        'd_ema_aligned':       feat_daily['d_ema_aligned'],
        'd_rsi':               feat_daily['d_rsi'],
        'd_atr_pct':           feat_daily['d_atr_pct'],
        'd_vol_ratio':         feat_daily['d_vol_ratio'],
        'd_return_20d':        feat_daily['d_return_20d'],
        # metadata (not model inputs)
        'close_raw': close,
        'high_raw':  high,
        'low_raw':   low,
    }, index=df.index)

    # Drop warmup rows, forward-fill FEATURE_COLS, zero-fill remainder
    feat = feat.iloc[_WARMUP_BARS:].copy()
    feat[FEATURE_COLS] = feat[FEATURE_COLS].ffill().fillna(0)
    feat = feat.replace([np.inf, -np.inf], 0)
    return feat


# ── Cross-stock dataset builder ────────────────────────────────────────────

def build_all_features(raw_bars=None, save: bool = True) -> pd.DataFrame:
    """
    Build the full feature+label DataFrame for all symbols in raw_bars.

    Iterates every non-index symbol present in raw_bars, which includes
    both WATCHLIST stocks and ML_EXTRA_TRAINING_SYMBOLS fetched by collect.py.

    Parameters
    ----------
    raw_bars : MultiIndex (symbol, timestamp) DataFrame from collect.fetch_bars().
               If None, loaded from ML_RAW_BARS_PATH.
    save     : persist result to ML_FEATURES_PATH when True
    """
    if raw_bars is None:
        print(f'[features] Loading raw bars from {ML_RAW_BARS_PATH}')
        raw_bars = pd.read_parquet(ML_RAW_BARS_PATH)

    from ml.labels import attach_labels

    qqq_df = raw_bars.loc['QQQ'] if 'QQQ' in raw_bars.index.get_level_values(0) else None
    spy_df = raw_bars.loc['SPY'] if 'SPY' in raw_bars.index.get_level_values(0) else None

    # All non-index symbols present in raw_bars (watchlist + extra training)
    all_symbols = [s for s in raw_bars.index.get_level_values(0).unique()
                   if s not in _INDEX_TICKERS]

    all_parts = []
    for symbol in sorted(all_symbols):
        stock_df = raw_bars.loc[symbol]
        tag = '' if symbol in WATCHLIST else ' [train-only]'
        print(f'  [features] {symbol}{tag}: {len(stock_df):,} bars -> computing features...')

        feat = build_feature_matrix(stock_df, qqq_df=qqq_df, spy_df=spy_df)
        feat['symbol'] = symbol
        feat.index.name = 'timestamp'
        all_parts.append(feat)

    if not all_parts:
        raise ValueError('[features] No data found for any symbol')

    combined = pd.concat(all_parts).sort_index()
    combined  = attach_labels(combined)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(ML_FEATURES_PATH)
        labeled = combined['label'].notna().sum()
        tp_rate = combined.loc[combined['label'].notna(), 'label'].mean()
        print(f'[features] Saved {len(combined):,} rows ({labeled:,} labeled) | '
              f'TP rate {tp_rate:.1%} -> {ML_FEATURES_PATH}')

    return combined


# ── Inference adapter ──────────────────────────────────────────────────────

def indicators_to_feature_row(
    indicators: dict,
    bar_df,
    timestamp,
    rs_vs_qqq: float = 0.0,
) -> dict:
    """
    Convert a live compute_indicators() dict + raw bar window to one ML feature row.

    Computes 30-min, 4H proxy, and daily features from bar_df so that the
    inference distribution matches the training distribution exactly.

    Parameters
    ----------
    indicators : output of technicals.compute_indicators()
    bar_df     : raw 30-min OHLCV DataFrame for the past N days (15+ days ideal)
    timestamp  : current bar's timestamp (tz-aware preferred)
    rs_vs_qqq  : output of technicals.compute_rs_vs_qqq()
    """
    import pytz

    close = indicators['close']
    vwap  = indicators['vwap']
    res   = indicators['resistance']
    atr   = indicators['atr']

    close_c = close if close != 0 else 1.0
    vwap_c  = vwap  if vwap  != 0 else close_c
    res_c   = res   if res   != 0 else close_c

    # ── 30-min momentum returns ───────────────────────────────────────────
    bar_close = bar_df['close'] if bar_df is not None else None
    prior_1d = float(bar_close.iloc[-1] / bar_close.iloc[-14] - 1) \
               if bar_close is not None and len(bar_close) >= 14 else 0.0
    prior_5d = float(bar_close.iloc[-1] / bar_close.iloc[-66] - 1) \
               if bar_close is not None and len(bar_close) >= 66 else 0.0

    # ── Time features ─────────────────────────────────────────────────────
    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
        ts_et = timestamp.astimezone(pytz.timezone('America/New_York'))
    else:
        ts_et = timestamp
    hour        = ts_et.hour + ts_et.minute / 60.0 if hasattr(ts_et, 'hour') else 12.0
    day_of_week = float(ts_et.weekday()) if hasattr(ts_et, 'weekday') else 2.0

    # ── 4H proxy features ─────────────────────────────────────────────────
    h4_close_ema_ratio = h4_rsi_val = h4_ret = 0.0
    if bar_close is not None and len(bar_close) >= 9:
        h4_ema = bar_close.ewm(span=8, adjust=False).mean()
        h4_close_ema_ratio = float(bar_close.iloc[-1] / h4_ema.iloc[-1] - 1)

        delta_h = bar_close.diff()
        gain_h  = delta_h.clip(lower=0).ewm(alpha=1/8, adjust=False).mean()
        loss_h  = (-delta_h).clip(lower=0).ewm(alpha=1/8, adjust=False).mean()
        rs_h    = gain_h / loss_h.replace(0, float('nan'))
        h4_rsi_ser = (100 - 100 / (1 + rs_h)).fillna(100)
        h4_rsi_val = float(h4_rsi_ser.iloc[-1])

        h4_ret = float(bar_close.iloc[-1] / bar_close.iloc[-9] - 1)

    # ── Daily features ────────────────────────────────────────────────────
    d_close_ema20_ratio = d_ema_aligned = d_rsi_val = 0.0
    d_atr_pct_val = d_vol_ratio_val = d_return_20d_val = 0.0

    if bar_df is not None and len(bar_df) >= 30:
        try:
            daily_feat = _compute_daily_features(bar_df)
            # Use last row — already shifted to previous day's data
            last = daily_feat.iloc[-1]
            d_close_ema20_ratio = float(last['d_close_ema20_ratio'])
            d_ema_aligned       = float(last['d_ema_aligned'])
            d_rsi_val           = float(last['d_rsi'])
            d_atr_pct_val       = float(last['d_atr_pct'])
            d_vol_ratio_val     = float(last['d_vol_ratio'])
            d_return_20d_val    = float(last['d_return_20d'])
        except Exception:
            pass   # graceful fallback to 0 if daily resample fails

    return {
        # 30-min
        'close_ema20_ratio':  close / indicators['ema20'] - 1,
        'ema20_ema50_ratio':  indicators['ema20'] / indicators['ema50'] - 1,
        'close_vwap_ratio':   (close - vwap)  / vwap_c,
        'close_resist_ratio': (close - res)   / res_c,
        'atr_pct':            atr / close_c,
        'atr_contraction':    indicators['atr_contraction_ratio'],
        'rsi':                indicators['rsi'],
        'macd_hist_pct':      indicators['macd_histogram'] / close_c,
        'macd_line_pct':      indicators['macd_line'] / close_c,
        'vwap_hold':          float(indicators['vwap_hold']),
        'prior_1d_return':    prior_1d,
        'prior_5d_return':    prior_5d,
        'rs_vs_qqq':          rs_vs_qqq,
        'hour':               hour,
        'day_of_week':        day_of_week,
        'spy_ema_aligned':    float(indicators.get('spy_ema_aligned', 0.0)),
        'qqq_ema_aligned':    float(indicators.get('qqq_ema_aligned', 0.0)),
        # 4H proxy
        'h4_close_ema_ratio': h4_close_ema_ratio,
        'h4_rsi':             h4_rsi_val,
        'h4_return':          h4_ret,
        # Daily
        'd_close_ema20_ratio': d_close_ema20_ratio,
        'd_ema_aligned':       d_ema_aligned,
        'd_rsi':               d_rsi_val,
        'd_atr_pct':           d_atr_pct_val,
        'd_vol_ratio':         d_vol_ratio_val,
        'd_return_20d':        d_return_20d_val,
    }
