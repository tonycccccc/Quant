"""
Step 3 — Forward-looking outcome labels (Triple-Barrier Method).

For each bar T, scan the next TIMEOUT_BARS bars and return:
  1  — upper barrier (TP) was reached before lower barrier (SL)
  0  — lower barrier (SL) reached first, OR neither within the timeout
  NaN — last TIMEOUT_BARS rows (insufficient forward window)

Two labeling modes:

  - Fixed barriers (legacy):    +ML_LABEL_TP_PCT / -HARD_STOP_LOSS_PCT
  - ATR-scaled triple-barrier:  +k_tp*atr_pct / -k_sl*atr_pct  (Lopez de Prado)

The ATR-scaled mode adapts each bar's TP/SL distance to that stock's CURRENT
volatility regime — a 5% move means very different things on a low-vol blue
chip vs a high-vol momentum name. Comparable labels across regimes makes the
model far easier to train.

Each bar also gets a `t1` (label end timestamp) used downstream for
sample-uniqueness weighting (correlated overlapping labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ML_LABEL_TP_PCT, HARD_STOP_LOSS_PCT, ML_LABEL_TIMEOUT_BARS,
    ML_TRIPLE_BARRIER_ENABLED,
    ML_TB_TP_ATR_MULT, ML_TB_SL_ATR_MULT,
    ML_TB_MIN_TP_PCT, ML_TB_MAX_TP_PCT,
    ML_TB_MIN_SL_PCT, ML_TB_MAX_SL_PCT,
)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_labels_fixed(
    df: pd.DataFrame,
    tp_pct: float = ML_LABEL_TP_PCT,
    sl_pct: float = HARD_STOP_LOSS_PCT,
    timeout_bars: int = ML_LABEL_TIMEOUT_BARS,
) -> pd.DataFrame:
    """
    Legacy fixed-barrier labels. Returns DataFrame with columns:
      label : 0 / 1 / NaN
      t1    : timestamp of resolution (or last scanned bar) — needed for uniqueness
      ret   : forward return at resolution (signed)
    """
    close = df['close'].to_numpy(dtype=float)
    high  = df['high'].to_numpy(dtype=float)
    low   = df['low'].to_numpy(dtype=float)
    n     = len(close)
    idx   = df.index

    labels  = np.full(n, np.nan)
    t1_arr  = np.full(n, -1, dtype=np.int64)
    ret_arr = np.full(n, np.nan)

    for i in range(n - 1):
        entry  = close[i]
        tp_lvl = entry * (1.0 + tp_pct)
        sl_lvl = entry * (1.0 - sl_pct)

        end = min(i + 1 + timeout_bars, n)
        fwd_high = high[i + 1 : end]
        fwd_low  = low[i + 1 : end]

        tp_hits = np.where(fwd_high >= tp_lvl)[0]
        sl_hits = np.where(fwd_low  <= sl_lvl)[0]

        tp_first = int(tp_hits[0]) if len(tp_hits) else timeout_bars + 1
        sl_first = int(sl_hits[0]) if len(sl_hits) else timeout_bars + 1
        first    = min(tp_first, sl_first, end - i - 2)
        resolve_i = (i + 1) + first if first <= timeout_bars else min(end - 1, n - 1)

        labels[i]  = 1.0 if tp_first < sl_first else 0.0
        t1_arr[i]  = resolve_i
        ret_arr[i] = close[resolve_i] / entry - 1.0

    labels[max(0, n - timeout_bars) :] = np.nan

    t1_timestamps = pd.Series(
        [idx[i] if i >= 0 else pd.NaT for i in t1_arr], index=idx, name='t1',
    )
    return pd.DataFrame({
        'label': pd.Series(labels, index=idx),
        't1':    t1_timestamps,
        'ret':   pd.Series(ret_arr, index=idx),
    })


def generate_labels_triple_barrier(
    df: pd.DataFrame,
    daily_atr_pct: pd.Series,
    tp_mult: float = ML_TB_TP_ATR_MULT,
    sl_mult: float = ML_TB_SL_ATR_MULT,
    timeout_bars: int = ML_LABEL_TIMEOUT_BARS,
) -> pd.DataFrame:
    """
    Triple-barrier labels with ATR-scaled barriers.

    Each bar T gets:
      tp_pct = clip(tp_mult * daily_atr_pct[T],  MIN_TP, MAX_TP)
      sl_pct = clip(sl_mult * daily_atr_pct[T],  MIN_SL, MAX_SL)

    Then forward-scans timeout_bars to determine which barrier is hit first.

    daily_atr_pct must be the SAME index as df (or reindex-compatible).
    Use the `d_atr_pct` feature column for this.
    """
    close = df['close'].to_numpy(dtype=float)
    high  = df['high'].to_numpy(dtype=float)
    low   = df['low'].to_numpy(dtype=float)
    n     = len(close)
    idx   = df.index

    atr_pct = daily_atr_pct.reindex(idx).fillna(0.02).to_numpy(dtype=float)

    labels  = np.full(n, np.nan)
    t1_arr  = np.full(n, -1, dtype=np.int64)
    ret_arr = np.full(n, np.nan)
    tp_arr  = np.full(n, np.nan)
    sl_arr  = np.full(n, np.nan)

    for i in range(n - 1):
        entry = close[i]
        a     = atr_pct[i] if atr_pct[i] > 0 else 0.02

        tp_pct = _clip(tp_mult * a, ML_TB_MIN_TP_PCT, ML_TB_MAX_TP_PCT)
        sl_pct = _clip(sl_mult * a, ML_TB_MIN_SL_PCT, ML_TB_MAX_SL_PCT)

        tp_lvl = entry * (1.0 + tp_pct)
        sl_lvl = entry * (1.0 - sl_pct)

        end = min(i + 1 + timeout_bars, n)
        fwd_high = high[i + 1 : end]
        fwd_low  = low[i + 1 : end]

        tp_hits = np.where(fwd_high >= tp_lvl)[0]
        sl_hits = np.where(fwd_low  <= sl_lvl)[0]

        tp_first = int(tp_hits[0]) if len(tp_hits) else timeout_bars + 1
        sl_first = int(sl_hits[0]) if len(sl_hits) else timeout_bars + 1
        first    = min(tp_first, sl_first, end - i - 2)
        resolve_i = (i + 1) + first if first <= timeout_bars else min(end - 1, n - 1)

        labels[i]  = 1.0 if tp_first < sl_first else 0.0
        t1_arr[i]  = resolve_i
        ret_arr[i] = close[resolve_i] / entry - 1.0
        tp_arr[i]  = tp_pct
        sl_arr[i]  = sl_pct

    labels[max(0, n - timeout_bars) :] = np.nan

    t1_timestamps = pd.Series(
        [idx[i] if i >= 0 else pd.NaT for i in t1_arr], index=idx, name='t1',
    )
    return pd.DataFrame({
        'label':  pd.Series(labels, index=idx),
        't1':     t1_timestamps,
        'ret':    pd.Series(ret_arr, index=idx),
        'tp_pct': pd.Series(tp_arr,  index=idx),
        'sl_pct': pd.Series(sl_arr,  index=idx),
    })


# ── Legacy alias kept for backward compatibility with older callers ────────

def generate_labels(*args, **kwargs) -> pd.Series:
    """Backward-compatible wrapper that returns only the label Series."""
    out = generate_labels_fixed(*args, **kwargs)
    return out['label'].rename('label')


def attach_labels(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach forward labels to a feature DataFrame that already has
    'close_raw', 'high_raw', 'low_raw', 'd_atr_pct', and 'symbol' columns.

    Selection between fixed and triple-barrier modes is controlled by
    ML_TRIPLE_BARRIER_ENABLED. Labels are computed per-symbol so forward
    scans don't cross stock boundaries.

    Adds columns: label, t1, ret  (+ tp_pct, sl_pct when triple-barrier).
    """
    parts = []
    for symbol, grp in feat_df.groupby('symbol'):
        ohlcv = pd.DataFrame({
            'close': grp['close_raw'],
            'high':  grp['high_raw'],
            'low':   grp['low_raw'],
        })

        if ML_TRIPLE_BARRIER_ENABLED and 'd_atr_pct' in grp.columns:
            lbl_df = generate_labels_triple_barrier(
                ohlcv, daily_atr_pct=grp['d_atr_pct'],
            )
        else:
            lbl_df = generate_labels_fixed(ohlcv)

        grp = grp.copy()
        for col in lbl_df.columns:
            grp[col] = lbl_df[col].values
        parts.append(grp)

    return pd.concat(parts).sort_index()
