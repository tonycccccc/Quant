"""
Step 3 — Forward-looking outcome labels.

For each bar T, scan the next TIMEOUT_BARS bars and return:
  1  — TP price (+7%) was reached before SL price (-3.5%)
  0  — SL was reached first, or neither was reached within the timeout
  NaN — the last TIMEOUT_BARS rows (insufficient future data)

Uses numpy for the forward scan instead of a Python loop.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ML_LABEL_TP_PCT, HARD_STOP_LOSS_PCT, ML_LABEL_TIMEOUT_BARS


def generate_labels(
    df: pd.DataFrame,
    tp_pct: float = ML_LABEL_TP_PCT,
    sl_pct: float = HARD_STOP_LOSS_PCT,
    timeout_bars: int = ML_LABEL_TIMEOUT_BARS,
) -> pd.Series:
    """
    Generate forward-outcome labels for a single-stock OHLCV DataFrame.

    Parameters
    ----------
    df           : OHLCV DataFrame, must have 'high', 'low', 'close' columns
    tp_pct       : take-profit as a positive fraction (e.g. 0.07 for +7%)
    sl_pct       : stop-loss magnitude as a positive fraction (e.g. 0.035 for -3.5%)
    timeout_bars : maximum bars to look forward before labelling as 0 (timeout)

    Returns
    -------
    pd.Series with same index as df; dtype float64 (NaN for last timeout_bars rows)
    """
    close = df['close'].to_numpy(dtype=float)
    high  = df['high'].to_numpy(dtype=float)
    low   = df['low'].to_numpy(dtype=float)
    n     = len(close)

    labels = np.full(n, np.nan)

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

        # TP must arrive strictly before SL; tie → count as SL (conservative)
        labels[i] = 1.0 if tp_first < sl_first else 0.0

    # Last timeout_bars rows cannot have a complete forward window → NaN
    labels[max(0, n - timeout_bars) :] = np.nan

    return pd.Series(labels, index=df.index, name='label')


def attach_labels(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach forward labels to a feature DataFrame that already has
    'close_raw', 'high_raw', 'low_raw' columns and a 'symbol' column.

    Labels are computed per-symbol so forward scans don't cross stock boundaries.
    """
    parts = []
    for symbol, grp in feat_df.groupby('symbol'):
        ohlcv = pd.DataFrame({
            'close': grp['close_raw'],
            'high':  grp['high_raw'],
            'low':   grp['low_raw'],
        })
        lbl = generate_labels(ohlcv)
        grp = grp.copy()
        grp['label'] = lbl.values
        parts.append(grp)

    return pd.concat(parts).sort_index()
