"""
Alert Logging — Production Training Data Capture

Every time a watchlist stock scores >= SIGNAL_WATCH_THRESHOLD during a Phase 1
polling cycle, we log a full feature snapshot to Logs/alerts.parquet.

This builds an ongoing dataset that mirrors the LIVE inference distribution
exactly — the same bar windows, the same indicator computations, the same
regime context. Periodic retraining merges this with the historical Alpaca
backfill to grow the training set with real-world data.

Outcome columns (tp_hit, sl_hit, exit_reason, max_favorable, max_adverse) are
left null at write time and filled in by a separate backfill job once the
5-day forward window has elapsed.

Schema:
  alert_id           : monotonically-increasing int (per file)
  timestamp_utc      : ISO timestamp the alert fired
  timestamp_et       : same, but in ET
  symbol             : ticker
  base_score         : rule-based score 0-135
  regime_multiplier  : 0.90 / 1.00 / 1.10
  final_score        : base * regime
  regime_bias        : bullish / neutral / bearish
  regime_confidence  : 0-1
  ml_probability     : P(TP_hit) from the model (NaN if model absent)
  ml_threshold_pass  : 1 if ml_probability >= ML_CONFIDENCE_THRESHOLD
  cleared            : 1 if Phase 0 cleared this ticker for trading today
  would_have_traded  : 1 if all gates passed (score, clearance, ML)
  vix                : current VIX
  entry_price        : close at alert time (= what entry would have been)
  <all FEATURE_COLS> : 26 ML features as numeric columns
  tp_hit             : filled by backfill — 1 if +5% reached within 5 days
  sl_hit             : filled by backfill — 1 if -3.5% reached first
  max_favorable      : filled by backfill — max % gain within window
  max_adverse        : filled by backfill — max % drawdown within window
  exit_reason        : 'tp' / 'sl' / 'timeout' / NaN
  resolved_at        : timestamp when outcome was determined
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOGS_DIR, SIGNAL_WATCH_THRESHOLD, ML_CONFIDENCE_THRESHOLD
from ml.features import FEATURE_COLS

ALERTS_PATH = LOGS_DIR / 'alerts.parquet'
ET = pytz.timezone('America/New_York')

# Columns filled in by the backfill job, not at write time
_OUTCOME_COLS = [
    'tp_hit', 'sl_hit', 'max_favorable', 'max_adverse',
    'exit_reason', 'resolved_at',
]


def _alert_schema_columns() -> list:
    """Canonical column order for the alerts parquet."""
    return [
        'alert_id', 'timestamp_utc', 'timestamp_et', 'symbol',
        'base_score', 'regime_multiplier', 'final_score',
        'regime_bias', 'regime_confidence',
        'ml_probability', 'ml_threshold_pass',
        'cleared', 'would_have_traded',
        'vix', 'entry_price',
        *FEATURE_COLS,
        *_OUTCOME_COLS,
    ]


def _load_existing() -> pd.DataFrame:
    """Load existing alerts.parquet, or return empty DataFrame with correct schema."""
    if ALERTS_PATH.exists():
        try:
            return pd.read_parquet(ALERTS_PATH)
        except Exception as e:
            print(f'[alert_log] Failed to read {ALERTS_PATH}: {e} — starting fresh')
    return pd.DataFrame(columns=_alert_schema_columns())


def log_alert(
    *,
    symbol: str,
    timestamp,
    base_score: float,
    regime_multiplier: float,
    final_score: float,
    regime_bias: str,
    regime_confidence: float,
    ml_probability: Optional[float],
    cleared: bool,
    feature_row: dict,
    entry_price: float,
    vix: float = 0.0,
) -> Optional[int]:
    """
    Append a single alert row to Logs/alerts.parquet.

    Only fires when final_score >= SIGNAL_WATCH_THRESHOLD — alerts below that
    threshold are not informative for training (the rule-based system would
    never consider them anyway).

    Returns the alert_id of the written row, or None if the alert was below
    threshold and skipped.
    """
    if final_score < SIGNAL_WATCH_THRESHOLD:
        return None

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_existing()
    next_id  = (int(existing['alert_id'].max()) + 1) if len(existing) else 1

    # Normalize timestamp to both UTC and ET for downstream tools
    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
        ts_utc = pd.Timestamp(timestamp).tz_convert('UTC')
        ts_et  = pd.Timestamp(timestamp).tz_convert(ET)
    else:
        ts_utc = pd.Timestamp(timestamp).tz_localize('UTC')
        ts_et  = ts_utc.tz_convert(ET)

    ml_prob_val   = float(ml_probability) if ml_probability is not None else np.nan
    ml_pass_val   = int(ml_prob_val >= ML_CONFIDENCE_THRESHOLD) if not np.isnan(ml_prob_val) else 0
    cleared_int   = int(bool(cleared))
    would_trade   = int(
        final_score >= 100               # BUY threshold
        and cleared_int
        and (np.isnan(ml_prob_val) or ml_pass_val)
    )

    row = {
        'alert_id':          next_id,
        'timestamp_utc':     ts_utc.isoformat(),
        'timestamp_et':      ts_et.isoformat(),
        'symbol':            symbol,
        'base_score':        float(base_score),
        'regime_multiplier': float(regime_multiplier),
        'final_score':       float(final_score),
        'regime_bias':       regime_bias,
        'regime_confidence': float(regime_confidence),
        'ml_probability':    ml_prob_val,
        'ml_threshold_pass': ml_pass_val,
        'cleared':           cleared_int,
        'would_have_traded': would_trade,
        'vix':               float(vix),
        'entry_price':       float(entry_price),
    }
    for col in FEATURE_COLS:
        row[col] = float(feature_row.get(col, 0.0))
    for col in _OUTCOME_COLS:
        row[col] = np.nan if col not in ('exit_reason',) else None

    new_row_df = pd.DataFrame([row], columns=_alert_schema_columns())
    combined   = pd.concat([existing, new_row_df], ignore_index=True)
    combined.to_parquet(ALERTS_PATH, index=False)

    print(f'  [alert_log] #{next_id} {symbol} score={final_score:.1f} '
          f'ml={ml_prob_val:.3f} would_trade={would_trade} -> {ALERTS_PATH.name}')
    return next_id


# ── Backfill: resolve outcomes once forward window has elapsed ──────────────

def backfill_outcomes(
    bars_lookup,
    tp_pct: float = 0.05,
    sl_pct: float = 0.035,
    timeout_bars: int = 65,
) -> int:
    """
    Walk through unresolved alerts and fill in the outcome columns by scanning
    forward in bars_lookup[symbol] for the first TP / SL touch.

    Parameters
    ----------
    bars_lookup  : dict {symbol: DataFrame} with at least 'high'/'low' columns
                   and a tz-aware DatetimeIndex matching alert timestamps.
    tp_pct       : forward return that counts as a TP touch (default 5%)
    sl_pct       : forward drawdown that counts as a SL touch (default 3.5%)
    timeout_bars : max forward bars to scan before declaring timeout

    Returns the number of rows newly resolved this call.
    """
    if not ALERTS_PATH.exists():
        return 0

    alerts = pd.read_parquet(ALERTS_PATH)
    if alerts.empty:
        return 0

    unresolved = alerts['exit_reason'].isna()
    if not unresolved.any():
        return 0

    resolved_count = 0
    for idx in alerts[unresolved].index:
        row = alerts.loc[idx]
        symbol = row['symbol']
        if symbol not in bars_lookup:
            continue

        df = bars_lookup[symbol]
        entry_ts = pd.Timestamp(row['timestamp_utc'])
        if df.index.tz is not None:
            entry_ts = entry_ts.tz_convert(df.index.tz)

        future = df[df.index > entry_ts].head(timeout_bars)
        if len(future) < 5:
            continue   # not enough forward data yet to resolve

        entry  = float(row['entry_price'])
        tp_lvl = entry * (1.0 + tp_pct)
        sl_lvl = entry * (1.0 - sl_pct)

        tp_idx = future.index[future['high'] >= tp_lvl]
        sl_idx = future.index[future['low']  <= sl_lvl]

        tp_first = tp_idx[0] if len(tp_idx) else None
        sl_first = sl_idx[0] if len(sl_idx) else None

        if tp_first is not None and (sl_first is None or tp_first < sl_first):
            exit_reason = 'tp'; resolved_at = tp_first
        elif sl_first is not None:
            exit_reason = 'sl'; resolved_at = sl_first
        elif len(future) >= timeout_bars:
            exit_reason = 'timeout'; resolved_at = future.index[-1]
        else:
            continue   # still pending

        max_fav  = float((future['high'].max() / entry) - 1)
        max_adv  = float((future['low'].min()  / entry) - 1)

        alerts.at[idx, 'tp_hit']        = int(exit_reason == 'tp')
        alerts.at[idx, 'sl_hit']        = int(exit_reason == 'sl')
        alerts.at[idx, 'max_favorable'] = max_fav
        alerts.at[idx, 'max_adverse']   = max_adv
        alerts.at[idx, 'exit_reason']   = exit_reason
        alerts.at[idx, 'resolved_at']   = pd.Timestamp(resolved_at).isoformat()
        resolved_count += 1

    if resolved_count > 0:
        alerts.to_parquet(ALERTS_PATH, index=False)
        print(f'[alert_log] Backfilled outcomes for {resolved_count} alerts')

    return resolved_count


def summary() -> dict:
    """
    Return a small dict summarising the alerts log (counts, resolved rate,
    TP hit rate among resolved). Useful for CLI status checks.
    """
    if not ALERTS_PATH.exists():
        return {'total': 0, 'resolved': 0, 'tp_rate': None}

    alerts = pd.read_parquet(ALERTS_PATH)
    resolved = alerts['exit_reason'].notna()
    return {
        'total':           len(alerts),
        'resolved':        int(resolved.sum()),
        'unresolved':      int((~resolved).sum()),
        'would_have_traded': int(alerts['would_have_traded'].sum()),
        'tp_rate':         float(alerts.loc[resolved, 'tp_hit'].mean())
                           if resolved.any() else None,
    }
