"""
TP/SL/timeout grid search — find the optimal bracket parameters empirically.

Two utilities:
  1. analyze_exit_reasons() — quick histogram of TP vs SL vs timeout in the
     existing walk-forward trade ledger. Diagnoses whether the current
     bracket is even doing the work, or if timeout is dominating.
  2. run_tp_sl_grid()       — sweeps TP x SL x timeout combinations on the
     historical features dataset and reports profit factor / Sharpe / win
     rate / max drawdown per combo. Output is sorted by Sharpe.

This is rule-only (no ML retraining per combo — that would take hours).
Once we find the winning bracket, we update config and retrain the ML
gate on labels matching the new bracket.

CLI:
  python Code/main.py tune-bracket [--quick]
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LOGS_DIR, ML_FEATURES_PATH, ML_RAW_BARS_PATH,
    SIGNAL_BUY_THRESHOLD, RISK_PER_TRADE, MAX_STOCK_CONCENTRATION,
    WATCHLIST,
)


# ── Exit-reason histogram ──────────────────────────────────────────────────

def analyze_exit_reasons(csv_path: Path = None) -> dict:
    """
    Read the walk-forward trade ledger and bucket exits by reason.

    Tells us whether TP/SL/timeout proportions justify tuning the bracket.
    If most exits are 'timeout', the current 5%/3.5% bracket is irrelevant
    and the real strategy is "hold for 5 days and exit at whatever".
    """
    if csv_path is None:
        csv_path = LOGS_DIR / 'backtest_walk_forward_trades.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'No walk-forward trades found at {csv_path}. '
                                 f'Run `python Code/main.py walk-forward-oos` first.')

    df = pd.read_csv(csv_path)
    print(f'\n[exit-reasons] Loaded {len(df):,} trades from {csv_path.name}')

    out = {}
    for variant in sorted(df['variant'].unique()):
        sub = df[df['variant'] == variant]
        reasons = Counter(sub['exit_reason'])
        total = len(sub)
        print(f'\n  Variant: {variant}  (n={total})')
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            avg_pnl = float(sub[sub['exit_reason'] == reason]['pnl'].mean())
            avg_pct = float(sub[sub['exit_reason'] == reason]['pnl_pct'].mean()) * 100
            print(f'    {reason:8s}  {count:5d}  ({pct:5.1f}%)  '
                  f'avg_pnl=${avg_pnl:+8.2f}  avg_pct={avg_pct:+5.2f}%')
        out[variant] = dict(reasons)

    return out


# ── TP/SL/timeout grid search ──────────────────────────────────────────────

def _simulate_one_trade(bars: pd.DataFrame, entry_idx: int, equity: float,
                          tp_pct: float, sl_pct: float, timeout_bars: int) -> dict:
    """
    Walk forward from entry_idx. Returns dict with pnl, exit_reason, etc.

    Sizing: RISK_PER_TRADE / stop_distance, with concentration cap.
    """
    entry_price = float(bars['close'].iloc[entry_idx])
    tp_lvl      = entry_price * (1 + tp_pct)
    sl_lvl      = entry_price * (1 - sl_pct)

    end_idx = min(entry_idx + 1 + timeout_bars, len(bars))
    if end_idx <= entry_idx + 1:
        return None
    fwd_high = bars['high'].iloc[entry_idx + 1 : end_idx].to_numpy()
    fwd_low  = bars['low'].iloc[entry_idx + 1 : end_idx].to_numpy()

    tp_hits = np.where(fwd_high >= tp_lvl)[0]
    sl_hits = np.where(fwd_low  <= sl_lvl)[0]
    tp_first = int(tp_hits[0]) if len(tp_hits) else 10**9
    sl_first = int(sl_hits[0]) if len(sl_hits) else 10**9

    if tp_first < sl_first:
        exit_price  = tp_lvl
        exit_reason = 'tp'
    elif sl_first < tp_first:
        exit_price  = sl_lvl
        exit_reason = 'sl'
    else:
        exit_price  = float(bars['close'].iloc[end_idx - 1])
        exit_reason = 'timeout'

    stop_distance = entry_price - sl_lvl
    if stop_distance <= 0:
        return None
    risk_dollars = equity * RISK_PER_TRADE
    shares = int(risk_dollars / stop_distance)
    max_dollars = equity * MAX_STOCK_CONCENTRATION
    if shares * entry_price > max_dollars:
        shares = int(max_dollars / entry_price)
    if shares <= 0:
        return None

    pnl = shares * (exit_price - entry_price)
    return {
        'pnl': pnl,
        'pnl_pct': (exit_price - entry_price) / entry_price,
        'exit_reason': exit_reason,
        'shares': shares,
    }


def _run_one_combo(features: pd.DataFrame, raw_bars: pd.DataFrame,
                    tp_pct: float, sl_pct: float, timeout_bars: int,
                    starting_equity: float = 10_000.0) -> dict:
    """
    Run rule-only backtest with one TP/SL/timeout combo across the full
    features window. Returns aggregate metrics.
    """
    equity = starting_equity
    open_until = {}
    pnls = []
    exit_reasons = []

    for row in features.itertuples():
        ts  = row.Index
        sym = row.symbol
        score = float(getattr(row, 'primary_score', 0.0) or 0.0)
        if np.isnan(score) or score < SIGNAL_BUY_THRESHOLD:
            continue
        if sym in open_until and ts < open_until[sym]:
            continue

        try:
            sym_bars = raw_bars.loc[sym]
        except (KeyError, TypeError):
            continue
        if ts not in sym_bars.index:
            continue
        entry_idx = sym_bars.index.get_loc(ts)
        if not isinstance(entry_idx, int):
            entry_idx = int(np.asarray(entry_idx).flat[0])

        trade = _simulate_one_trade(sym_bars, entry_idx, equity, tp_pct, sl_pct, timeout_bars)
        if trade is None:
            continue
        equity += trade['pnl']
        pnls.append(trade['pnl'])
        exit_reasons.append(trade['exit_reason'])
        # Block re-entry until forward window elapses
        exit_idx = min(entry_idx + 1 + timeout_bars, len(sym_bars) - 1)
        open_until[sym] = sym_bars.index[exit_idx]

    if not pnls:
        return {'tp': tp_pct, 'sl': sl_pct, 'timeout_bars': timeout_bars,
                'trades': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
                'profit_factor': 0, 'total_return': 0, 'sharpe': 0,
                'max_dd': 0, 'pct_tp': 0, 'pct_sl': 0, 'pct_timeout': 0}

    pnls_arr = np.array(pnls)
    wins   = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr <= 0]
    win_rate = len(wins) / len(pnls_arr)
    avg_win  = float(wins.mean())  if len(wins)  else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    pf = (float(wins.sum()) / abs(float(losses.sum()))) if len(losses) and losses.sum() != 0 else float('inf')

    # Sharpe per-trade -> annualized (~50 swing trades/yr)
    pnl_pcts = pnls_arr / starting_equity
    if pnl_pcts.std(ddof=1) > 0 and len(pnl_pcts) > 1:
        sharpe = float(pnl_pcts.mean() / pnl_pcts.std(ddof=1) * np.sqrt(252 / 5))
    else:
        sharpe = 0.0

    # Equity curve / max drawdown
    eq_curve = np.cumsum(pnls_arr) + starting_equity
    running_max = np.maximum.accumulate(eq_curve)
    max_dd = float(((eq_curve - running_max) / running_max).min())

    reasons = Counter(exit_reasons)
    n = len(exit_reasons)

    return {
        'tp':            tp_pct,
        'sl':            sl_pct,
        'timeout_bars':  timeout_bars,
        'timeout_days':  round(timeout_bars / 13, 1),
        'trades':        len(pnls_arr),
        'win_rate':      win_rate,
        'avg_win':       avg_win,
        'avg_loss':      avg_loss,
        'profit_factor': pf,
        'total_return':  (equity - starting_equity) / starting_equity,
        'sharpe':        sharpe,
        'max_dd':        max_dd,
        'pct_tp':        reasons.get('tp', 0) / n,
        'pct_sl':        reasons.get('sl', 0) / n,
        'pct_timeout':   reasons.get('timeout', 0) / n,
    }


def run_tp_sl_grid(quick: bool = False) -> pd.DataFrame:
    """
    Sweep TP × SL × timeout combinations on the full features dataset.
    Returns a DataFrame sorted by Sharpe ratio.

    quick=True uses a small grid (3×3×2 = 18 combos) for fast iteration.
    quick=False uses a fuller grid (5×4×3 = 60 combos).
    """
    print(f'\n[tune-bracket] Loading features from {ML_FEATURES_PATH}')
    features = pd.read_parquet(ML_FEATURES_PATH)
    raw_bars = pd.read_parquet(ML_RAW_BARS_PATH)

    features = features[features['symbol'].isin(WATCHLIST.keys())].sort_index()
    print(f'[tune-bracket] {len(features):,} feature rows over '
          f'{features["symbol"].nunique()} watchlist symbols')

    if quick:
        tp_grid      = [0.04, 0.07, 0.10]
        sl_grid      = [0.02, 0.035, 0.05]
        timeout_grid = [65, 130]   # 5 days, 10 days
    else:
        tp_grid      = [0.03, 0.05, 0.07, 0.10, 0.15]
        sl_grid      = [0.015, 0.025, 0.035, 0.05]
        timeout_grid = [65, 130, 195]   # 5, 10, 15 days

    combos = [(tp, sl, tb) for tp in tp_grid for sl in sl_grid for tb in timeout_grid]
    print(f'[tune-bracket] Sweeping {len(combos)} combinations...')

    rows = []
    for i, (tp, sl, tb) in enumerate(combos, 1):
        metrics = _run_one_combo(features, raw_bars, tp, sl, tb)
        rows.append(metrics)
        print(f'  [{i:3d}/{len(combos)}] TP={tp*100:4.1f}% SL={sl*100:4.1f}% '
              f'timeout={tb//13}d -> trades={metrics["trades"]:4d} '
              f'win={metrics["win_rate"]:.1%}  PF={metrics["profit_factor"]:.2f}  '
              f'ret={metrics["total_return"]:+.2%}  Sharpe={metrics["sharpe"]:.2f}  '
              f'DD={metrics["max_dd"]:+.2%}  '
              f'TP/SL/TO={metrics["pct_tp"]:.0%}/{metrics["pct_sl"]:.0%}/{metrics["pct_timeout"]:.0%}')

    df = pd.DataFrame(rows)
    df = df.sort_values('sharpe', ascending=False).reset_index(drop=True)

    # Save and print top results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / 'tune_bracket_results.csv'
    df.to_csv(out_path, index=False)
    print(f'\n[tune-bracket] Full results -> {out_path}')

    print(f'\n{"="*80}')
    print(f'  TOP 10 BY SHARPE')
    print(f'{"="*80}')
    print(f'  {"TP":>5} {"SL":>5} {"Time":>5} {"Trades":>7} {"Win":>5} {"PF":>5} '
          f'{"Return":>8} {"Sharpe":>7} {"MaxDD":>7} {"TP%":>5} {"SL%":>5} {"TO%":>5}')
    for _, r in df.head(10).iterrows():
        print(f'  {r["tp"]*100:>4.1f}% {r["sl"]*100:>4.1f}% {int(r["timeout_days"]):>4}d '
              f'{int(r["trades"]):>7} {r["win_rate"]:>4.1%} {r["profit_factor"]:>4.2f} '
              f'{r["total_return"]:>+7.2%} {r["sharpe"]:>6.2f} {r["max_dd"]:>+6.2%} '
              f'{r["pct_tp"]:>4.0%} {r["pct_sl"]:>4.0%} {r["pct_timeout"]:>4.0%}')

    # Also show the current baseline (5% / 3.5% / 5d) for comparison
    baseline = df[(df['tp'] == 0.05) & (df['sl'] == 0.035) & (df['timeout_bars'] == 65)]
    if len(baseline):
        print(f'\n  CURRENT BASELINE (5% / 3.5% / 5d):')
        r = baseline.iloc[0]
        print(f'  {r["tp"]*100:>4.1f}% {r["sl"]*100:>4.1f}% {int(r["timeout_days"]):>4}d '
              f'{int(r["trades"]):>7} {r["win_rate"]:>4.1%} {r["profit_factor"]:>4.2f} '
              f'{r["total_return"]:>+7.2%} {r["sharpe"]:>6.2f} {r["max_dd"]:>+6.2%} '
              f'{r["pct_tp"]:>4.0%} {r["pct_sl"]:>4.0%} {r["pct_timeout"]:>4.0%}')

    return df
