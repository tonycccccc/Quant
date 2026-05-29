"""
Backtest harness — compares strategy variants against SPY/QQQ buy-and-hold.

Three variants are run on the same bar-by-bar feature dataset:
  1. baseline       — buy-and-hold QQQ (and SPY for context)
  2. rule_only      — RATMB rule-based scoring with no ML gate
  3. rule_plus_ml   — RATMB + ML probability gate (using bundle.recommended_threshold)

For each variant we simulate:
  - One position per trade, sized via the RISK_PER_TRADE / equity formula
  - Bracket-style exits: TP at +5% / SL at -3.5% / timeout at 65 bars
  - Per-ticker re-entry only after the prior trade closes
  - All-cash between trades (no compounding of leverage)

Outputs:
  - Total return, Sharpe ratio, max drawdown
  - Win rate, average win, average loss, profit factor
  - Trade-by-trade ledger -> Logs/backtest_trades.csv

Usage:
  python Code/main.py backtest [--months 12] [--use-ml/--no-ml]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ML_FEATURES_PATH, ML_RAW_BARS_PATH, LOGS_DIR,
    SIGNAL_BUY_THRESHOLD, ML_CONFIDENCE_THRESHOLD,
    RISK_PER_TRADE, MAX_POSITIONS, MAX_STOCK_CONCENTRATION,
    HARD_STOP_LOSS_PCT, TAKE_PROFIT_PCT, TP1_PCT, TP1_SHARE_FRACTION,
    ML_LABEL_TIMEOUT_BARS, ML_LABEL_TP_PCT,
    WATCHLIST,
)


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:       str
    entry_time:   pd.Timestamp
    exit_time:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    shares:       int
    signal_score: float
    ml_prob:      float
    exit_reason:  str
    pnl:          float
    pnl_pct:      float


@dataclass
class BacktestResult:
    variant:       str
    trades:        list = field(default_factory=list)
    starting_equity: float = 10_000.0
    ending_equity:   float = 10_000.0
    benchmark_return_qqq: float = 0.0
    benchmark_return_spy: float = 0.0

    @property
    def total_return(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        if losses == 0:
            return float('inf') if wins > 0 else 0.0
        return wins / losses

    @property
    def sharpe(self) -> float:
        """
        Annualized Sharpe based on per-trade returns. Assumes trades are
        independent observations of strategy return; not a true daily-return
        Sharpe but the right ballpark for swing-trade comparisons.
        """
        if self.n_trades < 2:
            return 0.0
        rets = np.array([t.pnl_pct for t in self.trades])
        if rets.std(ddof=1) == 0:
            return 0.0
        # ~50 swing trades per year if we average 1 per week
        annualisation = np.sqrt(252 / 5)
        return float(rets.mean() / rets.std(ddof=1) * annualisation)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = [self.starting_equity]
        for t in sorted(self.trades, key=lambda x: x.exit_time):
            equity.append(equity[-1] + t.pnl)
        equity_arr = np.array(equity)
        running_max = np.maximum.accumulate(equity_arr)
        dd = (equity_arr - running_max) / running_max
        return float(dd.min())


# ── Forward outcome simulation per signal ─────────────────────────────────

def _simulate_trade(
    bars: pd.DataFrame,
    entry_idx: int,
    equity: float,
    tp_pct: float = ML_LABEL_TP_PCT,
    sl_pct: float = HARD_STOP_LOSS_PCT,
    timeout_bars: int = ML_LABEL_TIMEOUT_BARS,
    signal_score: float = 0.0,
    ml_prob: float = float('nan'),
    symbol: str = '',
) -> Trade:
    """
    Walk forward from entry_idx, return a Trade with outcome.

    Sizing uses RISK_PER_TRADE / equity / stop_distance. Bracket exits at
    TP / SL / timeout. The first barrier touched within timeout_bars decides
    the trade.
    """
    entry_price = float(bars['close'].iloc[entry_idx])
    tp_lvl      = entry_price * (1 + tp_pct)
    sl_lvl      = entry_price * (1 - sl_pct)

    end_idx = min(entry_idx + 1 + timeout_bars, len(bars))
    fwd_high = bars['high'].iloc[entry_idx + 1 : end_idx].to_numpy()
    fwd_low  = bars['low'].iloc[entry_idx + 1 : end_idx].to_numpy()

    tp_hits = np.where(fwd_high >= tp_lvl)[0]
    sl_hits = np.where(fwd_low  <= sl_lvl)[0]
    tp_first = int(tp_hits[0]) if len(tp_hits) else 10**9
    sl_first = int(sl_hits[0]) if len(sl_hits) else 10**9

    if tp_first < sl_first:
        exit_idx    = entry_idx + 1 + tp_first
        exit_price  = tp_lvl
        exit_reason = 'tp'
    elif sl_first < tp_first:
        exit_idx    = entry_idx + 1 + sl_first
        exit_price  = sl_lvl
        exit_reason = 'sl'
    else:
        exit_idx    = min(end_idx - 1, len(bars) - 1)
        exit_price  = float(bars['close'].iloc[exit_idx])
        exit_reason = 'timeout'

    stop_distance = entry_price - sl_lvl
    if stop_distance <= 0:
        shares = 0
    else:
        risk_dollars = equity * RISK_PER_TRADE
        shares = int(risk_dollars / stop_distance)
        max_dollars = equity * MAX_STOCK_CONCENTRATION
        if shares * entry_price > max_dollars:
            shares = int(max_dollars / entry_price)
    pnl     = shares * (exit_price - entry_price)
    pnl_pct = (exit_price - entry_price) / entry_price

    return Trade(
        symbol=symbol,
        entry_time=bars.index[entry_idx],
        exit_time=bars.index[exit_idx],
        entry_price=entry_price,
        exit_price=exit_price,
        shares=shares,
        signal_score=signal_score,
        ml_prob=ml_prob,
        exit_reason=exit_reason,
        pnl=pnl,
        pnl_pct=pnl_pct,
    )


# ── Variant runners ───────────────────────────────────────────────────────

def _benchmark_return(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Buy-and-hold % return between start and end timestamps."""
    if bars.empty:
        return 0.0
    in_window = bars[(bars.index >= start) & (bars.index <= end)]
    if len(in_window) < 2:
        return 0.0
    return float(in_window['close'].iloc[-1] / in_window['close'].iloc[0] - 1)


def run_variant(
    features: pd.DataFrame,
    raw_bars: pd.DataFrame,
    variant: str,
    ml_predict_fn=None,
    ml_threshold: float = ML_CONFIDENCE_THRESHOLD,
    starting_equity: float = 10_000.0,
) -> BacktestResult:
    """
    Walk through every bar in chronological order, fire trades when the
    variant's gating logic accepts the bar, advance equity per trade outcome.

    variant: 'rule_only' or 'rule_plus_ml'
    """
    from ml.features import compute_signal_score_col, FEATURE_COLS

    result = BacktestResult(variant=variant, starting_equity=starting_equity,
                            ending_equity=starting_equity)
    equity = starting_equity

    features = features.sort_index().copy()
    # Pre-compute primary score per row and attach as a column.
    # Multiple symbols share a timestamp, so .loc[ts] would return a Series —
    # safer to keep it row-aligned in the dataframe.
    if 'primary_score' not in features.columns:
        features['primary_score'] = compute_signal_score_col(features)

    open_until = {}    # symbol -> exit_time; new entries blocked until then

    rows = features.reset_index().to_records(index=False)
    for row in rows:
        ts = row['timestamp']
        sym = row['symbol']
        score = float(row['primary_score']) if row['primary_score'] is not None else 0.0
        if np.isnan(score):
            continue

        if score < SIGNAL_BUY_THRESHOLD:
            continue
        # Per-symbol exclusivity: don't open a new trade until the prior one closes
        if sym in open_until and ts < open_until[sym]:
            continue

        # ML gate
        ml_prob = float('nan')
        if variant == 'rule_plus_ml' and ml_predict_fn is not None:
            feature_row = {col: float(row[col]) for col in FEATURE_COLS if col in row.dtype.names}
            ml_prob = float(ml_predict_fn(feature_row))
            if ml_prob < ml_threshold:
                continue

        # Find raw bars for this symbol and the entry timestamp
        try:
            sym_bars = raw_bars.loc[sym]
        except (KeyError, TypeError):
            continue
        if ts not in sym_bars.index:
            continue
        entry_idx = sym_bars.index.get_loc(ts)
        if not isinstance(entry_idx, int):
            # In case of duplicate timestamps, take the first
            entry_idx = int(np.asarray(entry_idx).flat[0])

        trade = _simulate_trade(
            sym_bars, entry_idx, equity,
            signal_score=score, ml_prob=ml_prob, symbol=sym,
        )
        if trade.shares == 0:
            continue

        equity += trade.pnl
        result.trades.append(trade)
        open_until[sym] = trade.exit_time

    result.ending_equity = equity
    return result


# ── Public entry point ────────────────────────────────────────────────────

def run_backtest(use_ml: bool = True, starting_equity: float = 10_000.0) -> dict:
    """
    Full backtest: rule-only vs rule+ML vs SPY/QQQ buy-and-hold.

    Returns a dict of {variant_name: BacktestResult}.
    Writes Logs/backtest_trades.csv with every trade across variants.
    """
    print(f'\n[backtest] Loading features from {ML_FEATURES_PATH}')
    features = pd.read_parquet(ML_FEATURES_PATH)
    raw_bars = pd.read_parquet(ML_RAW_BARS_PATH)

    # Restrict backtest to the live watchlist (training-only symbols don't trade)
    features = features[features['symbol'].isin(WATCHLIST.keys())].copy()

    # Time window
    start = features.index.min()
    end   = features.index.max()
    print(f'[backtest] Window: {start.date()} -> {end.date()}  '
          f'({len(features):,} feature rows, {features["symbol"].nunique()} symbols)')

    # Benchmarks
    qqq_ret = _benchmark_return(raw_bars.loc['QQQ'], start, end) if 'QQQ' in raw_bars.index.get_level_values(0) else 0.0
    spy_ret = _benchmark_return(raw_bars.loc['SPY'], start, end) if 'SPY' in raw_bars.index.get_level_values(0) else 0.0
    print(f'[backtest] Benchmark — QQQ buy-and-hold: {qqq_ret:+.2%}  |  '
          f'SPY: {spy_ret:+.2%}')

    # Rule-only variant
    print(f'\n[backtest] Running rule_only variant...')
    rule_only = run_variant(features, raw_bars, variant='rule_only',
                              starting_equity=starting_equity)
    rule_only.benchmark_return_qqq = qqq_ret
    rule_only.benchmark_return_spy = spy_ret
    _print_result(rule_only)

    # Rule + ML variant
    results = {'rule_only': rule_only}
    if use_ml:
        try:
            from ml.predict import predict_success_prob, get_threshold, model_info
            if model_info() is None:
                print('\n[backtest] No trained model found — skipping rule_plus_ml variant')
            else:
                threshold = get_threshold()
                print(f'\n[backtest] Running rule_plus_ml variant (threshold={threshold:.3f})...')
                rule_plus_ml = run_variant(
                    features, raw_bars, variant='rule_plus_ml',
                    ml_predict_fn=predict_success_prob,
                    ml_threshold=threshold,
                    starting_equity=starting_equity,
                )
                rule_plus_ml.benchmark_return_qqq = qqq_ret
                rule_plus_ml.benchmark_return_spy = spy_ret
                _print_result(rule_plus_ml)
                results['rule_plus_ml'] = rule_plus_ml
        except Exception as e:
            print(f'[backtest] ML variant failed: {e}')

    # Write trade ledger
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / 'backtest_trades.csv'
    all_rows = []
    for name, res in results.items():
        for t in res.trades:
            all_rows.append({
                'variant':      name,
                'symbol':       t.symbol,
                'entry_time':   t.entry_time.isoformat(),
                'exit_time':    t.exit_time.isoformat(),
                'entry_price':  round(t.entry_price, 2),
                'exit_price':   round(t.exit_price, 2),
                'shares':       t.shares,
                'signal_score': round(t.signal_score, 1),
                'ml_prob':      round(t.ml_prob, 4) if not np.isnan(t.ml_prob) else '',
                'exit_reason':  t.exit_reason,
                'pnl':          round(t.pnl, 2),
                'pnl_pct':      round(t.pnl_pct, 4),
            })
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f'\n[backtest] Trade ledger -> {out_path}')

    return results


def _print_result(result: BacktestResult) -> None:
    print(f'  Variant:        {result.variant}')
    print(f'  Trades:         {result.n_trades}')
    print(f'  Win rate:       {result.win_rate:.1%}  '
          f'({result.n_wins} W / {result.n_trades - result.n_wins} L)')
    print(f'  Avg win:        ${result.avg_win:+.2f}')
    print(f'  Avg loss:       ${result.avg_loss:+.2f}')
    print(f'  Profit factor:  {result.profit_factor:.2f}')
    print(f'  Total return:   {result.total_return:+.2%}  '
          f'(${result.starting_equity:.0f} -> ${result.ending_equity:.2f})')
    print(f'  Sharpe (annl):  {result.sharpe:.2f}')
    print(f'  Max drawdown:   {result.max_drawdown:+.2%}')
    print(f'  vs QQQ:         {(result.total_return - result.benchmark_return_qqq):+.2%}')
    print(f'  vs SPY:         {(result.total_return - result.benchmark_return_spy):+.2%}')
