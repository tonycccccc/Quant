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


def run_oos_backtest(
    train_months: int = 18,
    starting_equity: float = 10_000.0,
) -> dict:
    """
    Out-of-sample backtest: train a FRESH model on the first `train_months`
    of the dataset, backtest on the remainder. The production model file is
    NOT modified.

    This is the honest read on whether the 58.6% in-sample win rate is real
    predictive power or memorization of the training distribution.

    Returns the same dict shape as run_backtest().
    """
    from ml.features import FEATURE_COLS, compute_signal_score_col
    from ml.train import build_model, _wrap_calibrated, compute_recommended_threshold
    from config import ML_SIGNAL_SCORE_THRESHOLD

    print(f'\n[oos-backtest] Loading features from {ML_FEATURES_PATH}')
    features = pd.read_parquet(ML_FEATURES_PATH)
    raw_bars = pd.read_parquet(ML_RAW_BARS_PATH)

    # Drop unlabeled rows and only train on bars that pass the signal filter
    labeled = features[features['label'].notna()].copy()
    labeled['label'] = labeled['label'].astype(int)
    if 'primary_score' in labeled.columns:
        signal_scores = labeled['primary_score']
    else:
        signal_scores = compute_signal_score_col(labeled)
    labeled = labeled[signal_scores >= ML_SIGNAL_SCORE_THRESHOLD].copy()

    # Temporal split
    t_min = labeled.index.min()
    t_max = labeled.index.max()
    cutoff = t_min + pd.DateOffset(months=train_months)
    if cutoff >= t_max:
        raise RuntimeError(
            f'train_months={train_months} >= total available history '
            f'({(t_max - t_min).days / 30.4:.1f} months). Lower train_months.'
        )

    train_df = labeled[labeled.index < cutoff]
    print(f'[oos-backtest] Train window: {t_min.date()} -> {cutoff.date()}  '
          f'({len(train_df):,} labeled rows)')

    # Train a fresh model (no calibration / weights — match production config)
    tp_rate = float(train_df['label'].mean())
    print(f'[oos-backtest] Training on {len(train_df):,} samples '
          f'(TP rate {tp_rate:.1%})...')
    base = build_model(tp_rate=tp_rate)
    model = _wrap_calibrated(base, train_df[FEATURE_COLS], train_df['label'].to_numpy())

    # Tune threshold on a holdout slice of training (NOT test) data
    rec_thr, top10_thr, notes = compute_recommended_threshold(
        model, train_df, train_df['label'], target_precision=0.50,
    )
    print(f'[oos-backtest] OOS-tuned threshold: {rec_thr:.3f}  ({notes})')

    # Restrict features dataframe to the OOS window only AND watchlist tickers
    oos_features = features[features.index >= cutoff].copy()
    oos_features = oos_features[oos_features['symbol'].isin(WATCHLIST.keys())]
    print(f'[oos-backtest] Test window:  {cutoff.date()} -> {t_max.date()}  '
          f'({len(oos_features):,} rows, {oos_features["symbol"].nunique()} symbols)')

    # Benchmarks over the OOS window only
    qqq_ret = _benchmark_return(raw_bars.loc['QQQ'], cutoff, t_max) if 'QQQ' in raw_bars.index.get_level_values(0) else 0.0
    spy_ret = _benchmark_return(raw_bars.loc['SPY'], cutoff, t_max) if 'SPY' in raw_bars.index.get_level_values(0) else 0.0
    print(f'[oos-backtest] OOS benchmark — QQQ: {qqq_ret:+.2%}  |  SPY: {spy_ret:+.2%}')

    def predict_fn(feature_row: dict) -> float:
        row_df = pd.DataFrame([{c: feature_row.get(c, 0.0) for c in FEATURE_COLS}])
        return float(model.predict_proba(row_df)[0][1])

    # Rule-only on OOS window
    print(f'\n[oos-backtest] Running rule_only variant on OOS window...')
    rule_only = run_variant(oos_features, raw_bars, variant='rule_only',
                              starting_equity=starting_equity)
    rule_only.benchmark_return_qqq = qqq_ret
    rule_only.benchmark_return_spy = spy_ret
    _print_result(rule_only)

    # Rule + ML on OOS window
    print(f'\n[oos-backtest] Running rule_plus_ml variant on OOS window '
          f'(threshold={rec_thr:.3f})...')
    rule_plus_ml = run_variant(
        oos_features, raw_bars, variant='rule_plus_ml',
        ml_predict_fn=predict_fn,
        ml_threshold=rec_thr,
        starting_equity=starting_equity,
    )
    rule_plus_ml.benchmark_return_qqq = qqq_ret
    rule_plus_ml.benchmark_return_spy = spy_ret
    _print_result(rule_plus_ml)

    # Write OOS trade ledger
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / 'backtest_oos_trades.csv'
    all_rows = []
    for name, res in (('rule_only', rule_only), ('rule_plus_ml', rule_plus_ml)):
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
        print(f'\n[oos-backtest] OOS trade ledger -> {out_path}')

    return {
        'rule_only': rule_only,
        'rule_plus_ml': rule_plus_ml,
        'oos_threshold': rec_thr,
        'train_months': train_months,
        'cutoff_date': cutoff.date(),
    }


def run_walk_forward_oos(
    n_folds: int = 4,
    test_months: int = 3,
    starting_equity: float = 10_000.0,
) -> dict:
    """
    Multi-fold out-of-sample backtest.

    Splits the dataset into n_folds rolling test windows. For each fold:
      1. Train a fresh model on ALL data before the test window
      2. Backtest rule-only and rule+ML on the test window
      3. Capture per-fold metrics

    Mean OOS performance across folds is a far more reliable read than a
    single 18/6 split — a single OOS window can land in a freakishly good
    or freakishly bad regime by accident.

    Returns dict with per-fold results AND aggregated mean/stdev metrics.
    """
    from ml.features import FEATURE_COLS, compute_signal_score_col
    from ml.train import build_model, _wrap_calibrated, compute_recommended_threshold
    from config import ML_SIGNAL_SCORE_THRESHOLD

    print(f'\n[walk-forward-oos] Loading features from {ML_FEATURES_PATH}')
    features = pd.read_parquet(ML_FEATURES_PATH)
    raw_bars = pd.read_parquet(ML_RAW_BARS_PATH)

    # Filter to labeled + signal-score-passing rows for training
    labeled = features[features['label'].notna()].copy()
    labeled['label'] = labeled['label'].astype(int)
    if 'primary_score' in labeled.columns:
        signal_scores = labeled['primary_score']
    else:
        signal_scores = compute_signal_score_col(labeled)
    labeled_filtered = labeled[signal_scores >= ML_SIGNAL_SCORE_THRESHOLD].copy()

    t_min = labeled_filtered.index.min()
    t_max = labeled_filtered.index.max()
    total_months = (t_max - t_min).days / 30.4
    print(f'[walk-forward-oos] Total span: {t_min.date()} -> {t_max.date()}  '
          f'({total_months:.1f} months, {len(labeled_filtered):,} filtered rows)')

    if total_months < n_folds * test_months + 6:
        raise RuntimeError(
            f'Not enough history for {n_folds} folds of {test_months} months. '
            f'Need >= {n_folds * test_months + 6} months, have {total_months:.1f}.'
        )

    # Test windows roll forward at test_months intervals; the LAST one ends at t_max
    first_test_start = t_max - pd.DateOffset(months=n_folds * test_months)
    fold_results = []

    for fold_idx in range(n_folds):
        test_start = first_test_start + pd.DateOffset(months=fold_idx * test_months)
        test_end   = test_start       + pd.DateOffset(months=test_months)

        train_df = labeled_filtered[labeled_filtered.index < test_start]
        if len(train_df) < 2000:
            print(f'  Fold {fold_idx+1}: skipped (only {len(train_df)} training rows)')
            continue

        oos_features = features[(features.index >= test_start) & (features.index < test_end)]
        oos_features = oos_features[oos_features['symbol'].isin(WATCHLIST.keys())]

        print(f'\n[walk-forward-oos] Fold {fold_idx+1}/{n_folds}: '
              f'train < {test_start.date()}  |  test [{test_start.date()} -> {test_end.date()})')
        print(f'  Train: {len(train_df):,} rows  |  Test feature window: {len(oos_features):,} rows')

        # Train fresh model
        tp_rate = float(train_df['label'].mean())
        base    = build_model(tp_rate=tp_rate)
        model   = _wrap_calibrated(base, train_df[FEATURE_COLS], train_df['label'].to_numpy())
        rec_thr, _, _ = compute_recommended_threshold(model, train_df, train_df['label'],
                                                       target_precision=0.50)

        # Benchmark on this fold's test window
        qqq_ret = _benchmark_return(raw_bars.loc['QQQ'], test_start, test_end) if 'QQQ' in raw_bars.index.get_level_values(0) else 0.0
        spy_ret = _benchmark_return(raw_bars.loc['SPY'], test_start, test_end) if 'SPY' in raw_bars.index.get_level_values(0) else 0.0

        def predict_fn(feature_row: dict) -> float:
            row_df = pd.DataFrame([{c: feature_row.get(c, 0.0) for c in FEATURE_COLS}])
            return float(model.predict_proba(row_df)[0][1])

        # Run both variants on this fold
        rule_only = run_variant(oos_features, raw_bars, variant='rule_only',
                                  starting_equity=starting_equity)
        rule_only.benchmark_return_qqq = qqq_ret
        rule_only.benchmark_return_spy = spy_ret

        rule_plus_ml = run_variant(
            oos_features, raw_bars, variant='rule_plus_ml',
            ml_predict_fn=predict_fn, ml_threshold=rec_thr,
            starting_equity=starting_equity,
        )
        rule_plus_ml.benchmark_return_qqq = qqq_ret
        rule_plus_ml.benchmark_return_spy = spy_ret

        print(f'  threshold={rec_thr:.3f}  '
              f'QQQ={qqq_ret:+.2%}  SPY={spy_ret:+.2%}')
        print(f'  rule_only    : trades={rule_only.n_trades:3d}  '
              f'win={rule_only.win_rate:.1%}  ret={rule_only.total_return:+.2%}  '
              f'vs_qqq={rule_only.total_return - qqq_ret:+.2%}')
        print(f'  rule_plus_ml : trades={rule_plus_ml.n_trades:3d}  '
              f'win={rule_plus_ml.win_rate:.1%}  ret={rule_plus_ml.total_return:+.2%}  '
              f'vs_qqq={rule_plus_ml.total_return - qqq_ret:+.2%}')

        fold_results.append({
            'fold':          fold_idx + 1,
            'test_start':    test_start.date(),
            'test_end':      test_end.date(),
            'threshold':     rec_thr,
            'qqq_return':    qqq_ret,
            'spy_return':    spy_ret,
            'rule_only':     rule_only,
            'rule_plus_ml':  rule_plus_ml,
        })

    # ── Aggregate across folds ────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  WALK-FORWARD OOS — AGGREGATE ({len(fold_results)} folds)')
    print(f'{"="*72}')

    def _agg(variant_key: str) -> dict:
        rets       = [f[variant_key].total_return for f in fold_results]
        win_rates  = [f[variant_key].win_rate     for f in fold_results]
        excess_qqq = [f[variant_key].total_return - f['qqq_return'] for f in fold_results]
        sharpes    = [f[variant_key].sharpe       for f in fold_results]
        return {
            'mean_return':      float(np.mean(rets)),
            'std_return':       float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
            'mean_win_rate':    float(np.mean(win_rates)),
            'mean_excess_qqq':  float(np.mean(excess_qqq)),
            'win_excess_folds': sum(1 for x in excess_qqq if x > 0),
            'mean_sharpe':      float(np.mean(sharpes)),
            'n_folds':          len(fold_results),
        }

    rule_agg = _agg('rule_only')
    ml_agg   = _agg('rule_plus_ml')
    qqq_mean = float(np.mean([f['qqq_return'] for f in fold_results]))

    print(f'\n  Mean QQQ buy-and-hold per fold: {qqq_mean:+.2%}')
    print(f'\n  Rule-only:')
    print(f'    Mean return / fold: {rule_agg["mean_return"]:+.2%}  '
          f'(stdev {rule_agg["std_return"]:.2%})')
    print(f'    Mean win rate:      {rule_agg["mean_win_rate"]:.1%}')
    print(f'    Mean vs QQQ:        {rule_agg["mean_excess_qqq"]:+.2%}')
    print(f'    Folds beat QQQ:     {rule_agg["win_excess_folds"]}/{rule_agg["n_folds"]}')
    print(f'    Mean Sharpe:        {rule_agg["mean_sharpe"]:.2f}')

    print(f'\n  Rule + ML:')
    print(f'    Mean return / fold: {ml_agg["mean_return"]:+.2%}  '
          f'(stdev {ml_agg["std_return"]:.2%})')
    print(f'    Mean win rate:      {ml_agg["mean_win_rate"]:.1%}')
    print(f'    Mean vs QQQ:        {ml_agg["mean_excess_qqq"]:+.2%}')
    print(f'    Folds beat QQQ:     {ml_agg["win_excess_folds"]}/{ml_agg["n_folds"]}')
    print(f'    Mean Sharpe:        {ml_agg["mean_sharpe"]:.2f}')

    # Save per-fold trades
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / 'backtest_walk_forward_trades.csv'
    rows = []
    for f in fold_results:
        for variant_key in ('rule_only', 'rule_plus_ml'):
            for t in f[variant_key].trades:
                rows.append({
                    'fold':         f['fold'],
                    'variant':      variant_key,
                    'symbol':       t.symbol,
                    'entry_time':   t.entry_time.isoformat(),
                    'exit_time':    t.exit_time.isoformat(),
                    'shares':       t.shares,
                    'pnl':          round(t.pnl, 2),
                    'pnl_pct':      round(t.pnl_pct, 4),
                    'exit_reason':  t.exit_reason,
                })
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f'\n[walk-forward-oos] Trade ledger -> {out_path}')

    return {
        'folds': fold_results,
        'rule_only_agg':    rule_agg,
        'rule_plus_ml_agg': ml_agg,
    }


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
