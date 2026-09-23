"""
Feature ablation experiment — find the minimum feature set that captures
most of the ML model's predictive power.

Motivation:
  With 40 features on 50K training rows, the model risks overfitting. This
  experiment trains 6 cumulative feature subsets and reports:
    - CV precision (top-10%, top-5%)
    - OOS walk-forward mean return vs QQQ
    - Sharpe
  The winner is the smallest set that captures ~85% of the best result.
  Occam's razor: simpler models generalize better.

Waves (each includes prior wave + additions):
  A_minimal      : 3 features   — daily trend + volatility + return
  B_momentum     : 6 features   — + RSI + volume + cross-sectional rank
  C_vix          : 9 features   — + VIX term structure (top-3 by importance)
  D_iv_rank      : 10 features  — + realized-vol percentile
  E_intraday     : 15 features  — + VWAP + intraday timing + meta-feature
  F_kitchen_sink : 40 features  — all FEATURE_COLS (current state)

CLI: python Code/main.py ablate-features
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ML_FEATURES_PATH, ML_RAW_BARS_PATH,
    ML_SIGNAL_SCORE_THRESHOLD, WATCHLIST, LOGS_DIR,
)
from ml.features import FEATURE_COLS, compute_signal_score_col

# ── Cumulative feature waves ─────────────────────────────────────────────
_WAVE_A = ['d_close_ema20_ratio', 'd_atr_pct', 'd_return_20d']

_WAVE_B = _WAVE_A + ['d_rsi', 'd_vol_ratio', 'momentum_rank_20d']

_WAVE_C = _WAVE_B + ['vix_9d', 'vix_3m', 'vix_term_ratio']

_WAVE_D = _WAVE_C + ['iv_rank_proxy']

_WAVE_E = _WAVE_D + ['close_vwap_ratio', 'close_ema20_ratio',
                      'prior_5d_return', 'rs_vs_qqq', 'primary_score']

_WAVE_F = FEATURE_COLS   # kitchen sink

WAVES = {
    'A_minimal':      _WAVE_A,
    'B_momentum':     _WAVE_B,
    'C_vix':          _WAVE_C,
    'D_iv_rank':      _WAVE_D,
    'E_intraday':     _WAVE_E,
    'F_kitchen_sink': _WAVE_F,
}


def _train_one_wave(features_df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Train an in-memory model on the given feature subset.
    Returns (model, tuned_threshold).
    """
    from ml.train import build_model, _wrap_calibrated, compute_recommended_threshold

    y = features_df['label']
    X = features_df

    tp_rate = float(y.mean())
    base    = build_model(tp_rate=tp_rate)
    # Use just the wave's features (subset of FEATURE_COLS)
    model = _wrap_calibrated(base, X[feature_cols], y.to_numpy(), sample_weight=None)

    # Threshold tuned on holdout
    rec_thr, _, _ = compute_recommended_threshold(
        model, X.assign(**{c: X[c] for c in feature_cols}),
        y, target_precision=0.50,
    ) if False else (None, None, None)
    # compute_recommended_threshold uses FEATURE_COLS globally — reimplement
    # here for arbitrary feature_cols
    from sklearn.metrics import precision_score
    n   = len(X)
    cut = int(n * 0.8)
    y_prob = model.predict_proba(X.iloc[cut:][feature_cols])[:, 1]
    y_hold = y.iloc[cut:].to_numpy()
    rec_thr = 0.5
    for thr in np.linspace(0.1, 0.95, 86):
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() < max(20, len(y_hold) * 0.005):
            continue
        prec = precision_score(y_hold, y_pred, zero_division=0)
        if prec >= 0.50:
            rec_thr = float(thr)
            break
    return model, rec_thr


def _walk_forward_wave(features_df: pd.DataFrame, raw_bars: pd.DataFrame,
                        feature_cols: list, n_folds: int = 5,
                        test_months: int = 6, starting_equity: float = 10_000.0):
    """
    Walk-forward OOS for one feature wave. Fits a fresh model per fold and
    runs rule_plus_ml backtest. Returns aggregate metrics.
    """
    from ml.backtest import run_variant, _benchmark_return
    from ml.train import build_model, _wrap_calibrated

    labeled = features_df[features_df['label'].notna()].copy()
    labeled['label'] = labeled['label'].astype(int)
    signal_scores = labeled['primary_score'] if 'primary_score' in labeled.columns else compute_signal_score_col(labeled)
    labeled = labeled[signal_scores >= ML_SIGNAL_SCORE_THRESHOLD].copy()

    t_min = labeled.index.min()
    t_max = labeled.index.max()
    first_test_start = t_max - pd.DateOffset(months=n_folds * test_months)

    fold_metrics = []
    for fold_idx in range(n_folds):
        test_start = first_test_start + pd.DateOffset(months=fold_idx * test_months)
        test_end   = test_start       + pd.DateOffset(months=test_months)

        train_df = labeled[labeled.index < test_start]
        if len(train_df) < 2000:
            continue

        tp_rate = float(train_df['label'].mean())
        base    = build_model(tp_rate=tp_rate)
        model   = _wrap_calibrated(base, train_df[feature_cols], train_df['label'].to_numpy())

        # Tune threshold on holdout slice of training
        cut = int(len(train_df) * 0.8)
        y_prob = model.predict_proba(train_df.iloc[cut:][feature_cols])[:, 1]
        y_hold = train_df['label'].iloc[cut:].to_numpy()
        rec_thr = 0.5
        from sklearn.metrics import precision_score
        for thr in np.linspace(0.1, 0.95, 86):
            y_pred = (y_prob >= thr).astype(int)
            if y_pred.sum() < max(20, len(y_hold) * 0.005):
                continue
            prec = precision_score(y_hold, y_pred, zero_division=0)
            if prec >= 0.50:
                rec_thr = float(thr)
                break

        oos_feat = features_df[(features_df.index >= test_start) & (features_df.index < test_end)]
        oos_feat = oos_feat[oos_feat['symbol'].isin(WATCHLIST.keys())]

        qqq_ret = _benchmark_return(raw_bars.loc['QQQ'], test_start, test_end)

        def predict_fn(feature_row: dict) -> float:
            row_df = pd.DataFrame([{c: feature_row.get(c, 0.0) for c in feature_cols}])
            return float(model.predict_proba(row_df)[0][1])

        rule_ml = run_variant(oos_feat, raw_bars, variant='rule_plus_ml',
                                ml_predict_fn=predict_fn, ml_threshold=rec_thr,
                                starting_equity=starting_equity)
        fold_metrics.append({
            'fold':       fold_idx + 1,
            'threshold':  rec_thr,
            'qqq_ret':    qqq_ret,
            'ml_ret':     rule_ml.total_return,
            'ml_sharpe':  rule_ml.sharpe,
            'ml_win':     rule_ml.win_rate,
            'ml_trades':  rule_ml.n_trades,
            'ml_vs_qqq':  rule_ml.total_return - qqq_ret,
        })
    return fold_metrics


def run_ablation() -> dict:
    """
    Run the full 6-wave ablation and report aggregate metrics per wave.
    """
    print(f'\n[ablation] Loading features from {ML_FEATURES_PATH}')
    features_df = pd.read_parquet(ML_FEATURES_PATH)
    raw_bars    = pd.read_parquet(ML_RAW_BARS_PATH)

    results = {}
    for wave_name, feats in WAVES.items():
        # Deduplicate + ensure features are all in FEATURE_COLS
        feats_available = [f for f in feats if f in features_df.columns]
        if len(feats_available) != len(feats):
            missing = set(feats) - set(feats_available)
            print(f'  [{wave_name}] Warning: {len(missing)} features missing: {missing}')

        print(f'\n{"="*70}')
        print(f'  WAVE {wave_name}  ({len(feats_available)} features)')
        print(f'{"="*70}')
        print(f'  Features: {feats_available}')

        t0 = time.time()
        fold_metrics = _walk_forward_wave(features_df, raw_bars, feats_available)
        elapsed = time.time() - t0

        if not fold_metrics:
            print(f'  [{wave_name}] No folds completed')
            continue

        mean_ret     = float(np.mean([f['ml_ret'] for f in fold_metrics]))
        mean_vs_qqq  = float(np.mean([f['ml_vs_qqq'] for f in fold_metrics]))
        mean_sharpe  = float(np.mean([f['ml_sharpe'] for f in fold_metrics]))
        mean_win     = float(np.mean([f['ml_win'] for f in fold_metrics]))
        folds_beat   = sum(1 for f in fold_metrics if f['ml_vs_qqq'] > 0)
        stdev_ret    = float(np.std([f['ml_ret'] for f in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0

        print(f'\n  {wave_name} AGGREGATE ({len(fold_metrics)} folds, {elapsed:.0f}s):')
        print(f'    Mean return / fold: {mean_ret:+.2%}  (stdev {stdev_ret:.2%})')
        print(f'    Mean vs QQQ:        {mean_vs_qqq:+.2%}')
        print(f'    Mean win rate:      {mean_win:.1%}')
        print(f'    Mean Sharpe:        {mean_sharpe:.2f}')
        print(f'    Folds beat QQQ:     {folds_beat}/{len(fold_metrics)}')

        results[wave_name] = {
            'n_features':  len(feats_available),
            'mean_ret':    mean_ret,
            'mean_vs_qqq': mean_vs_qqq,
            'mean_sharpe': mean_sharpe,
            'mean_win':    mean_win,
            'stdev_ret':   stdev_ret,
            'folds_beat':  folds_beat,
            'n_folds':     len(fold_metrics),
            'features':    feats_available,
            'elapsed':     elapsed,
        }

    # ── Final comparison table ────────────────────────────────────────────
    print(f'\n{"="*76}')
    print(f'  ABLATION SUMMARY — sorted by Sharpe')
    print(f'{"="*76}')
    print(f'  {"Wave":<20} {"#Feat":>6} {"Return":>10} {"vs QQQ":>10} '
          f'{"Sharpe":>8} {"Win%":>7} {"BeatQQQ":>8}')
    sorted_waves = sorted(results.items(), key=lambda x: -x[1]['mean_sharpe'])
    for name, r in sorted_waves:
        print(f'  {name:<20} {r["n_features"]:>6} '
              f'{r["mean_ret"]:>+9.2%} {r["mean_vs_qqq"]:>+9.2%} '
              f'{r["mean_sharpe"]:>7.2f} {r["mean_win"]:>6.1%} '
              f'{r["folds_beat"]:>3}/{r["n_folds"]}')

    # Save results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / 'feature_ablation_results.csv'
    rows = [{'wave': k, **v, 'features': ','.join(v.pop('features'))}
            for k, v in [(k, dict(v)) for k, v in results.items()]]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\n[ablation] Full results -> {out_path}')

    return results
