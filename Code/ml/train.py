"""
Step 4 — Model training and walk-forward cross-validation.

Walk-forward validation ensures training data always precedes test data,
preventing the lookahead bias that random splits introduce in time-series.

Methodology (López de Prado "Advances in Financial Machine Learning"):
  - Triple-barrier labels with ATR-scaled barriers (see ml/labels.py)
  - Sample-uniqueness weights for overlapping label periods
  - Isotonic probability calibration (CalibratedClassifierCV)
  - Walk-forward CV with expanding train window
  - Signal-quality filter so training distribution matches inference

Model priority:
  1. LightGBM (fast, best tabular performance)
  2. scikit-learn RandomForest (fallback if LightGBM not installed)

Trained model is saved to Models/quant_model.pkl as a ModelBundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ML_FEATURES_PATH, ML_MODEL_PATH, MODELS_DIR,
    ML_MIN_PRECISION, ML_SIGNAL_SCORE_THRESHOLD,
    ML_CALIBRATION_METHOD, ML_CALIBRATION_CV,
    ML_CONFIDENCE_THRESHOLD, ML_SAMPLE_WEIGHTS_ENABLED,
)
from ml.features import FEATURE_COLS, compute_signal_score_col


# ── Model bundle serialised to disk ───────────────────────────────────────

@dataclass
class ModelBundle:
    model:        Any
    feature_cols: list
    trained_at:   str
    n_samples:    int
    tp_rate:      float
    cv_metrics:   list = field(default_factory=list)
    calibrated:   bool = False
    method:       str  = 'lightgbm'


# ── Model factory ─────────────────────────────────────────────────────────

def build_model(tp_rate: float = 0.5):
    """
    Return an untrained classifier. LightGBM preferred; falls back to
    scikit-learn RandomForest when LightGBM is not installed.
    """
    neg_rate   = 1.0 - tp_rate
    pos_weight = neg_rate / tp_rate if tp_rate > 0 else 1.0

    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=30,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )


# ── Sample-uniqueness weights (López de Prado, Ch. 4) ──────────────────────

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-sample weights inversely proportional to label concurrency.

    When two labels' forward [t, t1] intervals overlap, their outcomes are
    correlated — both observe the same future price path. Down-weighting
    overlapping samples prevents redundant information from dominating
    training. Implemented with a sweep-line for O(n log n) complexity.

    Requires 't1' (label-end timestamp) and 'symbol' columns. Per-symbol so
    concurrency is only counted within the same instrument.
    """
    if 't1' not in df.columns or 'symbol' not in df.columns:
        return np.ones(len(df))

    weights = np.zeros(len(df))
    pos = 0
    for _, grp in df.groupby('symbol', sort=False):
        n = len(grp)
        if n == 0:
            continue
        starts = pd.DatetimeIndex(grp.index).astype('int64').to_numpy()
        ends   = pd.to_datetime(grp['t1'].values).astype('int64')

        events = np.concatenate([starts, ends + 1])
        deltas = np.concatenate([np.ones_like(starts), -np.ones_like(ends)])
        order  = np.argsort(events, kind='stable')
        ev_sorted = events[order]
        dl_sorted = deltas[order]
        cum = np.cumsum(dl_sorted)

        idx_start = np.searchsorted(ev_sorted, starts, side='right') - 1
        idx_end   = np.searchsorted(ev_sorted, ends,   side='right') - 1
        conc_start = np.clip(cum[idx_start], 1, None)
        conc_end   = np.clip(cum[idx_end],   1, None)
        avg_conc   = (conc_start + conc_end) / 2.0
        w = 1.0 / np.maximum(avg_conc, 1.0)
        weights[pos : pos + n] = w
        pos += n

    if weights.sum() > 0:
        weights = weights * (len(weights) / weights.sum())
    else:
        weights = np.ones(len(df))
    return weights


# ── Calibration wrapper ────────────────────────────────────────────────────

def _wrap_calibrated(model, X_train, y_train, sample_weight=None):
    """Wrap `model` in CalibratedClassifierCV (isotonic) and fit."""
    if not ML_CALIBRATION_METHOD:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_train, y_train)
        return model

    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=ML_CALIBRATION_METHOD,
        cv=ML_CALIBRATION_CV,
    )
    try:
        calibrated.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        calibrated.fit(X_train, y_train)
    return calibrated


# ── Walk-forward splits ────────────────────────────────────────────────────

def walk_forward_splits(X: pd.DataFrame, n_splits: int = 4, test_months: int = 3):
    """
    Yield (train_mask, test_mask) boolean arrays in temporal order.
    Training window expands; test window slides forward by test_months.
    """
    ts    = X.index
    t_min = ts.min()
    t_max = ts.max()
    total = (t_max - t_min).days / 30.4

    if total < n_splits * test_months + 2:
        mid = t_min + (t_max - t_min) / 2
        yield (ts < mid), (ts >= mid)
        return

    first_test_start = t_min + pd.DateOffset(months=int(total - n_splits * test_months))

    for fold in range(n_splits):
        test_start = first_test_start + pd.DateOffset(months=fold * test_months)
        test_end   = test_start       + pd.DateOffset(months=test_months)

        train_mask = ts < test_start
        test_mask  = (ts >= test_start) & (ts < test_end)

        if train_mask.sum() < 200 or test_mask.sum() < 50:
            continue

        yield train_mask, test_mask


# ── Walk-forward cross-validation ─────────────────────────────────────────

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray = None,
    n_splits: int = 4,
    test_months: int = 3,
) -> list:
    """Walk-forward CV; returns list of per-fold metric dicts."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    results = []
    for fold_idx, (train_mask, test_mask) in enumerate(
            walk_forward_splits(X, n_splits=n_splits, test_months=test_months), start=1):

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if sample_weights is not None:
            mask_arr = train_mask.to_numpy() if hasattr(train_mask, 'to_numpy') else np.asarray(train_mask)
            w_train  = sample_weights[mask_arr]
        else:
            w_train = None

        tp_rate = float(y_train.mean())
        base    = build_model(tp_rate=tp_rate)
        model   = _wrap_calibrated(base, X_train[FEATURE_COLS], y_train.to_numpy(),
                                    sample_weight=w_train)

        y_prob = model.predict_proba(X_test[FEATURE_COLS])[:, 1]

        # Production threshold metrics
        prod_threshold = ML_CONFIDENCE_THRESHOLD
        y_pred_prod    = (y_prob >= prod_threshold).astype(int)

        # Top-decile precision: precision among the top 10% most-confident predictions.
        # This shows actual model discrimination power independent of threshold choice
        # — it answers "if we trade only the model's strongest 10% of picks, what's the win rate?"
        top10_threshold = float(np.quantile(y_prob, 0.90))
        y_pred_top10    = (y_prob >= top10_threshold).astype(int)
        prec_top10      = float(precision_score(y_test, y_pred_top10, zero_division=0))

        # Top 5%: tighter cut for higher-confidence-only signals
        top5_threshold = float(np.quantile(y_prob, 0.95))
        y_pred_top5    = (y_prob >= top5_threshold).astype(int)
        prec_top5      = float(precision_score(y_test, y_pred_top5, zero_division=0))

        metrics = {
            'fold':          fold_idx,
            'train_size':    int(train_mask.sum()),
            'test_size':     int(test_mask.sum()),
            'threshold':     round(prod_threshold, 3),
            'precision':     round(float(precision_score(y_test, y_pred_prod, zero_division=0)), 4),
            'recall':        round(float(recall_score(y_test, y_pred_prod, zero_division=0)), 4),
            'f1':            round(float(f1_score(y_test, y_pred_prod, zero_division=0)), 4),
            'roc_auc':       round(float(roc_auc_score(y_test, y_prob)), 4),
            'baseline_rate': round(float(y_test.mean()), 4),
            'prec_top10':    round(prec_top10, 4),
            'prec_top5':     round(prec_top5,  4),
            'p90_threshold': round(top10_threshold, 4),
            'p95_threshold': round(top5_threshold,  4),
            'p_max':         round(float(y_prob.max()), 4),
        }
        results.append(metrics)
        print(f'  Fold {fold_idx}: prec@0.55={metrics["precision"]:.3f}  '
              f'recall={metrics["recall"]:.3f}  AUC={metrics["roc_auc"]:.3f}  '
              f'top10%={metrics["prec_top10"]:.3f}  top5%={metrics["prec_top5"]:.3f}  '
              f'p_max={metrics["p_max"]:.3f}  '
              f'(n_train={metrics["train_size"]:,}, n_test={metrics["test_size"]:,})')

    return results


# ── Final model training ───────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_metrics: list,
    sample_weights: np.ndarray = None,
) -> ModelBundle:
    """Train on the full labeled dataset and save to ML_MODEL_PATH."""
    import joblib

    tp_rate = float(y.mean())
    base    = build_model(tp_rate=tp_rate)
    model   = _wrap_calibrated(base, X[FEATURE_COLS], y.to_numpy(),
                                sample_weight=sample_weights)

    bundle = ModelBundle(
        model=model,
        feature_cols=FEATURE_COLS,
        trained_at=datetime.now().isoformat(),
        n_samples=len(X),
        tp_rate=round(tp_rate, 4),
        cv_metrics=cv_metrics,
        calibrated=ML_CALIBRATION_METHOD is not None,
        method=type(base).__name__,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ML_MODEL_PATH)
    print(f'[train] Model saved -> {ML_MODEL_PATH}  '
          f'({len(X):,} samples, TP rate {tp_rate:.1%})')

    _print_feature_importance(model)
    return bundle


def _print_feature_importance(model) -> None:
    """Print top-10 feature importances; unwraps CalibratedClassifierCV."""
    if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
        inner = model.calibrated_classifiers_[0].estimator
    else:
        inner = model

    if not hasattr(inner, 'feature_importances_'):
        return

    importance = pd.Series(inner.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=False)
    print('\n[train] Feature importance (top 10):')
    for feat, score in importance.head(10).items():
        bar = '#' * int(score / importance.max() * 20) if importance.max() > 0 else ''
        print(f'  {feat:<25} {bar} {score:.4f}')


# ── Public pipeline entry point ────────────────────────────────────────────

def run_training_pipeline(
    min_precision: float = ML_MIN_PRECISION,
    n_splits: int = 4,
    test_months: int = 3,
    signal_score_threshold: float = ML_SIGNAL_SCORE_THRESHOLD,
) -> ModelBundle:
    """
    Load features.parquet -> walk-forward CV -> train final model.

    Improvements vs prior version:
      - Triple-barrier ATR-scaled labels in the features parquet
      - Sample-uniqueness weights for overlapping labels
      - Probability calibration (isotonic) wrapping the base model
      - Imbalance-aware decision threshold during CV
    """
    print(f'\n[train] Loading features from {ML_FEATURES_PATH}')
    df = pd.read_parquet(ML_FEATURES_PATH)

    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    # Signal-quality filter — align training distribution with inference
    if signal_score_threshold > 0:
        if 'primary_score' in df.columns:
            signal_scores = df['primary_score']
        else:
            signal_scores = compute_signal_score_col(df)
        pre = len(df)
        df = df[signal_scores >= signal_score_threshold].copy()
        pct = len(df) / pre * 100 if pre > 0 else 0
        tp  = df['label'].mean() if len(df) > 0 else 0
        print(f'[train] Signal filter (score>={signal_score_threshold}): '
              f'{len(df):,} / {pre:,} rows ({pct:.1f}%)  |  TP rate in filtered set: {tp:.1%}')

        for thr in (50, 60, 70, 80, 90, 100):
            n = (signal_scores >= thr).sum()
            print(f'         score>={thr:3d}: {n:,} rows ({n/pre*100:.1f}%)')

    symbols = sorted(df['symbol'].unique())
    sym_map = {s: i for i, s in enumerate(symbols)}
    df['symbol_id'] = df['symbol'].map(sym_map).astype(float)

    X = df.copy()
    y = df['label']

    if ML_SAMPLE_WEIGHTS_ENABLED:
        print('[train] Computing sample-uniqueness weights...')
        sample_weights = compute_sample_weights(df)
        print(f'         weights — mean={sample_weights.mean():.3f}  '
              f'min={sample_weights.min():.3f}  max={sample_weights.max():.3f}')
    else:
        print('[train] Sample weighting disabled (ML_SAMPLE_WEIGHTS_ENABLED=False)')
        sample_weights = None

    print(f'[train] Dataset: {len(df):,} labeled rows | '
          f'TP rate {y.mean():.1%} | '
          f'{len(symbols)} stocks | '
          f'{df.index.min().date()} to {df.index.max().date()}')

    print(f'\n[train] Walk-forward CV ({n_splits} folds, {test_months}-month test window)')
    cv_results = walk_forward_cv(X, y, sample_weights=sample_weights,
                                  n_splits=n_splits, test_months=test_months)

    if not cv_results:
        raise RuntimeError('[train] No CV folds completed — insufficient data')

    avg_precision = np.mean([r['precision'] for r in cv_results])
    avg_auc       = np.mean([r['roc_auc']   for r in cv_results])
    avg_top10     = np.mean([r['prec_top10'] for r in cv_results])
    avg_top5      = np.mean([r['prec_top5']  for r in cv_results])
    print(f'\n[train] CV summary:')
    print(f'  Production threshold (0.55) precision: {avg_precision:.3f}')
    print(f'  Top-10% precision (gate signal):       {avg_top10:.3f}')
    print(f'  Top-5%  precision (high-conf only):    {avg_top5:.3f}')
    print(f'  AUC:                                   {avg_auc:.3f}')

    # The gate-filter quality metric: top-decile precision. This is what
    # actually matters for a "should I take this signal?" filter — we don't
    # care that the model can't perfectly classify everything; we care that
    # its most-confident picks are reliably positive.
    gate_precision = avg_top10
    if gate_precision < min_precision:
        raise RuntimeError(
            f'[train] CV top-10% precision {gate_precision:.3f} < minimum {min_precision:.3f}. '
            f'Model not saved. Consider more data or feature engineering.'
        )

    return train_final_model(X, y, cv_metrics=cv_results, sample_weights=sample_weights)
