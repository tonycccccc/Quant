"""
Step 4 — Model training and walk-forward cross-validation.

Walk-forward validation ensures training data always precedes test data,
preventing the lookahead bias that random splits introduce in time-series.

Model priority:
  1. LightGBM (fast, best tabular performance)
  2. scikit-learn RandomForest (fallback if LightGBM not installed)

Trained model is saved to Models/quant_model.pkl as a ModelBundle
(model + feature-column list + training metadata).
"""
from __future__ import annotations

import json
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
    ML_MIN_PRECISION, ML_HISTORY_MONTHS, ML_SIGNAL_SCORE_THRESHOLD,
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


# ── Model factory ─────────────────────────────────────────────────────────

def build_model(tp_rate: float = 0.5):
    """
    Return an untrained classifier.  LightGBM preferred; falls back to
    scikit-learn RandomForest when LightGBM is not installed.

    tp_rate: fraction of positive labels, used to set class balance weight.
    """
    neg_rate = 1.0 - tp_rate
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


# ── Walk-forward splits ────────────────────────────────────────────────────

def walk_forward_splits(X: pd.DataFrame, n_splits: int = 4, test_months: int = 3):
    """
    Yield (train_mask, test_mask) boolean arrays in temporal order.

    The training window expands with each fold; the test window slides
    forward by test_months.  The first fold trains on everything before
    the first test window.

    Guarantees: no timestamp in test_mask appears in any earlier train_mask.
    """
    ts      = X.index
    t_min   = ts.min()
    t_max   = ts.max()
    total   = (t_max - t_min).days / 30.4  # months

    if total < n_splits * test_months + 2:
        # Not enough history for the requested splits; yield one split
        mid = t_min + (t_max - t_min) / 2
        yield (ts < mid), (ts >= mid)
        return

    # Start of first test window: leave enough history to train on
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
    n_splits: int = 4,
    test_months: int = 3,
) -> list:
    """
    Run walk-forward CV and return a list of per-fold metric dicts.

    Each dict contains:
      fold, train_size, test_size, precision, recall, f1, roc_auc, baseline_rate
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    results = []
    for fold_idx, (train_mask, test_mask) in enumerate(
            walk_forward_splits(X, n_splits=n_splits, test_months=test_months), start=1):

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        tp_rate = float(y_train.mean())
        model   = build_model(tp_rate=tp_rate)
        model.fit(X_train[FEATURE_COLS], y_train.to_numpy())

        y_pred = model.predict(X_test[FEATURE_COLS])
        y_prob = model.predict_proba(X_test[FEATURE_COLS])[:, 1]

        metrics = {
            'fold':          fold_idx,
            'train_size':    int(train_mask.sum()),
            'test_size':     int(test_mask.sum()),
            'precision':     round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall':        round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'f1':            round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'roc_auc':       round(float(roc_auc_score(y_test, y_prob)), 4),
            'baseline_rate': round(float(y_test.mean()), 4),
        }
        results.append(metrics)
        print(f'  Fold {fold_idx}: precision={metrics["precision"]:.3f}  '
              f'recall={metrics["recall"]:.3f}  AUC={metrics["roc_auc"]:.3f}  '
              f'(n_train={metrics["train_size"]:,}, n_test={metrics["test_size"]:,})')

    return results


# ── Final model training ───────────────────────────────────────────────────

def train_final_model(X: pd.DataFrame, y: pd.Series, cv_metrics: list) -> ModelBundle:
    """
    Train on the full labeled dataset and save to ML_MODEL_PATH.
    Returns the ModelBundle for inspection.
    """
    import joblib

    tp_rate = float(y.mean())
    model   = build_model(tp_rate=tp_rate)
    model.fit(X[FEATURE_COLS], y.to_numpy())

    bundle = ModelBundle(
        model=model,
        feature_cols=FEATURE_COLS,
        trained_at=datetime.now().isoformat(),
        n_samples=len(X),
        tp_rate=round(tp_rate, 4),
        cv_metrics=cv_metrics,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ML_MODEL_PATH)
    print(f'[train] Model saved -> {ML_MODEL_PATH}  ({len(X):,} samples, TP rate {tp_rate:.1%})')

    _print_feature_importance(model)
    return bundle


def _print_feature_importance(model) -> None:
    if not hasattr(model, 'feature_importances_'):
        return
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=False)
    print('\n[train] Feature importance (top 10):')
    for feat, score in importance.head(10).items():
        bar = '#' * int(score / importance.max() * 20)
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

    Filters training data to bars that score >= signal_score_threshold on the
    rule-based signal scorer, so the training distribution matches inference
    (where only high-scoring setups reach the ML gate).

    Raises RuntimeError if average CV precision falls below min_precision.
    """
    print(f'\n[train] Loading features from {ML_FEATURES_PATH}')
    df = pd.read_parquet(ML_FEATURES_PATH)

    # Keep only rows with a valid label
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    # Signal-quality filter — align training distribution with inference distribution
    if signal_score_threshold > 0:
        signal_scores = compute_signal_score_col(df)
        pre = len(df)
        df = df[signal_scores >= signal_score_threshold].copy()
        pct = len(df) / pre * 100
        tp  = df['label'].mean()
        print(f'[train] Signal filter (score>={signal_score_threshold}): '
              f'{len(df):,} / {pre:,} rows ({pct:.1f}%)  |  TP rate in filtered set: {tp:.1%}')

        # Show score distribution at key thresholds for tuning reference
        for thr in (50, 60, 70, 80, 90, 100):
            n = (signal_scores >= thr).sum()
            print(f'         score>={thr:3d}: {n:,} rows ({n/pre*100:.1f}%)')

    # Symbol as ordinal categorical feature (0-based int encoding)
    symbols = sorted(df['symbol'].unique())
    sym_map = {s: i for i, s in enumerate(symbols)}
    df['symbol_id'] = df['symbol'].map(sym_map).astype(float)

    X = df.copy()
    y = df['label']

    print(f'[train] Dataset: {len(df):,} labeled rows | '
          f'TP rate {y.mean():.1%} | '
          f'{len(symbols)} stocks | '
          f'{df.index.min().date()} to {df.index.max().date()}')

    # Walk-forward cross-validation
    print(f'\n[train] Walk-forward CV ({n_splits} folds, {test_months}-month test window)')
    cv_results = walk_forward_cv(X, y, n_splits=n_splits, test_months=test_months)

    if not cv_results:
        raise RuntimeError('[train] No CV folds completed — insufficient data')

    avg_precision = np.mean([r['precision'] for r in cv_results])
    avg_auc       = np.mean([r['roc_auc']   for r in cv_results])
    print(f'\n[train] CV summary: avg_precision={avg_precision:.3f}  avg_AUC={avg_auc:.3f}')

    if avg_precision < min_precision:
        raise RuntimeError(
            f'[train] CV precision {avg_precision:.3f} < minimum {min_precision:.3f}. '
            f'Model not saved. Consider more data or feature engineering.'
        )

    # Train on full dataset
    return train_final_model(X, y, cv_metrics=cv_results)
