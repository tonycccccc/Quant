"""
Unit tests for the ML pipeline.
All tests use synthetic OHLCV data — no network calls, no Alpaca credentials.

Run with:
    python -m pytest tests/test_ml_pipeline.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'Code'))

from ml.features import FEATURE_COLS, build_feature_matrix, indicators_to_feature_row
from ml.labels import generate_labels
from ml.train import build_model, walk_forward_splits, walk_forward_cv


# ── Synthetic data helpers ─────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 200,
    base_price: float = 100.0,
    trend: float = 0.0001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic 30-min OHLCV with a gentle uptrend and random noise.
    Timestamps are ET-timezone-aware starting 2025-01-02 09:30.
    """
    rng = np.random.default_rng(seed)
    closes = base_price * (1 + trend) ** np.arange(n) + rng.normal(0, 0.3, n)
    closes = np.maximum(closes, 1.0)
    noise  = np.abs(rng.normal(0, 0.4, n))
    highs  = closes + noise
    lows   = closes - noise
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    vols   = rng.integers(500_000, 5_000_000, n).astype(float)

    ts = pd.date_range(
        '2025-01-02 09:30', periods=n, freq='30min',
        tz='America/New_York',
    )
    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': vols},
        index=ts,
    )


def _make_flat_ohlcv(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    """Perfectly flat OHLCV — high = price+0.05, low = price-0.05, no trend."""
    ts = pd.date_range('2025-01-02 09:30', periods=n, freq='30min', tz='America/New_York')
    return pd.DataFrame({
        'open':   np.full(n, price),
        'high':   np.full(n, price + 0.05),
        'low':    np.full(n, price - 0.05),
        'close':  np.full(n, price),
        'volume': np.full(n, 1_000_000.0),
    }, index=ts)


def _make_synthetic_features_labels(n: int = 2000, seed: int = 0) -> tuple:
    """
    Return (X, y) suitable for training tests.
    Features are random floats; labels are random 0/1.
    Index is a datetime range so walk_forward_splits works.
    """
    rng = np.random.default_rng(seed)
    ts  = pd.date_range('2024-01-01', periods=n, freq='30min', tz='UTC')
    X   = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLS))),
                       columns=FEATURE_COLS, index=ts)
    y   = pd.Series(rng.integers(0, 2, n).astype(float), index=ts, name='label')
    return X, y


# ══════════════════════════════════════════════════════════════════════════
# Feature tests
# ══════════════════════════════════════════════════════════════════════════

class TestFeatureMatrix:
    def test_all_feature_cols_present(self):
        """build_feature_matrix must return every column in FEATURE_COLS."""
        df   = _make_ohlcv(n=200)
        feat = build_feature_matrix(df)
        for col in FEATURE_COLS:
            assert col in feat.columns, f'Missing feature column: {col}'

    def test_output_length(self):
        """Output has fewer rows than input (warmup rows are dropped)."""
        df   = _make_ohlcv(n=200)
        feat = build_feature_matrix(df)
        assert len(feat) < len(df)
        assert len(feat) > 0

    def test_no_inf_in_features(self):
        """No infinite values after feature computation."""
        df   = _make_ohlcv(n=200)
        feat = build_feature_matrix(df)
        assert not np.isinf(feat[FEATURE_COLS].values).any()

    def test_rsi_bounds(self):
        """RSI values must be in [0, 100]."""
        df   = _make_ohlcv(n=200)
        feat = build_feature_matrix(df)
        rsi  = feat['rsi']
        assert (rsi >= 0).all() and (rsi <= 100).all(), \
            f'RSI out of bounds: min={rsi.min():.2f} max={rsi.max():.2f}'

    def test_boolean_features_binary(self):
        """Binary features must only contain 0 or 1."""
        df   = _make_ohlcv(n=200)
        feat = build_feature_matrix(df)
        for col in ('vwap_hold', 'spy_ema_aligned', 'qqq_ema_aligned', 'd_ema_aligned'):
            unique = set(feat[col].dropna().unique())
            assert unique.issubset({0.0, 1.0}), \
                f'{col} contains non-binary values: {unique}'

    def test_h4_rsi_bounds(self):
        """4H proxy RSI must be in [0, 100]."""
        df   = _make_ohlcv(n=200)
        feat = build_feature_matrix(df)
        assert (feat['h4_rsi'] >= 0).all() and (feat['h4_rsi'] <= 100).all()

    def test_daily_features_present(self):
        """All daily-timeframe features must be present and finite."""
        df   = _make_ohlcv(n=400)   # more bars -> more daily bars for warmup
        feat = build_feature_matrix(df)
        for col in ('d_close_ema20_ratio', 'd_ema_aligned', 'd_rsi',
                    'd_atr_pct', 'd_vol_ratio', 'd_return_20d'):
            assert col in feat.columns, f'Missing daily feature: {col}'
            assert np.isfinite(feat[col]).all(), f'{col} contains non-finite values'

    def test_nan_free_after_warmup(self):
        """Core feature columns should have <5% NaN after warmup."""
        df      = _make_ohlcv(n=200)
        feat    = build_feature_matrix(df)
        nan_pct = feat[FEATURE_COLS].isna().mean()
        assert (nan_pct < 0.05).all(), \
            f'Excessive NaN in: {nan_pct[nan_pct >= 0.05].to_dict()}'

    def test_qqq_rs_feature_nonzero(self):
        """When QQQ is provided, rs_vs_qqq should differ from 0 in at least some bars."""
        stock = _make_ohlcv(n=200, trend=0.0003, seed=1)
        qqq   = _make_ohlcv(n=200, trend=0.0001, seed=2)
        feat  = build_feature_matrix(stock, qqq_df=qqq)
        # At least some bars should have non-zero RS
        assert feat['rs_vs_qqq'].abs().sum() > 0


class TestInferenceAdapter:
    def test_indicators_to_feature_row_keys(self):
        """indicators_to_feature_row returns all FEATURE_COLS keys."""
        indicators = {
            'close': 150.0, 'ema20': 148.0, 'ema50': 145.0,
            'atr': 2.5, 'vwap': 149.0, 'volume': 3_000_000,
            'volume_avg': 2_000_000, 'atr_contraction_ratio': 0.85,
            'higher_highs': True, 'higher_lows': True,
            'resistance': 148.0, 'vwap_hold': True,
            'rsi': 62.0, 'macd_line': 0.5, 'macd_histogram': 0.1,
        }
        bar_df = _make_ohlcv(n=100)
        from datetime import datetime
        import pytz
        ts = datetime(2025, 6, 15, 14, 30, tzinfo=pytz.timezone('America/New_York'))
        row = indicators_to_feature_row(indicators, bar_df, ts, rs_vs_qqq=0.012)
        for col in FEATURE_COLS:
            assert col in row, f'Missing key: {col}'

    def test_ratio_features_are_finite(self):
        """All ratio features must be finite floats."""
        indicators = {
            'close': 100.0, 'ema20': 99.0, 'ema50': 97.0,
            'atr': 1.5, 'vwap': 99.5, 'volume': 2_000_000,
            'volume_avg': 1_500_000, 'atr_contraction_ratio': 0.90,
            'higher_highs': False, 'higher_lows': True,
            'resistance': 99.0, 'vwap_hold': False,
            'rsi': 55.0, 'macd_line': -0.1, 'macd_histogram': 0.05,
        }
        bar_df = _make_ohlcv(n=100)
        from datetime import datetime
        ts = datetime(2025, 3, 5, 10, 0)
        row = indicators_to_feature_row(indicators, bar_df, ts)
        for col in FEATURE_COLS:
            assert np.isfinite(row[col]), f'{col} = {row[col]} is not finite'


# ══════════════════════════════════════════════════════════════════════════
# Label tests
# ══════════════════════════════════════════════════════════════════════════

class TestLabels:
    def test_tp_hit_first_gives_label_1(self):
        """High crossing TP on the very next bar → label = 1.0."""
        df = _make_flat_ohlcv(n=200, price=100.0).copy()
        # Bar 0 close = 100; TP = 107; set high[1] = 108 (above TP)
        df.iloc[1, df.columns.get_loc('high')] = 108.0
        labels = generate_labels(df, tp_pct=0.07, sl_pct=0.035, timeout_bars=130)
        assert labels.iloc[0] == 1.0, f'Expected 1.0, got {labels.iloc[0]}'

    def test_sl_hit_first_gives_label_0(self):
        """Low dropping below SL on the next bar → label = 0.0."""
        df = _make_flat_ohlcv(n=200, price=100.0).copy()
        # SL level = 100 * 0.965 = 96.5; set low[1] = 95 (below SL)
        df.iloc[1, df.columns.get_loc('low')] = 95.0
        labels = generate_labels(df, tp_pct=0.07, sl_pct=0.035, timeout_bars=130)
        assert labels.iloc[0] == 0.0, f'Expected 0.0, got {labels.iloc[0]}'

    def test_sl_before_tp_gives_label_0(self):
        """SL hit at bar 2, TP hit at bar 5 → label = 0 (SL is first)."""
        df = _make_flat_ohlcv(n=200, price=100.0).copy()
        df.iloc[2, df.columns.get_loc('low')]  = 95.0    # SL at bar 2
        df.iloc[5, df.columns.get_loc('high')] = 108.0   # TP at bar 5
        labels = generate_labels(df, tp_pct=0.07, sl_pct=0.035, timeout_bars=130)
        assert labels.iloc[0] == 0.0, 'SL (bar 2) must beat TP (bar 5)'

    def test_timeout_gives_label_0(self):
        """Neither TP nor SL reached within timeout → label = 0."""
        df = _make_flat_ohlcv(n=300, price=100.0)
        # Flat price ±0.05 never reaches TP (+7%) or SL (-3.5%)
        labels = generate_labels(df, tp_pct=0.07, sl_pct=0.035, timeout_bars=130)
        first_valid = labels.dropna().iloc[0]
        assert first_valid == 0.0, f'Expected timeout → 0.0, got {first_valid}'

    def test_last_rows_are_nan(self):
        """The last timeout_bars rows must be NaN (incomplete forward window)."""
        timeout = 50
        df      = _make_ohlcv(n=200)
        labels  = generate_labels(df, timeout_bars=timeout)
        tail    = labels.iloc[-timeout:]
        assert tail.isna().all(), 'Last timeout_bars rows should be NaN'

    def test_labels_binary(self):
        """All non-NaN labels must be exactly 0.0 or 1.0."""
        df     = _make_ohlcv(n=200)
        labels = generate_labels(df)
        valid  = labels.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0}), \
            f'Non-binary labels: {set(valid.unique())}'


# ══════════════════════════════════════════════════════════════════════════
# Training tests
# ══════════════════════════════════════════════════════════════════════════

class TestTraining:
    def test_walk_forward_splits_no_overlap(self):
        """Train timestamps must never appear in the test set."""
        X, y = _make_synthetic_features_labels(n=5000)
        for train_mask, test_mask in walk_forward_splits(X, n_splits=3, test_months=2):
            train_ts = set(X.index[train_mask])
            test_ts  = set(X.index[test_mask])
            assert not (train_ts & test_ts), 'Train/test timestamp overlap detected'

    def test_walk_forward_splits_test_after_train(self):
        """Every test timestamp must be later than every train timestamp."""
        X, y = _make_synthetic_features_labels(n=5000)
        for train_mask, test_mask in walk_forward_splits(X, n_splits=3, test_months=2):
            max_train_ts = X.index[train_mask].max()
            min_test_ts  = X.index[test_mask].min()
            assert min_test_ts > max_train_ts, \
                f'Test starts before train ends: {min_test_ts} ≤ {max_train_ts}'

    def test_build_model_returns_classifier(self):
        """build_model() returns an object with fit() and predict_proba()."""
        model = build_model(tp_rate=0.45)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict_proba')

    def test_model_fit_and_predict(self):
        """Model trains on synthetic data and outputs valid probabilities."""
        X, y = _make_synthetic_features_labels(n=500)
        model = build_model(tp_rate=float(y.mean()))
        model.fit(X.values, y.astype(int).values)
        proba = model.predict_proba(X.values[:10])
        assert proba.shape == (10, 2)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
        # Each row must sum to 1 (within floating-point tolerance)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_walk_forward_cv_runs(self):
        """walk_forward_cv completes without error and returns metric dicts."""
        X, y = _make_synthetic_features_labels(n=3000)
        results = walk_forward_cv(X, y, n_splits=2, test_months=2)
        assert len(results) >= 1
        for r in results:
            assert 'precision' in r
            assert 0.0 <= r['precision'] <= 1.0
            assert 0.0 <= r['roc_auc'] <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# Predict (inference) tests
# ══════════════════════════════════════════════════════════════════════════

class TestPredict:
    def test_fallback_when_no_model(self, tmp_path, monkeypatch):
        """predict_success_prob returns 0.5 when the model file is missing."""
        import ml.predict as mp
        monkeypatch.setattr(mp, 'ML_MODEL_PATH', tmp_path / 'no_model.pkl')
        monkeypatch.setattr(mp, '_bundle', None)

        row  = {col: 0.0 for col in FEATURE_COLS}
        prob = mp.predict_success_prob(row)
        assert prob == 0.5

    def test_probability_in_range(self, tmp_path, monkeypatch):
        """predict_success_prob always returns a float in [0, 1]."""
        import ml.predict as mp
        monkeypatch.setattr(mp, 'ML_MODEL_PATH', tmp_path / 'no_model.pkl')
        monkeypatch.setattr(mp, '_bundle', None)

        for val in [-999.0, 0.0, 1.0, 999.0]:
            row  = {col: val for col in FEATURE_COLS}
            prob = mp.predict_success_prob(row)
            assert 0.0 <= prob <= 1.0, f'Out-of-range probability {prob} for val={val}'

    def test_is_confident_threshold(self, tmp_path, monkeypatch):
        """is_confident returns False when model is absent (prob == 0.5 < 0.6)."""
        import ml.predict as mp
        monkeypatch.setattr(mp, 'ML_MODEL_PATH', tmp_path / 'no_model.pkl')
        monkeypatch.setattr(mp, '_bundle', None)

        row = {col: 0.0 for col in FEATURE_COLS}
        assert not mp.is_confident(row, threshold=0.6)
