"""Regression tests for genuinely unseen threshold-validation data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Code'))
from ml import train


def dataset():
    times = pd.date_range('2025-01-01', periods=600, freq='30min', tz='UTC')
    index = times.repeat(2)
    X = pd.DataFrame({c: np.arange(len(index)) for c in train._MODEL_FEATURES}, index=index)
    X['t1'] = index + pd.Timedelta(hours=2)
    y = pd.Series(np.arange(len(index)) % 2, index=index)
    # Mimic symbol-grouped input rather than chronological input.
    order = np.r_[np.arange(0, len(X), 2), np.arange(1, len(X), 2)]
    return X.iloc[order], y.iloc[order]


class SpyModel:
    def __init__(self):
        self.fit_calls = []
        self.prediction_rows = []

    def fit(self, X, y, sample_weight=None):
        self.fit_calls.append((X.copy(), np.array(y), sample_weight))
        return self

    def predict_proba(self, X):
        assert not X.index.isin(self.fit_calls[0][0].index).any()
        self.prediction_rows.append(X.copy())
        return np.tile([0.4, 0.6], (len(X), 1))


@pytest.mark.parametrize('custom_features', [False, True])
def test_unseen_holdout_purges_labels_and_aligns_weights(monkeypatch, custom_features):
    X, y = dataset()
    model = SpyModel()
    monkeypatch.setattr(train, 'build_model', lambda **kw: model)
    monkeypatch.setattr(train, 'ML_CALIBRATION_METHOD', None)
    weights = np.arange(len(X)) + 1
    cols = train._MODEL_FEATURES[:2] if custom_features else None
    result = train.fit_with_threshold_holdout(X, y, weights, feature_cols=cols)
    fit_X, fit_y, fit_weights = model.fit_calls[0]
    boundary = X.index.unique().sort_values()[480]
    mask = (X.index < boundary) & (X['t1'] < boundary)
    pd.testing.assert_frame_equal(fit_X, X.loc[mask, cols or train._MODEL_FEATURES])
    np.testing.assert_array_equal(fit_y, y.loc[mask])
    np.testing.assert_array_equal(fit_weights, weights[mask])
    assert (X.loc[mask, 't1'] < boundary).all()
    assert model.prediction_rows[0].index.min() == boundary
    assert len(model.prediction_rows[0]) == 240
    assert len(model.fit_calls) == 1
    assert result[4] == mask.sum()


def test_saved_model_is_not_refit_on_holdout(monkeypatch, tmp_path):
    X, y = dataset()
    model = SpyModel()
    monkeypatch.setattr(train, 'build_model', lambda **kw: model)
    monkeypatch.setattr(train, 'ML_CALIBRATION_METHOD', None)
    monkeypatch.setattr(train, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(train, 'ML_MODEL_PATH', tmp_path / 'model.pkl')
    monkeypatch.setattr(train, '_print_feature_importance', lambda model: None)
    bundle = train.train_final_model(X, y, [])
    import joblib
    saved = joblib.load(tmp_path / 'model.pkl')
    assert len(saved.model.fit_calls) == 1
    assert bundle.n_samples == len(model.fit_calls[0][0]) < len(X)
    assert saved.recommended_threshold == bundle.recommended_threshold


def test_small_holdout_uses_static_threshold_without_refit(monkeypatch):
    X, y = dataset()
    X, y = X.iloc[:100], y.iloc[:100]
    # Supply both classes after selecting one symbol's rows.
    y = pd.Series(np.arange(len(X)) % 2, index=X.index)
    model = SpyModel()
    monkeypatch.setattr(train, 'build_model', lambda **kw: model)
    monkeypatch.setattr(train, 'ML_CALIBRATION_METHOD', None)
    result = train.fit_with_threshold_holdout(X, y)
    assert result[1] == 0.55
    assert len(model.fit_calls) == 1
    assert not model.prediction_rows


@pytest.mark.parametrize('problem', ['missing_t1', 'null_t1', 'single_class', 'short'])
def test_invalid_split_fails_before_fitting(monkeypatch, problem):
    X, y = dataset()
    if problem == 'missing_t1':
        X = X.drop(columns='t1')
    elif problem == 'null_t1':
        X['t1'] = pd.NaT
    elif problem == 'single_class':
        y[:] = 1
    else:
        X, y = X.iloc[:1], y.iloc[:1]
    monkeypatch.setattr(train, 'build_model', lambda **kw: pytest.fail('must not fit'))
    with pytest.raises(ValueError):
        train.fit_with_threshold_holdout(X, y)
