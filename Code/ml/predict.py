"""
Step 5 — Inference: load the trained model and score a live signal.

Usage in Phase 1:
    from ml.predict import predict_success_prob
    from ml.features import indicators_to_feature_row

    feature_row = indicators_to_feature_row(indicators, bar_df, timestamp, rs_vs_qqq)
    p_success   = predict_success_prob(feature_row)   # float in [0.0, 1.0]
    # Gate: only proceed if p_success >= ML_CONFIDENCE_THRESHOLD (0.60)

Falls back to 0.5 (neutral) when the model file is absent, so Phase 1
continues to function using the rule-based score alone.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ML_MODEL_PATH, ML_CONFIDENCE_THRESHOLD
from ml.features import FEATURE_COLS

_bundle = None   # module-level cache: load once, reuse every call


def load_model():
    """Load ModelBundle from disk (cached after first call)."""
    global _bundle
    if _bundle is None and ML_MODEL_PATH.exists():
        import joblib
        try:
            _bundle = joblib.load(ML_MODEL_PATH)
        except Exception as exc:
            print(f'  [ml.predict] Failed to load model: {exc}')
    return _bundle


def predict_success_prob(feature_row: dict) -> float:
    """
    Return P(TP hit before SL) for the given feature row.

    Parameters
    ----------
    feature_row : dict keyed by FEATURE_COLS (from indicators_to_feature_row())

    Returns
    -------
    float in [0.0, 1.0] — 0.5 if the model is unavailable (neutral fallback)
    """
    bundle = load_model()
    if bundle is None:
        return 0.5

    try:
        import numpy as np
        import pandas as pd

        row = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in FEATURE_COLS}])
        row = row.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)

        prob = float(bundle.model.predict_proba(row)[0][1])
        return max(0.0, min(1.0, prob))   # clamp for safety
    except Exception as exc:
        print(f'  [ml.predict] Inference error: {exc}')
        return 0.5


def get_threshold(threshold: float = None) -> float:
    """
    Return the production decision threshold.

    Priority:
      1. Explicit `threshold` argument (caller override)
      2. ModelBundle.recommended_threshold (per-model, tuned during training)
      3. ML_CONFIDENCE_THRESHOLD from config (last-resort static default)

    The bundle threshold is the one to trust: every retrain re-tunes it to
    target a precision of 0.50 on the most recent holdout. The static config
    value only kicks in when the model file is missing or pre-dates this field.
    """
    if threshold is not None:
        return threshold
    bundle = load_model()
    if bundle is not None and getattr(bundle, 'recommended_threshold', None) is not None:
        return float(bundle.recommended_threshold)
    return ML_CONFIDENCE_THRESHOLD


def is_confident(feature_row: dict, threshold: float = None) -> bool:
    """
    Return True if P(success) >= the production threshold.
    Pass an explicit threshold to override the model's recommended value.
    """
    return predict_success_prob(feature_row) >= get_threshold(threshold)


def model_info() -> Optional[dict]:
    """Return training metadata from the loaded bundle, or None if absent."""
    bundle = load_model()
    if bundle is None:
        return None
    return {
        'trained_at':  bundle.trained_at,
        'n_samples':   bundle.n_samples,
        'tp_rate':     bundle.tp_rate,
        'feature_cols': bundle.feature_cols,
        'cv_metrics':  bundle.cv_metrics,
    }
