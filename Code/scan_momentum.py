"""
Scan the entire universe for momentum-model buy signals.

Runs the rule-based score + ML gate + fundamental context for every
ticker in cache and returns a compact table sorted by conviction.

Not a new strategy — just an efficient way to answer "what's ripe for
BUY right now?" across the whole universe without running the full
advisor 41 times.

CLI: python Code/main.py scan-momentum [--min-score 80] [--only-buys]
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ML_RAW_BARS_PATH, WATCHLIST, ML_EXTRA_TRAINING_SYMBOLS,
    SIGNAL_BUY_THRESHOLD, SIGNAL_WATCH_THRESHOLD,
)
import technicals as ta


def scan_momentum(min_score: float = 0, only_buys: bool = False) -> pd.DataFrame:
    """Return a DataFrame with technical + ML + fundamental verdicts for all tickers."""
    if not ML_RAW_BARS_PATH.exists():
        raise FileNotFoundError('No cached bars — run daily-update first')
    raw = pd.read_parquet(ML_RAW_BARS_PATH)

    # Compute macro once
    from ml.macro_features import get_latest_macro
    macro = get_latest_macro()

    # Compute SPY/QQQ regime alignment once
    def _aligned(df):
        if df is None or len(df) < 55:
            return 0.0
        c = df['close']
        return float(c.ewm(span=20, adjust=False).mean().iloc[-1] >
                     c.ewm(span=50, adjust=False).mean().iloc[-1])
    spy_df = raw.loc['SPY'] if 'SPY' in raw.index.get_level_values(0) else None
    qqq_df = raw.loc['QQQ'] if 'QQQ' in raw.index.get_level_values(0) else None
    spy_aligned = _aligned(spy_df); qqq_aligned = _aligned(qqq_df)
    if spy_aligned and qqq_aligned:
        regime_bias, regime_conf = 'bullish', 0.75
    elif not spy_aligned and not qqq_aligned:
        regime_bias, regime_conf = 'bearish', 0.75
    else:
        regime_bias, regime_conf = 'neutral', 0.50

    # Load fundamentals
    from ml.fundamentals import refresh_all, compute_fair_value
    fund_df = refresh_all(force=False)

    from ml.features import indicators_to_feature_row
    try:
        from ml.predict import predict_success_prob, get_threshold
        ml_threshold = get_threshold()
        ml_available = True
    except Exception:
        ml_threshold = 0.55
        ml_available = False

    all_tickers = list(WATCHLIST.keys()) + list(ML_EXTRA_TRAINING_SYMBOLS)
    rows = []
    for ticker in all_tickers:
        try:
            df = raw.loc[ticker]
        except (KeyError, TypeError):
            continue
        if len(df) < 100:
            continue

        indicators = ta.compute_indicators(df)
        if indicators is None:
            continue
        rs = ta.compute_rs_vs_qqq(df, qqq_df) if qqq_df is not None else 0.0
        indicators.update({
            'spy_ema_aligned':  spy_aligned,
            'qqq_ema_aligned':  qqq_aligned,
            'vix_9d':           macro['vix_9d'],
            'vix_3m':           macro['vix_3m'],
            'vix_term_ratio':   macro['vix_term_ratio'],
            'put_call_ratio':   macro['put_call_ratio'],
        })

        base_score, components = ta.score_signal(indicators, rs)
        final_score = ta.apply_regime_multiplier(base_score, regime_bias, regime_conf)

        # ML probability
        ml_prob = float('nan')
        if ml_available:
            try:
                feature_row = indicators_to_feature_row(indicators, df, df.index[-1], rs_vs_qqq=rs)
                ml_prob = predict_success_prob(feature_row)
            except Exception:
                pass

        # Recommendation
        if final_score >= SIGNAL_BUY_THRESHOLD and (np.isnan(ml_prob) or ml_prob >= ml_threshold):
            verdict = 'BUY'
        elif final_score >= SIGNAL_BUY_THRESHOLD:
            verdict = 'BUY-noML'   # rule fires but ML gate fails
        elif final_score >= SIGNAL_WATCH_THRESHOLD:
            verdict = 'WATCH'
        else:
            verdict = 'SKIP'

        # Fundamental context
        fund_row = fund_df.loc[ticker] if ticker in fund_df.index else pd.Series()
        fund_score = float(fund_row.get('quality_score', float('nan')))
        fv = compute_fair_value(fund_row) if len(fund_row) else {}
        discount_to_fair = fv.get('discount_to_fair', float('nan')) if fv else float('nan')

        rows.append({
            'ticker':          ticker,
            'price':           indicators['close'],
            'rule_score':      round(final_score, 1),
            'ml_prob':         ml_prob,
            'verdict':         verdict,
            'quality_score':   fund_score,
            'vs_fair_pct':     discount_to_fair * 100 if not pd.isna(discount_to_fair) else float('nan'),
        })

    result = pd.DataFrame(rows).sort_values(['rule_score'], ascending=False)
    if only_buys:
        result = result[result['verdict'].str.startswith('BUY')]
    if min_score > 0:
        result = result[result['rule_score'] >= min_score]

    _print_scan(result, ml_threshold, regime_bias)
    return result


def _print_scan(df: pd.DataFrame, ml_threshold: float, regime_bias: str):
    print(f'\n{"="*82}')
    print(f'  MOMENTUM SCAN — {len(df)} tickers  '
          f'(regime: {regime_bias}, ML threshold: {ml_threshold:.3f})')
    print(f'{"="*82}')
    header = f'  {"#":<3} {"Ticker":<7} {"Price":>8} {"Score":>6} {"ML":>6} {"Verdict":<10} {"Qual":>5} {"vs Fair":>8}'
    print(header)
    print(f'  {"-"*80}')
    for i, r in enumerate(df.itertuples(index=False), 1):
        emoji = {'BUY': '🟢', 'BUY-noML': '🟡', 'WATCH': '🟡', 'SKIP': '⚪'}.get(r.verdict, '⚪')
        ml_str = f'{r.ml_prob:.3f}' if not pd.isna(r.ml_prob) else '  N/A'
        qual   = f'{r.quality_score:>3.0f}' if not pd.isna(r.quality_score) else '  ?'
        vfair  = f'{r.vs_fair_pct:+6.1f}%' if not pd.isna(r.vs_fair_pct) else '     N/A'
        print(f'  {i:<3} {r.ticker:<7} ${r.price:>7.2f} {r.rule_score:>5.1f} {ml_str} '
              f'{emoji} {r.verdict:<8} {qual}   {vfair}')

    # Summary counts
    counts = df['verdict'].value_counts()
    print(f'\n  Summary: {dict(counts)}')
