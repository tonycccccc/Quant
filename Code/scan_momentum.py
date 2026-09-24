"""
Scan the entire universe for momentum-model buy signals.

Runs the same evaluation as the advisor (rule score + ML gate + regime) for
every cached ticker, plus fair-value context, sorted by rule score.

CLI: python Code/main.py scan-momentum [--min-score 80] [--only-buys]
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import WATCHLIST, ML_EXTRA_TRAINING_SYMBOLS
from advisor import load_bars, market_context, compute_ranks, evaluate


def scan_momentum(min_score: float = 0, only_buys: bool = False) -> pd.DataFrame:
    bars = load_bars()
    macro, regime = market_context(bars)
    ranks = compute_ranks(bars)

    from ml.fundamentals import refresh_all, compute_fair_value
    from ml.predict import get_threshold
    fund_df = refresh_all(force=False)

    rows = []
    for ticker in list(WATCHLIST.keys()) + list(ML_EXTRA_TRAINING_SYMBOLS):
        r = evaluate(ticker, bars, macro, regime, ranks)
        if r is None:
            continue
        fund_row = fund_df.loc[ticker].copy() if ticker in fund_df.index else pd.Series(dtype=object)
        fund_row['ticker'] = ticker
        fv = compute_fair_value(fund_row) if ticker in fund_df.index else {}
        rows.append({
            'ticker':        ticker,
            'price':         r['price'],
            'rule_score':    r['final_score'],
            'ml_prob':       r['ml_prob'],
            'verdict':       r['recommendation'],
            'reason':        r['reason'],
            'quality_score': float(fund_row.get('quality_score', float('nan'))),
            'vs_fair_pct':   fv.get('upside_to_fair_value', float('nan')) * 100,
            'buy_price':     fv.get('buy_price', float('nan')),
        })

    result = pd.DataFrame(rows).sort_values('rule_score', ascending=False)
    if only_buys:
        result = result[result['verdict'] == 'BUY']
    if min_score > 0:
        result = result[result['rule_score'] >= min_score]

    _print_scan(result, get_threshold(), regime['regime_bias'])
    return result


def _print_scan(df: pd.DataFrame, ml_threshold: float, regime_bias: str):
    print(f'\n{"="*96}')
    print(f'  MOMENTUM SCAN — {len(df)} tickers  '
          f'(regime: {regime_bias}, ML threshold: {ml_threshold:.3f})')
    print(f'{"="*96}')
    print(f'  {"#":<3} {"Ticker":<7} {"Price":>8} {"Score":>6} {"ML":>6} {"Verdict":<10} '
          f'{"Qual":>5} {"vs Fair":>8} {"Buy@MoS":>9}  Reason')
    print(f'  {"-"*94}')
    for i, r in enumerate(df.itertuples(index=False), 1):
        emoji = {'BUY': '🟢', 'WATCH': '🟡', 'BLOCKED': '🚫'}.get(r.verdict, '⚪')
        ml_str = f'{r.ml_prob:.3f}' if not pd.isna(r.ml_prob) else '  N/A'
        qual   = f'{r.quality_score:>3.0f}' if not pd.isna(r.quality_score) else '  ?'
        vfair  = f'{r.vs_fair_pct:+6.1f}%' if not pd.isna(r.vs_fair_pct) else '     N/A'
        buy    = f'${r.buy_price:>8.2f}' if not pd.isna(r.buy_price) else '      N/A'
        print(f'  {i:<3} {r.ticker:<7} ${r.price:>7.2f} {r.rule_score:>5.1f} {ml_str} '
              f'{emoji} {r.verdict:<8} {qual}   {vfair} {buy}  {r.reason}')
    print(f'\n  Summary: {df["verdict"].value_counts().to_dict()}')
