"""
Advisory Mode — ask the model about a specific ticker.

Runs the momentum pipeline for a symbol and returns:
  - Rule-based score with per-category breakdown
  - ML confidence + gate decision (using bundle-tuned threshold)
  - Suggested entry, TP1/TP2, ATR-aware stop, position size, risk
  - Regime context (VIX term, SPY/QQQ EMA alignment)
  - Fundamental context and fair-value buy price

Default reads from cached Models/raw_bars.parquet (fresh if daily-update ran
this morning). Pass --refresh to force a live Alpaca fetch.

CLI:
  python Code/main.py advise --ticker AVGO
  python Code/main.py advise --ticker NVDA --equity 25000
  python Code/main.py advise --ticker COIN --refresh
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ML_RAW_BARS_PATH, MARKET_INDEXES,
    RISK_PER_TRADE, MAX_STOCK_CONCENTRATION,
    TAKE_PROFIT_PCT, TP1_PCT, TP1_SHARE_FRACTION,
    STRUCTURAL_STOP_MIN_DISTANCE,
    SIGNAL_BUY_THRESHOLD, SIGNAL_WATCH_THRESHOLD,
    ML_ENABLED, VIX_HARD_BLOCK,
)
import technicals as ta

ET = pytz.timezone('America/New_York')


# ── Shared pipeline (also used by scan_momentum) ──────────────────────────

def load_bars(refresh: bool = False) -> pd.DataFrame:
    """MultiIndex (symbol, timestamp) bars: cached by default, live with refresh."""
    if refresh:
        from ml.collect import fetch_recent
        print('[advise] Fetching live bars for the full universe...')
        return fetch_recent(days_back=90)
    if not ML_RAW_BARS_PATH.exists():
        raise FileNotFoundError(
            f'No cached bars at {ML_RAW_BARS_PATH}. '
            f'Run `python Code/main.py daily-update` first or use --refresh.'
        )
    age = (datetime.now().timestamp() - ML_RAW_BARS_PATH.stat().st_mtime) / 3600
    print(f'[advise] Using cached bars (age: {age:.1f}h). Pass --refresh for live fetch.')
    return pd.read_parquet(ML_RAW_BARS_PATH)


def market_context(bars: pd.DataFrame) -> tuple[dict, dict]:
    """Return (macro, regime): VIX/put-call features and SPY/QQQ EMA regime."""
    from ml.macro_features import get_latest_macro
    macro = get_latest_macro()
    symbols = bars.index.get_level_values(0)

    def _aligned(sym):
        if sym not in symbols:
            return 0.0
        c = bars.loc[sym]['close']
        if len(c) < 55:
            return 0.0
        return float(c.ewm(span=20, adjust=False).mean().iloc[-1] >
                     c.ewm(span=50, adjust=False).mean().iloc[-1])

    spy_ema, qqq_ema = _aligned('SPY'), _aligned('QQQ')
    if spy_ema and qqq_ema:
        regime_bias, regime_conf = 'bullish', 0.75
    elif not spy_ema and not qqq_ema:
        regime_bias, regime_conf = 'bearish', 0.75
    else:
        regime_bias, regime_conf = 'neutral', 0.50
    regime = {
        'spy_ema_aligned': spy_ema,
        'qqq_ema_aligned': qqq_ema,
        'regime_bias':     regime_bias,
        'regime_conf':     regime_conf,
    }
    return macro, regime


def compute_ranks(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional percentile ranks of each symbol's latest bar across the
    whole non-index universe — the same universe training ranks against.
    """
    rows = {}
    for sym in bars.index.get_level_values(0).unique():
        if sym in MARKET_INDEXES:
            continue
        df = bars.loc[sym]
        if len(df) < 100:
            continue
        ind = ta.compute_indicators(df)
        if ind is None:
            continue
        close = df['close']
        by_day = close.resample('B').last().dropna()
        rows[sym] = {
            # 65 bars = 5 trading days of 30-min bars
            'rs_rank_5d':        float(close.iloc[-1] / close.iloc[-66] - 1) if len(close) >= 66 else 0.0,
            'rsi_rank':          float(ind.get('rsi', 50.0)),
            'momentum_rank_20d': float(by_day.iloc[-1] / by_day.iloc[-21] - 1) if len(by_day) >= 21 else 0.0,
            'vol_ratio_rank':    float(ind['volume'] / ind['volume_avg']) if ind.get('volume_avg', 0) > 0 else 1.0,
        }
    return pd.DataFrame(rows).T.rank(pct=True, method='average').fillna(0.5)


def evaluate(ticker: str, bars: pd.DataFrame, macro: dict, regime: dict,
             ranks: pd.DataFrame, equity: float = 10_000.0) -> dict | None:
    """Score one ticker: rule score, ML gate, recommendation and entry plan.

    Returns None when the ticker lacks enough bars for indicators.
    """
    if ticker not in bars.index.get_level_values(0):
        return None
    df = bars.loc[ticker]
    qqq_df = bars.loc['QQQ'] if 'QQQ' in bars.index.get_level_values(0) else None
    if len(df) < 100:
        return None
    indicators = ta.compute_indicators(df)
    if indicators is None:
        return None

    indicators.update({
        'spy_ema_aligned':  regime['spy_ema_aligned'],
        'qqq_ema_aligned':  regime['qqq_ema_aligned'],
        'vix_9d':           macro['vix_9d'],
        'vix_3m':           macro['vix_3m'],
        'vix_term_ratio':   macro['vix_term_ratio'],
        'put_call_ratio':   macro['put_call_ratio'],
    })
    if ticker in ranks.index:
        indicators.update(ranks.loc[ticker].to_dict())

    rs = ta.compute_rs_vs_qqq(df, qqq_df) if qqq_df is not None else 0.0
    base_score, components = ta.score_signal(indicators, rs)
    final_score = ta.apply_regime_multiplier(
        base_score, regime['regime_bias'], regime['regime_conf'],
    )
    indicators['primary_base_score'] = round(base_score, 1)

    from ml.features import indicators_to_feature_row
    from ml.predict import predict_success_prob, get_threshold
    feature_row = indicators_to_feature_row(indicators, df, df.index[-1], rs_vs_qqq=rs)
    ml_prob = predict_success_prob(feature_row) if ML_ENABLED else float('nan')
    ml_threshold = get_threshold() if ML_ENABLED else 0.55

    entry_price = indicators['close']
    vwap        = indicators.get('vwap', 0.0)
    d_atr_pct   = indicators.get('d_atr_pct', 0.02) or (indicators['atr'] / entry_price)
    sl_price = ta.compute_stop(entry_price, vwap, d_atr_pct=d_atr_pct)
    stop_distance = entry_price - sl_price
    tp1_price = round(entry_price * (1 + TP1_PCT),        2)
    tp2_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

    # Position sizing
    if stop_distance <= 0:
        shares = 0
    else:
        risk_dollars = equity * RISK_PER_TRADE
        shares = int(risk_dollars / stop_distance)
        max_dollars = equity * MAX_STOCK_CONCENTRATION
        if shares * entry_price > max_dollars:
            shares = int(max_dollars / entry_price)
    tp1_shares  = int(shares * TP1_SHARE_FRACTION)
    tp2_shares  = shares - tp1_shares
    exposure    = shares * entry_price
    risk        = shares * stop_distance

    vix_spot = macro.get('vix_9d', 0)  # closest available proxy for spot VIX
    if vix_spot >= VIX_HARD_BLOCK:
        rec_class, reason = 'BLOCKED', f'VIX={vix_spot:.1f} >= {VIX_HARD_BLOCK}'
    elif final_score >= SIGNAL_BUY_THRESHOLD and (np.isnan(ml_prob) or ml_prob >= ml_threshold):
        rec_class, reason = 'BUY', 'score and ML gate pass'
    elif final_score >= SIGNAL_BUY_THRESHOLD:
        rec_class, reason = 'WATCH', 'score OK but ML gate failed'
    elif final_score >= SIGNAL_WATCH_THRESHOLD:
        rec_class, reason = 'WATCH', 'score below BUY threshold'
    else:
        rec_class, reason = 'SKIP', 'insufficient signal'

    return {
        'ticker':         ticker,
        'indicators':     indicators,
        'reason':         reason,
        'as_of':          df.index[-1].isoformat(),
        'price':          entry_price,
        'recommendation': rec_class,
        'base_score':     base_score,
        'final_score':    round(final_score, 1),
        'components':     components,
        'regime':         regime,
        'macro':          macro,
        'rs_vs_qqq':      rs,
        'ml_prob':        ml_prob,
        'ml_threshold':   ml_threshold,
        'entry_price':    entry_price,
        'tp1_price':      tp1_price,
        'tp2_price':      tp2_price,
        'sl_price':       sl_price,
        'stop_distance':  stop_distance,
        'shares':         shares,
        'tp1_shares':     tp1_shares,
        'tp2_shares':     tp2_shares,
        'exposure':       exposure,
        'risk_dollars':   risk,
        'equity':         equity,
    }


def advise(ticker: str, equity: float = 10_000.0,
           refresh: bool = False, verbose: bool = False) -> dict:
    """Run the momentum pipeline + fundamentals for one ticker and print the report."""
    ticker = ticker.upper()
    bars = load_bars(refresh)
    macro, regime = market_context(bars)
    result = evaluate(ticker, bars, macro, regime, compute_ranks(bars), equity)
    if result is None:
        raise ValueError(f'{ticker}: not in bar data or fewer than 100 bars')

    # Fundamentals are advisory context only — they never change the signal.
    try:
        from ml.fundamentals import get_ticker
        result['fundamentals'] = get_ticker(ticker)
    except Exception as e:
        print(f'  [advise] fundamentals fetch failed: {e}')
        result['fundamentals'] = {}

    from ml.predict import model_info
    _print_report(result, result['indicators'], model_info(), verbose=verbose)
    return result


def _print_report(r: dict, indicators: dict, bundle_info, verbose: bool = False) -> None:
    """Pretty-print the advisory report."""
    ticker = r['ticker']
    ts_local = pd.Timestamp(r['as_of']).tz_convert(ET) if 'as_of' in r else datetime.now(ET)

    print(f'\n{"="*72}')
    print(f'  ADVISORY REPORT — {ticker}  (as of {ts_local.strftime("%Y-%m-%d %H:%M ET")})')
    print(f'{"="*72}')
    print(f'\n  Current price: ${r["price"]:.2f}')
    reg = r['regime']
    print(f'  Regime:        {reg["regime_bias"]} '
          f'(SPY EMA aligned: {"YES" if reg["spy_ema_aligned"] else "no"}, '
          f'QQQ EMA aligned: {"YES" if reg["qqq_ema_aligned"] else "no"})')
    m = r['macro']
    print(f'  VIX regime:    VIX9D={m["vix_9d"]:.1f}  VIX3M={m["vix_3m"]:.1f}  '
          f'term_ratio={m["vix_term_ratio"]:.3f} '
          f'({"BACKWARDATION (fear)" if m["vix_term_ratio"] > 1.0 else "contango"})')

    # ── Rule-based score breakdown ────────────────────────────────────────
    c = r['components']
    print(f'\n  RULE-BASED SCORE:')
    max_pts = {'trend_score': 25, 'breakout_strength': 20, 'volume_quality': 20,
                'vwap_support': 20, 'relative_strength': 15, 'rsi_quality': 20,
                'macd_momentum': 15, 'structural_setup': 15}
    for key, mx in max_pts.items():
        val = c.get(key, 0)
        bar = '#' * int(val / max(mx, 1) * 15)
        print(f'    {key:<22} {val:>4}/{mx:<4} {bar}')
    penalty = c.get('vol_regime_penalty', 0)
    if penalty < 0:
        print(f'    {"vol_regime_penalty":<22} {penalty:>4} (iv_rank={c.get("iv_rank_proxy", 0):.2f})')
    print(f'    {"─" * 40}')
    print(f'    Base score:            {r["base_score"]:>4}/150')
    if r["final_score"] != r["base_score"]:
        mult = r["final_score"] / r["base_score"] if r["base_score"] > 0 else 1.0
        print(f'    Regime multiplier:     x{mult:.2f} ({reg["regime_bias"]})')
    print(f'    FINAL SCORE:           {r["final_score"]:>4.1f} '
          f'(BUY >= {SIGNAL_BUY_THRESHOLD}, WATCH >= {SIGNAL_WATCH_THRESHOLD})')

    # ── ML confidence ─────────────────────────────────────────────────────
    print(f'\n  ML CONFIDENCE:')
    if not np.isnan(r['ml_prob']):
        pass_str = 'PASS ✓' if r['ml_prob'] >= r['ml_threshold'] else 'FAIL ✗'
        print(f'    P(TP hit within 15 days):  {r["ml_prob"]:.3f}')
        print(f'    Bundle threshold:          {r["ml_threshold"]:.3f}')
        print(f'    Gate:                      {pass_str}')
        if bundle_info:
            print(f'    Model trained: {bundle_info["trained_at"][:19]}  '
                  f'({bundle_info["n_samples"]:,} samples, TP rate {bundle_info["tp_rate"]:.1%})')
    else:
        print(f'    Model disabled/absent — gate skipped')

    # ── Fundamental context (advisory only — does not change trading decisions) ──
    fund = r.get('fundamentals') or {}
    if fund.get('fetch_ok'):
        from ml.fundamentals import quality_label
        emoji, verdict, desc = quality_label(fund.get('quality_score', float('nan')))
        print(f'\n  FUNDAMENTAL CONTEXT (yfinance):')

        def _pct(v): return 'N/A' if v is None or pd.isna(v) else f'{v:+.1%}'
        def _num(v, d=1): return 'N/A' if v is None or pd.isna(v) else f'{v:.{d}f}'
        def _price(v): return 'N/A' if v is None or pd.isna(v) else f'${v:.2f}'
        target = fund.get('analyst_target')
        price  = fund.get('current_price', r.get('price'))
        upside = (target / price - 1) if target and price and price > 0 else None

        print(f'    Growth:       revenue {_pct(fund.get("revenue_growth"))} YoY, '
              f'earnings {_pct(fund.get("earnings_growth"))} YoY')
        print(f'    Valuation:    Forward PE {_num(fund.get("forward_pe"))} '
              f'| PEG {_num(fund.get("peg_ratio"), 2)} '
              f'| P/S {_num(fund.get("price_to_sales"), 2)}')
        print(f'    Profitability: ROE {_pct(fund.get("roe"))} '
              f'| profit margin {_pct(fund.get("profit_margin"))}')
        de = fund.get("debt_to_equity")
        if not pd.isna(de) and de and de > 5: de = de / 100
        print(f'    Balance:      Debt/Equity {_num(de, 2)} '
              f'| FCF {"positive" if fund.get("free_cashflow", 0) and fund.get("free_cashflow", 0) > 0 else "n/a"}')
        rec = fund.get('analyst_rec_mean')
        rec_label = ('strong buy' if rec and rec < 1.5 else 'buy' if rec and rec < 2.5
                      else 'hold' if rec and rec < 3.5 else 'sell' if rec else 'N/A')
        print(f'    Analysts:     {rec_label} (mean {_num(rec, 2)}) '
              f'| target {_price(target)} '
              f'({("+" if upside and upside > 0 else "")+f"{upside:.1%}" if upside is not None else "N/A"} upside)')
        # Fair value estimate — sector-appropriate methodology
        try:
            from ml.fundamentals import compute_fair_value
            fv = compute_fair_value(pd.Series(fund))
            if fv and 'fair_value' not in fv:
                print(f'\n    FAIR VALUE ESTIMATE: unavailable ({fv.get("data_quality", "insufficient data")})')
            elif fv:
                method = fv.get('method', 'default')
                method_label = {
                    'semi':         'semiconductor FCF DCF',
                    'saas':         'SaaS FCF DCF',
                    'sum_of_parts': 'segment sum-of-parts DCF',
                    'default':      'growth-tier P/E screen',
                }.get(method, method)
                print(f'\n    FAIR VALUE ESTIMATE ({method_label}):')
                if fv.get('method_detail'):
                    print(f'      {fv["method_detail"]}')
                if 'growth_start' in fv:
                    print(f'      Revenue growth:     {fv["growth_start"]*100:+.1f}% -> '
                          f'{fv["growth_end"]*100:+.1f}% (yr1 -> yr5, anchored on history)')
                    print(f'      FCF margin (TTM):   {fv["fcf_margin"]*100:.1f}%   '
                          f'WACC {fv["wacc"]*100:.1f}%, terminal g {fv["terminal_growth"]*100:.1f}%')
                print(f'      Fair value:         ${fv["fair_value"]:.2f}')
                print(f'      Buy price ({fv["margin_of_safety"]:.0%} MoS): ${fv["buy_price"]:.2f}')
                disc = fv["upside_to_fair_value"]
                verdict_price = ('🟢 UNDERVALUED' if disc > 0.10
                                  else '🟡 fair' if disc > -0.10
                                  else '🟠 overvalued' if disc > -0.25
                                  else '🔴 EXPENSIVE')
                print(f'      vs current ${fund.get("current_price", 0):.2f}: '
                      f'{disc*100:+.1f}%  {verdict_price}')
        except Exception:
            pass

        print(f'\n    Quality score: {fund.get("quality_score", 0):.0f}/100  {emoji} {verdict}')
        print(f'    {desc}')

        # Combined view — cross-reference technical vs fundamental
        tech_verdict = r["recommendation"]
        fund_score   = fund.get('quality_score', 50)
        if tech_verdict == 'BUY' and fund_score >= 70:
            print(f'\n    COMBINED VIEW: 🟢 HIGH CONVICTION — technical + fundamental both strong')
        elif tech_verdict == 'BUY' and fund_score < 40:
            print(f'\n    COMBINED VIEW: 🟠 CAUTION — technical BUY but fundamentals weak. '
                  f'Consider reduced size or skip.')
        elif tech_verdict == 'SKIP' and fund_score >= 70:
            print(f'\n    COMBINED VIEW: 🟡 QUALITY DIP — technical says no BUT fundamentals strong. '
                  f'Watch for reversal signals; no entry until technical confirms.')
        elif tech_verdict == 'WATCH' and fund_score >= 70:
            print(f'\n    COMBINED VIEW: 🟡 PROMISING — WATCH signal + strong fundamentals. '
                  f'Higher-priority candidate if score climbs above 100.')
    else:
        print(f'\n  FUNDAMENTAL CONTEXT: unavailable (yfinance fetch failed for {r["ticker"]})')

    # ── Recommendation ────────────────────────────────────────────────────
    label = {'BUY': '🟢 BUY SIGNAL', 'WATCH': '🟡 WATCH', 'SKIP': '⚪ SKIP'}.get(r['recommendation'], '🚫 BLOCKED')
    print(f'\n  RECOMMENDATION: {label} — {r["reason"]}')

    # ── Entry plan ────────────────────────────────────────────────────────
    print(f'\n  ENTRY PLAN (equity assumption: ${r["equity"]:,.0f}):')
    print(f'    Entry price (limit):     ${r["entry_price"]:>8.2f}   '
          f'(use limit at/below this)')
    print(f'    Take profit 1 (30% off): ${r["tp1_price"]:>8.2f}   '
          f'(+{TP1_PCT:.0%})')
    print(f'    Take profit 2 (70% off): ${r["tp2_price"]:>8.2f}   '
          f'(+{TAKE_PROFIT_PCT:.0%})')
    print(f'    Stop loss:               ${r["sl_price"]:>8.2f}   '
          f'({(r["sl_price"]/r["entry_price"] - 1):+.2%} — '
          f'{"ATR-scaled" if r["stop_distance"] > r["entry_price"] * STRUCTURAL_STOP_MIN_DISTANCE else "structural min"})')
    print(f'    Position size:           {r["shares"]:>8} shares '
          f'(TP1: {r["tp1_shares"]} + TP2: {r["tp2_shares"]})')
    print(f'    Exposure:                ${r["exposure"]:>8,.0f}  '
          f'({r["exposure"]/r["equity"]*100:>4.1f}% of equity)')
    print(f'    Risk:                    ${r["risk_dollars"]:>8,.0f}  '
          f'({r["risk_dollars"]/r["equity"]*100:>4.1f}% of equity — target {RISK_PER_TRADE:.1%})')

    # ── Caveats ──────────────────────────────────────────────────────────
    print(f'\n  CAVEATS:')
    if r['recommendation'] == 'BUY':
        print(f'    - Earnings dates and macro events are NOT checked — verify before entry')
        print(f'    - Assumes {r["equity"]:,.0f} equity; adjust with --equity')
    if reg["regime_bias"] == "bearish":
        print(f'    - Regime is BEARISH — reduce size or wait for regime flip')
    if m["vix_term_ratio"] > 1.05:
        print(f'    - VIX backwardation ({m["vix_term_ratio"]:.2f}) — fear peak, contrarian buy zone')

    if verbose:
        print(f'\n  RAW FEATURE VALUES (for debugging):')
        for k, v in sorted(indicators.items()):
            if isinstance(v, (int, float)):
                print(f'    {k:<28} {v:>10.4f}')
    print()
