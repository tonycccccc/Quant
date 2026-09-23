"""
Advisory Mode — ask the model about a specific ticker.

Runs the full RATMB pipeline on demand for a single symbol and returns:
  - Rule-based score with per-category breakdown
  - ML confidence + gate decision (using bundle-tuned threshold)
  - Suggested entry, TP1/TP2, ATR-aware stop, position size, risk
  - Model context (OOS win rate, avg win/loss) so you can gauge trust
  - Regime context (VIX term, SPY/QQQ EMA alignment)

Default reads from cached Models/raw_bars.parquet (fresh if daily-update ran
this morning). Pass --refresh to force a live Alpaca fetch.

CLI:
  python Code/main.py advise --ticker AVGO
  python Code/main.py advise --ticker NVDA --equity 25000
  python Code/main.py advise --ticker COIN --refresh
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ML_RAW_BARS_PATH, WATCHLIST, ML_EXTRA_TRAINING_SYMBOLS,
    RISK_PER_TRADE, MAX_STOCK_CONCENTRATION,
    HARD_STOP_LOSS_PCT, TAKE_PROFIT_PCT, TP1_PCT, TP1_SHARE_FRACTION,
    STRUCTURAL_STOP_MIN_DISTANCE,
    SIGNAL_BUY_THRESHOLD, SIGNAL_WATCH_THRESHOLD,
    ML_CONFIDENCE_THRESHOLD, ML_ENABLED, VIX_HARD_BLOCK,
)
import technicals as ta

ET = pytz.timezone('America/New_York')


# ── Cache helpers ─────────────────────────────────────────────────────────

def _load_cached_bars() -> tuple:
    """
    Load raw_bars.parquet (from daily-update). Returns (df, age_hours).
    Raises FileNotFoundError if cache is missing.
    """
    if not ML_RAW_BARS_PATH.exists():
        raise FileNotFoundError(
            f'No cached bars at {ML_RAW_BARS_PATH}. '
            f'Run `python Code/main.py daily-update` first or use --refresh.'
        )
    df = pd.read_parquet(ML_RAW_BARS_PATH)
    age = (datetime.now().timestamp() - ML_RAW_BARS_PATH.stat().st_mtime) / 3600
    return df, age


def _fetch_fresh_bars(ticker: str) -> pd.DataFrame:
    """Fetch 90 days of live 30-min bars for a single ticker + QQQ + SPY."""
    from phase1_polling import fetch_bars
    tickers = list({ticker, 'QQQ', 'SPY'} | set(WATCHLIST.keys()))
    all_bars = fetch_bars(tickers, days_back=90)
    return all_bars


# ── Cross-sectional rank helper ───────────────────────────────────────────

def _compute_ranks_across_watchlist(all_bars_indexed, target: str) -> dict:
    """
    Compute cross-sectional ranks (rs_rank_5d, rsi_rank, momentum_rank_20d,
    vol_ratio_rank) across all in-cache watchlist tickers. Returns target's rank.
    """
    import pandas as pd
    rows = {}
    tickers = list(WATCHLIST.keys())
    if target not in tickers:
        tickers.append(target)

    for t in tickers:
        try:
            df = all_bars_indexed.loc[t]
        except (KeyError, TypeError):
            continue
        if len(df) < 100:
            continue
        indicators = ta.compute_indicators(df)
        if indicators is None:
            continue
        bar_close = df['close']
        prior_5d = float(bar_close.iloc[-1] / bar_close.iloc[-66] - 1) if len(bar_close) >= 66 else 0.0
        vol_ratio = (indicators.get('volume', 0.0) / indicators['volume_avg']) \
            if indicators.get('volume_avg', 0) > 0 else 1.0
        # d_return_20d proxy: 20 trading days ~= 260 30-min bars
        try:
            idx_et = df.index.tz_convert('America/New_York') if df.index.tz else df.index
            d = df.copy(); d.index = idx_et
            by_day = d.resample('B').agg(close=('close', 'last')).dropna()
            d_return_20d = float(by_day['close'].iloc[-1] / by_day['close'].iloc[-21] - 1) \
                if len(by_day) >= 21 else 0.0
        except Exception:
            d_return_20d = 0.0
        rows[t] = {
            'prior_5d_return': prior_5d,
            'rsi':             float(indicators.get('rsi', 50.0)),
            'd_return_20d':    d_return_20d,
            'vol_ratio':       float(vol_ratio),
        }

    if target not in rows:
        return {'rs_rank_5d': 0.5, 'rsi_rank': 0.5,
                 'momentum_rank_20d': 0.5, 'vol_ratio_rank': 0.5}
    cross = pd.DataFrame(rows).T
    rank_map = {'prior_5d_return': 'rs_rank_5d',
                'rsi':             'rsi_rank',
                'd_return_20d':    'momentum_rank_20d',
                'vol_ratio':       'vol_ratio_rank'}
    out = {}
    for src, dst in rank_map.items():
        ranks = cross[src].rank(pct=True, method='average').fillna(0.5)
        out[dst] = float(ranks[target])
    return out


# ── Regime determination ─────────────────────────────────────────────────

def _get_regime_context(spy_df, qqq_df, macro: dict) -> dict:
    """
    Compute macro regime context: SPY/QQQ EMA alignment + VIX term structure.
    Returns dict with regime_bias, spy_ema_aligned, qqq_ema_aligned.
    """
    def _aligned(df):
        if df is None or len(df) < 55:
            return 0.0
        c = df['close']
        return float(c.ewm(span=20, adjust=False).mean().iloc[-1] >
                     c.ewm(span=50, adjust=False).mean().iloc[-1])

    spy_ema = _aligned(spy_df)
    qqq_ema = _aligned(qqq_df)

    # Simple regime bias mapping
    if spy_ema and qqq_ema:
        regime_bias, regime_conf = 'bullish', 0.75
    elif not spy_ema and not qqq_ema:
        regime_bias, regime_conf = 'bearish', 0.75
    else:
        regime_bias, regime_conf = 'neutral', 0.50

    return {
        'spy_ema_aligned':  spy_ema,
        'qqq_ema_aligned':  qqq_ema,
        'regime_bias':      regime_bias,
        'regime_conf':      regime_conf,
        'vix_backwardation': macro.get('vix_term_ratio', 1.0) > 1.0,
    }


# ── Main advisory function ───────────────────────────────────────────────

def advise(ticker: str, equity: float = 10_000.0,
            refresh: bool = False, verbose: bool = False) -> dict:
    """
    Run the full RATMB pipeline for one ticker and print the advisory report.
    Returns a dict with all computed values for programmatic use.
    """
    ticker = ticker.upper()

    # ── 1. Load bar data ─────────────────────────────────────────────────
    if refresh:
        print(f'[advise] Fetching fresh bars for {ticker} + watchlist...')
        all_bars = _fetch_fresh_bars(ticker)
        # phase1_polling.fetch_bars returns dict; convert to indexed df
        combined = []
        for t, df in all_bars.items():
            df2 = df.copy()
            df2['symbol'] = t
            combined.append(df2.set_index([pd.Index([t]*len(df2)), df2.index]))
        cached = pd.concat(combined) if combined else None
        age_hours = 0
    else:
        cached, age_hours = _load_cached_bars()
        print(f'[advise] Using cached bars (age: {age_hours:.1f}h). '
              f'Pass --refresh for live fetch.')

    if ticker not in cached.index.get_level_values(0):
        raise ValueError(
            f'{ticker} not found in cache. Available: '
            f'{sorted(cached.index.get_level_values(0).unique().tolist())[:20]}...'
        )

    df     = cached.loc[ticker]
    qqq_df = cached.loc['QQQ'] if 'QQQ' in cached.index.get_level_values(0) else None
    spy_df = cached.loc['SPY'] if 'SPY' in cached.index.get_level_values(0) else None

    if len(df) < 100:
        raise ValueError(f'{ticker} has only {len(df)} bars — need at least 100 for indicators')

    # ── 2. Compute indicators ────────────────────────────────────────────
    indicators = ta.compute_indicators(df)
    if indicators is None:
        raise ValueError(f'Could not compute indicators for {ticker}')

    # ── 3. Fetch macro features (VIX term + put/call, uses 24h cache) ────
    from ml.macro_features import get_latest_macro
    macro = get_latest_macro()

    # ── 4. Regime context ────────────────────────────────────────────────
    regime = _get_regime_context(spy_df, qqq_df, macro)
    indicators.update({
        'spy_ema_aligned':  regime['spy_ema_aligned'],
        'qqq_ema_aligned':  regime['qqq_ema_aligned'],
        'vix_9d':           macro['vix_9d'],
        'vix_3m':           macro['vix_3m'],
        'vix_term_ratio':   macro['vix_term_ratio'],
        'put_call_ratio':   macro['put_call_ratio'],
    })

    # ── 5. Cross-sectional ranks ─────────────────────────────────────────
    ranks = _compute_ranks_across_watchlist(cached, ticker)
    indicators.update(ranks)

    # ── 6. Relative strength vs QQQ ──────────────────────────────────────
    rs = ta.compute_rs_vs_qqq(df, qqq_df) if qqq_df is not None else 0.0

    # ── 7. Rule-based score ──────────────────────────────────────────────
    base_score, components = ta.score_signal(indicators, rs)
    final_score = ta.apply_regime_multiplier(
        base_score, regime['regime_bias'], regime['regime_conf'],
    )

    # ── 8. ML probability ───────────────────────────────────────────────
    from ml.features import indicators_to_feature_row
    from ml.predict import predict_success_prob, get_threshold, model_info

    feature_row = indicators_to_feature_row(indicators, df, df.index[-1], rs_vs_qqq=rs)
    ml_prob = predict_success_prob(feature_row) if ML_ENABLED else float('nan')
    ml_threshold = get_threshold() if ML_ENABLED else 0.55
    bundle_info = model_info()

    # ── 9. Entry / exit plan ─────────────────────────────────────────────
    from phase2_execution import PortfolioManager
    entry_price = indicators['close']
    vwap        = indicators.get('vwap', 0.0)
    d_atr_pct   = indicators.get('d_atr_pct', 0.02) or (indicators['atr'] / entry_price)

    pm = PortfolioManager.__new__(PortfolioManager)   # skip __init__ (no Alpaca client needed)
    sl_price = pm._compute_stop(entry_price, vwap, d_atr_pct=d_atr_pct)
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

    # ── 10. Recommendation ──────────────────────────────────────────────
    if final_score >= SIGNAL_BUY_THRESHOLD and (np.isnan(ml_prob) or ml_prob >= ml_threshold):
        recommendation = '🟢 BUY SIGNAL'
        rec_class = 'BUY'
    elif final_score >= SIGNAL_BUY_THRESHOLD:
        recommendation = '🟡 WATCH — score OK but ML gate failed'
        rec_class = 'WATCH'
    elif final_score >= SIGNAL_WATCH_THRESHOLD:
        recommendation = '🟡 WATCH — score below BUY threshold'
        rec_class = 'WATCH'
    else:
        recommendation = '⚪ SKIP — insufficient signal'
        rec_class = 'SKIP'

    # VIX hard block warning
    vix_spot = macro.get('vix_9d', 0)  # closest proxy
    if vix_spot >= VIX_HARD_BLOCK:
        recommendation = f'🚫 BLOCKED — VIX={vix_spot:.1f} >= {VIX_HARD_BLOCK}'
        rec_class = 'BLOCKED'

    # ── 10b. Fundamentals (advisory context — does NOT change trading logic) ──
    try:
        from ml.fundamentals import get_ticker as get_fundamentals
        fund = get_fundamentals(ticker)
    except Exception as e:
        print(f'  [advise] fundamentals fetch failed: {e}')
        fund = {}

    result = {
        'ticker':         ticker,
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
        'fundamentals':   fund,
    }
    _print_report(result, indicators, bundle_info, verbose=verbose)
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
        # Fair value estimate (growth-adjusted, independent of analyst targets)
        try:
            from ml.fundamentals import compute_fair_value
            fv = compute_fair_value(pd.Series(fund))
            if fv:
                print(f'\n    FAIR VALUE ESTIMATE (growth-adjusted):')
                print(f'      Forward EPS:       ${fv["forward_eps"]:.2f}  (price / forward_PE)')
                print(f'      Sustainable growth: {fv["sustainable_growth"]*100:+.1f}%  (60% rev + 40% earn)')
                tier = ('hyper-growth' if fv['sustainable_growth'] > 0.30
                          else 'strong growth' if fv['sustainable_growth'] > 0.15
                          else 'mature' if fv['sustainable_growth'] > 0.05
                          else 'slow' if fv['sustainable_growth'] > 0
                          else 'declining')
                print(f'      Fair PE for tier:   {fv["fair_pe"]}x  ({tier})')
                print(f'      Fair value:         ${fv["fair_value"]:.2f}')
                disc = fv["discount_to_fair"]
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
    print(f'\n  RECOMMENDATION: {"🟢 BUY SIGNAL" if r["recommendation"] == "BUY" else "🟡 WATCH" if r["recommendation"] == "WATCH" else "⚪ SKIP" if r["recommendation"] == "SKIP" else "🚫 BLOCKED"}')

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

    # ── Historical context ────────────────────────────────────────────────
    print(f'\n  HISTORICAL CONTEXT (walk-forward OOS, 60-mo history):')
    print(f'    Rule-only mean vs QQQ:   +11.23% per 6-month period, 4/5 folds beat QQQ')
    print(f'    Rule+ML mean vs QQQ:     +7.72% per 6-month period, Sharpe 1.04')
    print(f'    Typical win/loss:        avg TP hit +10%, avg SL hit -3.5%')
    print(f'    Backtest win rate:       ~37% (asymmetric bracket: 3:1 reward/risk)')

    # ── Caveats ──────────────────────────────────────────────────────────
    print(f'\n  CAVEATS:')
    if r['recommendation'] == 'BUY':
        print(f'    - Phase 0 clearance NOT checked here (earnings / macro veto)')
        print(f'    - Assumes {r["equity"]:,.0f} equity; adjust with --equity')
        print(f'    - Rule-only variant has BEST return; consider ignoring ML gate for max upside')
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
