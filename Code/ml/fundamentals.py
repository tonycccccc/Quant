"""Fundamentals module: valuation context for semiconductor and SaaS stocks.

This is an advisory screening model, not a forecast or trade signal. It uses
yfinance fields, which can be stale or inconsistently defined. Inspect the
returned assumptions and data quality before relying on an output.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import warnings
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, WATCHLIST, ML_EXTRA_TRAINING_SYMBOLS

FUNDAMENTALS_CACHE_PATH = MODELS_DIR / 'fundamentals.parquet'
_ALL_SYMBOLS = list(dict.fromkeys(list(WATCHLIST.keys()) + list(ML_EXTRA_TRAINING_SYMBOLS)))
DEFAULT_MARGIN_OF_SAFETY = 0.25
_MEMORY_TICKERS = {'MU', 'WDC', 'STX', 'SNDK'}
SEGMENT_DISCLOSURES_PATH = Path(__file__).parent / 'segment_disclosures.json'
# Multi-segment companies: a single consolidated DCF would blend businesses
# with different growth, margins and capital intensity, so they are valued
# only via sum-of-parts from filed segment data.
_SEGMENT_REQUIRED = {
    'MSFT': 'Productivity & Business Processes, Intelligent Cloud, More Personal Computing',
}


def _num(value, default=np.nan):
    """Return a finite float or a default; handles None, pandas NA and inf."""
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _row_num(row, key, default=np.nan):
    return _num(row.get(key, default), default)


def _quarterly_fcf_ttm(ticker) -> float:
    """Sum the latest four quarterly FCF observations to align with TTM revenue."""
    try:
        cashflow = ticker.quarterly_cashflow
        if cashflow is None or cashflow.empty:
            return np.nan
        label = next((name for name in ('Free Cash Flow', 'FreeCashFlow')
                      if name in cashflow.index), None)
        if label:
            values = pd.to_numeric(cashflow.loc[label], errors='coerce').dropna()
            values = values.sort_index(ascending=False).head(4)
            return float(values.sum()) if len(values) == 4 else np.nan
        # Fallback when the provider omits its FCF row: CFO + CapEx (CapEx
        # is generally represented as a negative cash flow in yfinance).
        cfo = next((name for name in ('Operating Cash Flow', 'OperatingCashFlow')
                    if name in cashflow.index), None)
        capex = next((name for name in ('Capital Expenditure', 'CapitalExpenditures')
                      if name in cashflow.index), None)
        if cfo and capex:
            cfo_values = pd.to_numeric(cashflow.loc[cfo], errors='coerce').dropna().sort_index(ascending=False).head(4)
            capex_values = pd.to_numeric(cashflow.loc[capex], errors='coerce').dropna().sort_index(ascending=False).head(4)
            if len(cfo_values) == 4 and len(capex_values) == 4:
                return float(cfo_values.sum() + capex_values.sum())
    except Exception:
        pass
    return np.nan


def _fetch_one_ticker(ticker: str) -> dict:
    import yfinance as yf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = yf.Ticker(ticker)
            info = t.info or {}
    except Exception as exc:
        print(f'  [fundamentals] {ticker}: fetch failed: {exc}')
        return {'ticker': ticker, 'fetched_at': datetime.now().isoformat(), 'fetch_ok': False}
    if not info or info.get('quoteType') is None:
        return {'ticker': ticker, 'fetched_at': datetime.now().isoformat(), 'fetch_ok': False}

    def get(key):
        return _num(info.get(key))

    price = get('currentPrice')
    if not np.isfinite(price) or price <= 0:
        price = get('regularMarketPrice')

    # Annual reported revenue history. Useful as a company-specific anchor,
    # though acquisitions and fiscal-year changes can distort these growth rates.
    history = []
    try:
        statement = t.income_stmt
        if statement is not None and not statement.empty:
            label = next((k for k in ('Total Revenue', 'TotalRevenue') if k in statement.index), None)
            if label:
                values = pd.to_numeric(statement.loc[label], errors='coerce').dropna().sort_index()
                history = [float(v) for v in values.tail(6).tolist() if np.isfinite(v) and v > 0]
    except Exception:
        history = []
    annual_growths = [history[i] / history[i - 1] - 1 for i in range(1, len(history)) if history[i - 1] > 0]
    historical_growth_median = float(np.median(annual_growths)) if annual_growths else np.nan
    historical_growth_volatility = float(np.std(annual_growths, ddof=1)) if len(annual_growths) > 1 else np.nan
    fcf_ttm = _quarterly_fcf_ttm(t)
    fcf_source = 'quarterly_cashflow_ttm'
    if not np.isfinite(fcf_ttm):
        fcf_ttm = get('freeCashflow')
        fcf_source = 'yfinance_info_fallback'
    return {
        'ticker': ticker, 'fetched_at': datetime.now().isoformat(), 'fetch_ok': True,
        'sector': info.get('sector', ''), 'industry': info.get('industry', ''),
        'market_cap': get('marketCap'), 'current_price': price,
        'forward_pe': get('forwardPE'), 'trailing_pe': get('trailingPE'),
        'peg_ratio': get('trailingPegRatio') if np.isfinite(get('trailingPegRatio')) else get('pegRatio'),
        'price_to_book': get('priceToBook'), 'price_to_sales': get('priceToSalesTrailing12Months'),
        'revenue_growth': get('revenueGrowth'), 'earnings_growth': get('earningsGrowth'),
        'historical_revenue_growth_median': historical_growth_median,
        'historical_revenue_growth_volatility': historical_growth_volatility,
        # yfinance does not expose RPO/backlog consistently. Populate these
        # from filings or earnings releases when available; leave missing otherwise.
        'rpo': np.nan, 'rpo_next_12m': np.nan, 'rpo_growth': np.nan,
        'profit_margin': get('profitMargins'), 'operating_margin': get('operatingMargins'),
        'gross_margin': get('grossMargins'), 'roe': get('returnOnEquity'), 'roa': get('returnOnAssets'),
        'enterprise_value': get('enterpriseValue'), 'total_revenue': get('totalRevenue'),
        'ebitda': get('ebitda'), 'ev_to_ebitda': get('enterpriseToEbitda'),
        'ev_to_revenue': get('enterpriseToRevenue'), 'total_debt': get('totalDebt'),
        'total_cash': get('totalCash'), 'shares_outstanding': get('sharesOutstanding'),
        'debt_to_equity': get('debtToEquity'), 'current_ratio': get('currentRatio'),
        'free_cashflow': fcf_ttm, 'free_cashflow_source': fcf_source,
        'analyst_rec_mean': get('recommendationMean'),
        'analyst_target': get('targetMeanPrice'), 'num_analysts': get('numberOfAnalystOpinions'),
        'beta': get('beta'),
    }


def _load_disclosures() -> dict:
    if not SEGMENT_DISCLOSURES_PATH.exists():
        return {}
    return json.loads(SEGMENT_DISCLOSURES_PATH.read_text(encoding='utf-8'))


def refresh_all(force: bool = False, disclosures: dict | None = None) -> pd.DataFrame:
    if disclosures is None:
        disclosures = _load_disclosures()
    if not force and FUNDAMENTALS_CACHE_PATH.exists():
        age = (datetime.now().timestamp() - FUNDAMENTALS_CACHE_PATH.stat().st_mtime) / 3600
        if age < 24:
            cached = pd.read_parquet(FUNDAMENTALS_CACHE_PATH)
            if len(cached) >= len(_ALL_SYMBOLS) * 0.8:
                cached = _merge_disclosures(cached, disclosures)
                if disclosures:
                    cached['quality_score'] = cached.apply(_compute_quality_score, axis=1)
                return cached
    rows = [_fetch_one_ticker(t) for t in _ALL_SYMBOLS]
    df = pd.DataFrame(rows).set_index('ticker')
    df = _merge_disclosures(df, disclosures)
    df['quality_score'] = df.apply(_compute_quality_score, axis=1)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUNDAMENTALS_CACHE_PATH)
    return df


def _merge_disclosures(df: pd.DataFrame, disclosures: dict | None) -> pd.DataFrame:
    """Overlay filing/release fields absent from yfinance (e.g. RPO).

    Shape: {'AVGO': {'rpo': 179200, 'rpo_next_12m': 44800,
                     'rpo_growth': 0.25}}. Amounts use the same currency
    and scale as total_revenue/free_cashflow (yfinance: usually USD).
    Update the figures each reporting period; stale contracts are not useful.
    """
    if not disclosures:
        return df
    allowed = {'rpo', 'rpo_next_12m', 'rpo_growth', 'segments'}
    df = df.copy()
    for ticker, fields in disclosures.items():
        ticker = str(ticker).upper()
        if ticker not in df.index:
            continue
        for key, value in fields.items():
            if key in allowed:
                # Store nested segment data as JSON so Parquet has a stable,
                # portable scalar representation across refreshes.
                df.loc[ticker, key] = json.dumps(value) if key == 'segments' and isinstance(value, (dict, list)) else (_num(value) if key != 'segments' else value)
    return df


def get_ticker(ticker: str, force_refresh: bool = False) -> dict:
    ticker = ticker.upper()
    df = refresh_all(force=force_refresh)
    if ticker in df.index:
        result = df.loc[ticker].to_dict()
        result['ticker'] = ticker
        return result
    row = _fetch_one_ticker(ticker)
    row = _merge_disclosures(pd.DataFrame([row]).set_index('ticker'),
                             _load_disclosures()).reset_index().iloc[0].to_dict()
    row['quality_score'] = _compute_quality_score(pd.Series(row))
    return row


def compute_fair_value(row: pd.Series, margin_of_safety: float = DEFAULT_MARGIN_OF_SAFETY) -> dict:
    """Return intrinsic-value estimate and a margin-of-safety buy price.

    Company-reported FCF is used as a proxy for FCFF in semi/SaaS DCFs. That
    proxy is not perfectly unlevered; the EV-to-equity bridge may therefore
    double count financing effects. Outputs are screening estimates only.
    """
    if not 0 <= margin_of_safety < 1:
        raise ValueError('margin_of_safety must be in [0, 1)')
    segments = _parse_segments(row.get('segments'))
    ticker = str(row.get('ticker', getattr(row, 'name', '')) or '').upper()
    if not segments and ticker in _SEGMENT_REQUIRED:
        return {'method': 'sum_of_parts',
                'data_quality': (f'{ticker} reports segments ({_SEGMENT_REQUIRED[ticker]}); '
                                 f'add them to {SEGMENT_DISCLOSURES_PATH.name} for a sum-of-parts value')}
    method = 'sum_of_parts' if segments else _pick_method(row)
    if segments:
        result = _value_segments(row, segments)
    else:
        result = _value_dcf(row, method) if method in {'semi', 'saas'} else _value_default(row)
    if result and np.isfinite(_num(result.get('fair_value'))):
        result['method'] = method
        result['margin_of_safety'] = margin_of_safety
        result['buy_price'] = result['fair_value'] * (1 - margin_of_safety)
        price = _row_num(row, 'current_price')
        result['upside_to_fair_value'] = (result['fair_value'] / price - 1) if price > 0 else np.nan
        result['buy_price_gap'] = (result['buy_price'] / price - 1) if price > 0 else np.nan
    return result or {}


def _parse_segments(value) -> list[dict]:
    """Read optional segment disclosures; each segment must have its own cash flow."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, dict):
        value = value.get('segments', [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict) and item.get('name')]


def _value_segments(row: pd.Series, segments: list[dict]) -> dict:
    """Sum segment FCFF DCFs, then bridge enterprise value to equity once.

    Segment schema (amounts in the same currency/scale):
      name, revenue, revenue_growth, fcff; optional EBIT, tax_rate, D&A,
      capex, change_working_capital, wacc, terminal_growth, forecast_years.
    FCFF may be supplied directly or derived from EBIT(1-tax)+D&A-capex-ΔWC.
    Segment EVs are additive only when segment coverage is complete and
    revenues/cash flows are non-overlapping.
    """
    rows = []
    for segment in segments:
        name = str(segment['name'])
        revenue = _num(segment.get('revenue'))
        growth = _num(segment.get('revenue_growth'))
        fcff = _num(segment.get('fcff'))
        if not np.isfinite(fcff):
            ebit = _num(segment.get('ebit'))
            tax = _num(segment.get('tax_rate'), 0.21)
            da = _num(segment.get('da'), 0.0)
            capex = _num(segment.get('capex'), 0.0)
            delta_wc = _num(segment.get('change_working_capital'), 0.0)
            if np.isfinite(ebit):
                fcff = ebit * (1 - np.clip(tax, 0, 0.60)) + da - capex - delta_wc
        if not (revenue > 0 and np.isfinite(growth) and np.isfinite(fcff) and fcff > 0):
            return {'data_quality': f'incomplete segment inputs: {name}', 'segments_valued': len(rows)}
        wacc = _num(segment.get('wacc'), 0.10)
        terminal_growth = _num(segment.get('terminal_growth'), 0.025)
        years = int(_num(segment.get('forecast_years'), 5))
        if years < 2 or wacc <= terminal_growth or terminal_growth < -0.02 or wacc > 0.30:
            return {'data_quality': f'invalid DCF parameters: {name}', 'segments_valued': len(rows)}
        margin = fcff / revenue
        growth_end = _num(segment.get('terminal_revenue_growth'), terminal_growth)
        growth_start = float(np.clip(growth, -0.30, 0.50))
        growth_end = float(np.clip(growth_end, -0.05, 0.08))
        pv = 0.0
        revenue_y, fcff_y = revenue, fcff
        for year in range(1, years + 1):
            g = growth_start + (growth_end - growth_start) * (year - 1) / (years - 1)
            revenue_y *= 1 + g
            fcff_y = revenue_y * margin
            pv += fcff_y / (1 + wacc) ** year
        terminal_value = fcff_y * (1 + terminal_growth) / (wacc - terminal_growth)
        ev = pv + terminal_value / (1 + wacc) ** years
        rows.append({'name': name, 'revenue': revenue, 'fcff': fcff,
                     'growth_start': growth_start, 'growth_end': growth_end,
                     'wacc': wacc, 'terminal_growth': terminal_growth,
                     'enterprise_value': float(ev)})
    total_revenue = sum(x['revenue'] for x in rows)
    reported_revenue = _row_num(row, 'total_revenue')
    coverage = total_revenue / reported_revenue if reported_revenue > 0 else np.nan
    # Segment cash flows are FCFF, so financing claims are bridged once here.
    equity = sum(x['enterprise_value'] for x in rows) - max(0, _row_num(row, 'total_debt', 0)) + max(0, _row_num(row, 'total_cash', 0))
    shares = _row_num(row, 'shares_outstanding')
    price = _row_num(row, 'current_price')
    if not (shares > 0 and equity > 0):
        return {'data_quality': 'missing shares or non-positive equity value', 'segments': rows,
                'segment_revenue_coverage': coverage}
    fair_value = equity / shares
    if price > 0 and fair_value > price * 5:
        return {'data_quality': 'implausible output; review segment inputs', 'segments': rows,
                'segment_revenue_coverage': coverage}
    coverage_note = ('segment revenue coverage below 90%; likely incomplete segment disclosures'
                     if np.isfinite(coverage) and coverage < 0.90 else
                     'segment revenue coverage exceeds reported company revenue; check overlap/units'
                     if np.isfinite(coverage) and coverage > 1.10 else
                     'segment revenue coverage consistent with reported revenue'
                     if np.isfinite(coverage) else 'company revenue unavailable; coverage not assessed')
    return {'fair_value': float(fair_value), 'enterprise_value': float(sum(x['enterprise_value'] for x in rows)),
            'equity_value': float(equity), 'segments': rows, 'segments_valued': len(rows),
            'segment_revenue_coverage': coverage,
            'data_quality': f'segment FCFF DCF; {coverage_note}; verify currency and non-overlap'}


def _pick_method(row: pd.Series) -> str:
    industry = str(row.get('industry', '') or '').lower()
    ticker = str(row.get('ticker', getattr(row, 'name', '')) or '').upper()
    if 'semiconductor' in industry or ticker in _MEMORY_TICKERS:
        return 'semi'
    if ticker in {'NET', 'DDOG', 'CRWD', 'PANW', 'SNOW', 'MDB', 'CRM', 'NOW'}:
        return 'saas'
    return 'default'


def _value_dcf(row: pd.Series, method: str) -> dict:
    """Two-stage FCFF-proxy DCF with conservative guards and visible inputs."""
    revenue = _row_num(row, 'total_revenue')
    fcf = _row_num(row, 'free_cashflow')
    growth = _row_num(row, 'revenue_growth')
    price = _row_num(row, 'current_price')
    shares = _row_num(row, 'shares_outstanding')
    debt = max(0.0, _row_num(row, 'total_debt', 0.0))
    cash = max(0.0, _row_num(row, 'total_cash', 0.0))
    if not (revenue > 0 and fcf > 0 and shares > 0 and price > 0):
        return {}
    historical_median = _row_num(row, 'historical_revenue_growth_median')
    historical_vol = _row_num(row, 'historical_revenue_growth_volatility')
    if not np.isfinite(growth) and not np.isfinite(historical_median):
        return {}

    # Historical growth level sets the year-5 anchor. Recent growth informs
    # year 1; historical volatility is used to cap the plausible range, not as
    # a growth rate. This avoids a universal 5% year-5 assumption.
    if not np.isfinite(historical_median):
        historical_median = growth
    if not np.isfinite(growth):
        growth = historical_median
    growth_end = float(np.clip(historical_median, -0.20, 0.40))
    growth_start = 0.40 * growth + 0.60 * historical_median

    # Optional RPO/bookings signal. It nudges the near-term growth anchor only
    # when the comparable RPO growth rate is supplied; absolute RPO is not
    # added to revenue, because contracted revenue is already in the base.
    rpo_growth = _row_num(row, 'rpo_growth')
    rpo_adjustment = float(np.clip(0.10 * rpo_growth, -0.05, 0.05)) if np.isfinite(rpo_growth) else 0.0
    growth_start += rpo_adjustment

    # Keep the near-term estimate within one historical standard deviation
    # of the historical median where enough annual observations exist.
    if np.isfinite(historical_vol) and historical_vol > 0:
        growth_start = float(np.clip(growth_start, historical_median - historical_vol,
                                     historical_median + historical_vol))
    growth_start = float(np.clip(growth_start, -0.20, 0.40))
    if method == 'semi':
        wacc, terminal_growth, horizon = 0.10, 0.025, 5
        segment = _semi_segment(row)
        # Memory is more cyclical. Discount current FCF when forward earnings
        # imply a peak; do not increase FCF for a perceived trough.
        fwd_pe = _row_num(row, 'forward_pe')
        cycle_factor = 0.70 if segment == 'memory' and 0 < fwd_pe < 10 else 1.0
        method_detail = f'{segment}; reported FCF proxy; cycle factor {cycle_factor:.2f}'
    else:
        wacc, terminal_growth, horizon = 0.10, 0.025, 5
        cycle_factor = 1.0
        method_detail = 'SaaS; reported FCF margin held/improved gradually, capped at 30%'

    fcf_margin = fcf / revenue
    if method == 'saas':
        # Avoid assuming an extreme turnaround: negative/very low FCF is not
        # valued by this route; positive margins converge at most to 30%.
        target_margin = max(fcf_margin, 0.20)
        target_margin = min(target_margin, 0.30)
        margin_start = fcf_margin
    else:
        margin_start = fcf_margin
        target_margin = fcf_margin

    fcf_base = fcf * cycle_factor
    pv = 0.0
    revenue_y = revenue
    fcf_y = fcf_base
    for year in range(1, horizon + 1):
        g = growth_start + (growth_end - growth_start) * (year - 1) / (horizon - 1)
        revenue_y *= (1 + g)
        margin_y = margin_start + (target_margin - margin_start) * year / horizon
        fcf_y = revenue_y * margin_y * cycle_factor
        pv += fcf_y / (1 + wacc) ** year
    tv = fcf_y * (1 + terminal_growth) / (wacc - terminal_growth)
    enterprise_value = pv + tv / (1 + wacc) ** horizon
    equity_value = enterprise_value - debt + cash
    fair_value = equity_value / shares
    if not np.isfinite(fair_value) or fair_value <= 0 or fair_value > price * 5:
        return {}
    return {
        'fair_value': float(fair_value), 'discount_to_fair': float(fair_value / price - 1),
        'enterprise_value': float(enterprise_value), 'equity_value': float(equity_value),
        'fcf_margin': float(fcf_margin), 'growth_start': growth_start,
        'growth_end': growth_end, 'wacc': wacc, 'terminal_growth': terminal_growth,
        'historical_growth_median': historical_median,
        'historical_growth_volatility': historical_vol,
        'rpo_growth_adjustment': rpo_adjustment,
        'rpo': _row_num(row, 'rpo'), 'rpo_next_12m': _row_num(row, 'rpo_next_12m'),
        'rpo_12m_coverage': (_row_num(row, 'rpo_next_12m') / revenue
                             if np.isfinite(_row_num(row, 'rpo_next_12m')) and revenue > 0 else np.nan),
        'method_detail': method_detail,
        'data_quality': 'proxy: company-reported FCF may not equal unlevered FCFF',
    }


def _semi_segment(row):
    ticker = str(row.get('ticker', getattr(row, 'name', '')) or '').upper()
    industry = str(row.get('industry', '') or '').lower()
    if ticker in _MEMORY_TICKERS or 'memory' in industry:
        return 'memory'
    if ticker in {'ASML', 'AMAT', 'LRCX', 'KLAC'} or 'equipment' in industry:
        return 'equipment'
    if ticker == 'TSM':
        return 'foundry'
    return 'design/logic'


def _value_default(row: pd.Series) -> dict:
    """Simple growth-tier P/E screen for companies outside the focus sectors."""
    fwd_pe = _row_num(row, 'forward_pe')
    price = _row_num(row, 'current_price')
    if not (fwd_pe > 0 and price > 0):
        return {}
    rev_g = _row_num(row, 'revenue_growth')
    earn_g = _row_num(row, 'earnings_growth')
    growths = [x for x in (rev_g, earn_g) if np.isfinite(x)]
    if not growths:
        return {}
    growth = float(np.clip(np.mean(growths), -0.30, 0.40))
    base_pe = 35 if growth > .30 else 25 if growth > .15 else 18 if growth > .05 else 12 if growth > 0 else 8
    roe = _row_num(row, 'roe')
    roe_mult = 1.20 if roe > .30 else 1.10 if roe > .20 else 1.0 if roe > .10 else .90
    fair_value = (price / fwd_pe) * base_pe * roe_mult
    return {'fair_value': float(fair_value), 'discount_to_fair': float(fair_value / price - 1),
            'method_detail': 'growth-tier P/E screen; lower confidence than sector DCF'}


def ticker_is_memory(row) -> bool:
    return str(row.get('ticker', getattr(row, 'name', '')) or '').upper() in _MEMORY_TICKERS


def _compute_quality_score(row: pd.Series) -> float:
    """Legacy screening score retained for compatibility; not a buy-price input."""
    if not row.get('fetch_ok', True):
        return np.nan
    score = 0.0
    rev = _row_num(row, 'revenue_growth')
    earn = _row_num(row, 'earnings_growth')
    if np.isfinite(rev): score += 15 if rev > .30 else 10 if rev > .15 else 5 if rev > .05 else 0
    if np.isfinite(earn): score += 10 if earn > .20 else 5 if earn > .05 else 0
    pe = _row_num(row, 'forward_pe')
    if pe > 0: score += 12 if pe < 15 else 8 if pe < 25 else 4 if pe < 40 else 0
    peg = _row_num(row, 'peg_ratio')
    if peg > 0: score += 8 if peg < 1 else 3 if peg < 2 else -3 if peg > 3 else 0
    roe = _row_num(row, 'roe')
    pm = _row_num(row, 'profit_margin')
    om = _row_num(row, 'operating_margin')
    if np.isfinite(roe): score += 12 if roe > .30 else 9 if roe > .20 else 5 if roe > .12 else 2 if roe > .05 else 0
    if np.isfinite(pm): score += 8 if pm > .25 else 5 if pm > .15 else 2 if pm > .05 else 0
    if np.isfinite(om): score += 5 if om > .20 else 2 if om > .10 else 0
    de = _row_num(row, 'debt_to_equity')
    if np.isfinite(de):
        de = de / 100 if de > 5 else de
        score += 10 if de < .5 else 7 if de < 1 else 3 if de < 2 else 0
    if _row_num(row, 'free_cashflow') > 0: score += 5
    rec = _row_num(row, 'analyst_rec_mean')
    if np.isfinite(rec): score += 10 if rec < 1.5 else 7 if rec < 2 else 3 if rec < 2.5 else 0
    fv = compute_fair_value(row)
    if fv:
        d = fv.get('discount_to_fair', 0)
        score += 5 if d > .20 else 3 if d > .05 else -10 if d < -.30 else -5 if d < -.15 else 0
    return float(np.clip(round(score, 1), 0, 100))


def quality_label(score: float) -> tuple:
    if pd.isna(score): return ('⚪', 'UNKNOWN', 'fundamentals data unavailable')
    if score >= 70: return ('🟢', 'STRONG', 'higher screening score')
    if score >= 50: return ('🟡', 'DECENT', 'mixed screening results')
    if score >= 30: return ('🟠', 'WEAK', 'weaker screening results')
    return ('🔴', 'POOR', 'low screening score')


def leaderboard(filter_mode: str = 'all') -> pd.DataFrame:
    df = refresh_all()
    if 'fetch_ok' in df: df = df[df['fetch_ok'] == True]
    if filter_mode == 'strong': df = df[df['quality_score'] >= 70]
    df = df.sort_values('quality_score', ascending=False)
    for ticker, row in df.iterrows():
        fv = compute_fair_value(row)
        print(f"{ticker:<6} score={row.get('quality_score', np.nan):>5}  "
              f"price={_row_num(row, 'current_price'):>8.2f}  "
              f"fair={fv.get('fair_value', np.nan):>8.2f}  "
              f"buy@25%MoS={fv.get('buy_price', np.nan):>8.2f}  "
              f"{fv.get('method_detail', 'N/A')}")
    return df
