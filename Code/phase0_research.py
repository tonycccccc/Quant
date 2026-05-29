"""
Phase 0: Pre-Market Clearance Engine

Data source: Yahoo Finance (yfinance) — no API keys required.
Logic: pure rule-based — no LLM.

For each watchlist stock:
  1. Pull earnings calendar, analyst rating, news, and 5-day price momentum from yfinance
  2. Apply hard veto rules:
       - Earnings within 3 trading days  → clearance = 0
       - Structural damage keyword in headlines → clearance = 0
  3. Derive regime_bias from analyst consensus + price momentum
  4. Write daily_clearance row to SQLite
  5. Save research summary to Metholody/Research/{TICKER}.md
  6. Send Discord "Daily Battle Plan" embed
"""
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytz
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from config import WATCHLIST, RESEARCH_DIR
import database as db
import discord_client as discord
from macro_calendar import is_macro_event_day

ET = pytz.timezone('America/New_York')

# Keywords that trigger an immediate clearance = 0 (structural damage)
DAMAGE_KEYWORDS = [
    'sec investigation', 'sec charges', 'fraud', 'bankruptcy',
    'product recall', 'class action', 'delisted', 'criminal charges',
    'subpoena', 'accounting irregularit',
]

# ── Yahoo Finance data fetch ───────────────────────────────────────────────

def _get_next_earnings(ticker_obj: yf.Ticker) -> tuple:
    """
    Returns (next_earnings_date, days_away).
    Returns (None, None) if no earnings data is found.
    """
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return None, None

        dates = []
        if isinstance(cal, dict):
            raw = cal.get('Earnings Date', [])
            dates = list(raw) if raw is not None else []
        elif hasattr(cal, 'index') and 'Earnings Date' in cal.index:
            # Older yfinance returns a transposed DataFrame
            val = cal.loc['Earnings Date']
            dates = list(val) if hasattr(val, '__iter__') else [val]

        if not dates:
            return None, None

        now = datetime.now(ET)
        for d in dates:
            ts = pd.Timestamp(d)
            if ts.tzinfo is None:
                ts = ts.tz_localize('America/New_York')
            else:
                ts = ts.tz_convert('America/New_York')
            days_away = (ts.date() - now.date()).days
            if days_away >= 0:
                return ts, days_away
    except Exception:
        pass
    return None, None


def _get_analyst_signal(ticker_obj: yf.Ticker) -> tuple:
    """
    Returns (regime_bias, confidence) from analyst consensus.
    recommendationMean: 1.0 = Strong Buy … 5.0 = Sell
    """
    try:
        info = ticker_obj.info or {}
        rec_key  = (info.get('recommendationKey') or 'hold').lower().strip()
        rec_mean = float(info.get('recommendationMean') or 3.0)

        if rec_key in ('strong buy', 'buy') or rec_mean <= 2.0:
            return 'bullish', min(0.5 + (2.5 - rec_mean) * 0.1, 0.9)
        if rec_key in ('sell', 'strong sell', 'underperform') or rec_mean >= 4.0:
            return 'bearish', min(0.5 + (rec_mean - 3.5) * 0.1, 0.9)
        return 'neutral', 0.5
    except Exception:
        return 'neutral', 0.5


def _get_news_headlines(ticker_obj: yf.Ticker) -> list:
    """Return list of recent news titles (last ~20 items)."""
    headlines = []
    try:
        items = ticker_obj.news or []
        for item in items[:20]:
            if isinstance(item, dict):
                # New yfinance format nests under 'content'
                if 'content' in item and isinstance(item['content'], dict):
                    title = item['content'].get('title', '')
                else:
                    title = item.get('title', '')
                if title:
                    headlines.append(title)
    except Exception:
        pass
    return headlines


def _get_price_momentum(ticker_obj: yf.Ticker) -> float:
    """5-day price return in percent. Returns 0.0 on failure."""
    try:
        hist = ticker_obj.history(period='7d', interval='1d')
        if len(hist) >= 2:
            return float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)
    except Exception:
        pass
    return 0.0


# ── Rule-based clearance decision ─────────────────────────────────────────

def _evaluate_clearance(ticker: str, company: str) -> tuple:
    """
    Pull Yahoo Finance data and apply all clearance rules.
    Returns (result_dict, raw_summary_text).
    """
    ticker_obj = yf.Ticker(ticker)

    clearance     = 1
    earnings_risk = False
    key_catalyst  = 'No significant catalyst'
    notes         = []

    # ── 1. Earnings veto ──────────────────────────────────────────────────
    earnings_date, days_away = _get_next_earnings(ticker_obj)
    if earnings_date is not None:
        label = f"Earnings: {earnings_date.strftime('%Y-%m-%d')} ({days_away}d away)"
        notes.append(label)
        if days_away <= 3:
            clearance     = 0
            earnings_risk = True
            key_catalyst  = f"Earnings in {days_away} trading day(s) — DO NOT TRADE"
    else:
        notes.append('Earnings: date unavailable')

    # ── 2. Macro event veto ───────────────────────────────────────────────
    is_macro, macro_desc = is_macro_event_day()
    if is_macro:
        clearance    = 0
        key_catalyst = f"Macro event: {macro_desc} — DO NOT TRADE"
        notes.append(f"MACRO VETO: {macro_desc}")

    # ── 3. Analyst consensus ──────────────────────────────────────────────
    analyst_bias, analyst_conf = _get_analyst_signal(ticker_obj)
    try:
        mean_val = ticker_obj.info.get('recommendationMean', 'N/A')
        notes.append(f"Analyst: {analyst_bias} (mean={mean_val})")
    except Exception:
        notes.append(f"Analyst: {analyst_bias}")

    # ── 4. News — structural damage scan ─────────────────────────────────
    headlines = _get_news_headlines(ticker_obj)
    damage_hit = False
    for title in headlines:
        title_lower = title.lower()
        for kw in DAMAGE_KEYWORDS:
            if kw in title_lower:
                clearance   = 0
                damage_hit  = True
                key_catalyst = f"Structural risk: '{kw}' detected in headlines"
                notes.append(f"NEWS ALERT: {title[:100]}")
                break
        if damage_hit:
            break

    if not damage_hit and headlines:
        notes.append(f"Top news: {headlines[0][:90]}")

    # ── 5. Price momentum ─────────────────────────────────────────────────
    momentum = _get_price_momentum(ticker_obj)
    notes.append(f"5d momentum: {momentum:+.1f}%")

    # Combine analyst signal and price momentum for final regime_bias
    if analyst_bias == 'bullish' or momentum > 5:
        regime_bias = 'bullish'
        confidence  = max(analyst_conf, 0.6 if momentum > 5 else 0.5)
    elif analyst_bias == 'bearish' or momentum < -5:
        regime_bias = 'bearish'
        confidence  = max(analyst_conf, 0.6 if momentum < -5 else 0.5)
    else:
        regime_bias = 'neutral'
        confidence  = 0.5

    if key_catalyst == 'No significant catalyst':
        if momentum > 3 and analyst_bias in ('bullish', 'neutral'):
            key_catalyst = f"+{momentum:.1f}% 5-day momentum with {analyst_bias} analyst consensus"
        elif momentum < -3:
            key_catalyst = f"{momentum:.1f}% 5-day pullback — monitor for reversal"
        else:
            key_catalyst = f"Analyst consensus: {analyst_bias}"

    status  = 'CLEARED' if clearance == 1 else 'BLOCKED'
    summary = f"{status}: {key_catalyst}. " + "; ".join(notes[:2])

    raw_text = "\n".join([
        f"Ticker:           {ticker} ({company})",
        f"Earnings:         {earnings_date.strftime('%Y-%m-%d') if earnings_date else 'N/A'} ({days_away}d)",
        f"Macro event:      {macro_desc if is_macro else 'None within 1 day'}",
        f"Analyst bias:     {analyst_bias} (conf {confidence:.0%})",
        f"5d momentum:      {momentum:+.1f}%",
        f"Damage keywords:  {'YES — ' + key_catalyst if damage_hit else 'None found'}",
        f"Headlines:",
    ] + [f"  - {h}" for h in headlines[:5]])

    result = {
        'daily_clearance': clearance,
        'regime_bias':     regime_bias,
        'confidence':      confidence,
        'summary':         summary[:300],
        'earnings_risk':   earnings_risk,
        'key_catalyst':    key_catalyst,
    }
    return result, raw_text


# ── Research file writer ───────────────────────────────────────────────────

def _save_research_file(ticker: str, company: str, raw_text: str,
                        result: dict, date: str):
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    icon = '✅ CLEARED' if result['daily_clearance'] == 1 else '🚫 BLOCKED'
    content = (
        f"# {company} ({ticker}) — Research Report\n"
        f"Date: {date}\n"
        f"Status: {icon}\n\n"
        f"## Decision Summary\n{result['summary']}\n\n"
        f"## Key Catalyst\n{result['key_catalyst']}\n\n"
        f"## Regime Bias\n"
        f"{result['regime_bias'].capitalize()} — confidence {result['confidence']:.0%}\n\n"
        f"## Earnings Risk\n"
        f"{'⚠️ YES — DO NOT TRADE near earnings' if result['earnings_risk'] else 'No earnings risk in next 3 days'}\n\n"
        f"## Yahoo Finance Data\n```\n{raw_text}\n```\n\n"
        f"---\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
    )
    (RESEARCH_DIR / f'{ticker}.md').write_text(content, encoding='utf-8')


# ── Public entry point ─────────────────────────────────────────────────────

def run_clearance_engine(dry_run: bool = False) -> list:
    """
    Run Phase 0 for every watchlist ticker.

    dry_run=True  → skips all network calls, uses stub data.
    dry_run=False → fetches live data from Yahoo Finance (no API key needed).
    """
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n{'='*60}")
    print(f"  PHASE 0 — Pre-Market Clearance Engine  [{today}]")
    print(f"  Data source: {'DRY RUN (stub)' if dry_run else 'Yahoo Finance'}")
    print(f"{'='*60}")

    results = []
    for ticker, company in WATCHLIST.items():
        print(f'\n[{ticker}] Fetching Yahoo Finance data for {company}...')

        if dry_run:
            raw_text = f'[DRY RUN] No real data fetched for {ticker}.'
            result = {
                'daily_clearance': 1,
                'regime_bias':     'bullish',
                'confidence':      0.75,
                'summary':         f'Dry run — {ticker} auto-cleared for pipeline testing.',
                'earnings_risk':   False,
                'key_catalyst':    'Dry run mode',
            }
        else:
            try:
                result, raw_text = _evaluate_clearance(ticker, company)
                icon = '✅' if result['daily_clearance'] == 1 else '🚫'
                print(f"  {icon} clearance={result['daily_clearance']} | "
                      f"{result['regime_bias']} ({result['confidence']:.0%}) | "
                      f"{result['key_catalyst'][:60]}")
            except Exception as e:
                print(f'  [ERROR] {e}')
                raw_text = f'Data fetch failed: {e}'
                result = {
                    'daily_clearance': 0,
                    'regime_bias':     'neutral',
                    'confidence':      0.0,
                    'summary':         f'Fetch error: {e}',
                    'earnings_risk':   False,
                    'key_catalyst':    'Error during data fetch',
                }

        db.upsert_clearance(
            ticker=ticker, date=today,
            clearance=result['daily_clearance'],
            summary=result['summary'],
            raw_research=raw_text,
            regime_bias=result['regime_bias'],
            confidence=result['confidence'],
            key_catalyst=result['key_catalyst'],
            earnings_risk=result['earnings_risk'],
        )
        _save_research_file(ticker, company, raw_text, result, today)
        results.append({'ticker': ticker, **result})

    cleared = sum(1 for r in results if r['daily_clearance'] == 1)
    print(f'\n[Phase 0] Complete — {cleared}/{len(results)} stocks cleared\n')
    discord.send_battle_plan(db.get_all_today_clearances())
    return results
