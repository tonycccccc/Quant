"""
Dynamic macro event calendar.

Fetches FOMC, CPI, and NFP release dates from official government sources
(federalreserve.gov, bls.gov) using httpx, then caches locally for
CACHE_TTL_DAYS. Falls back to hardcoded 2026 dates if both scraping and
the cache are unavailable.
"""
import json
import re
import httpx
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR   = Path(__file__).parent.parent
CACHE_PATH = BASE_DIR / 'macro_calendar_cache.json'
CACHE_TTL_DAYS = 7

_MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

# ── Web scrapers ───────────────────────────────────────────────────────────

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml',
}


def _fetch_fomc_dates() -> list:
    """
    Scrape FOMC decision dates from the Fed's public calendar page.
    Two-day meetings end on the second day (the decision day).
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
        r.raise_for_status()
        # Matches e.g. "January 27-28, 2026" or "March 18–19, 2026" (en-dash variant)
        pattern = re.compile(
            r'(\w+)\s+(\d{1,2})\s*[-–]\s*(\d{1,2}),\s*(202\d)',
            re.IGNORECASE,
        )
        seen, dates = set(), []
        for m in pattern.finditer(r.text):
            month_str, _, day2, year = m.group(1), m.group(2), m.group(3), m.group(4)
            month = _MONTHS.get(month_str.lower())
            if not month:
                continue
            try:
                d = date(int(year), month, int(day2)).isoformat()
                if d not in seen:
                    seen.add(d)
                    dates.append(d)
            except ValueError:
                pass
        # Also match single-day meetings: "January 29, 2025"
        single = re.compile(r'(\w+)\s+(\d{1,2}),\s*(202\d)', re.IGNORECASE)
        for m in single.finditer(r.text):
            month_str, day, year = m.group(1), m.group(2), m.group(3)
            month = _MONTHS.get(month_str.lower())
            if not month:
                continue
            try:
                d = date(int(year), month, int(day)).isoformat()
                if d not in seen:
                    seen.add(d)
                    dates.append(d)
            except ValueError:
                pass
        result = sorted(dates)
        if result:
            print(f"  [macro_calendar] Fetched {len(result)} FOMC dates from Fed website")
        return result
    except Exception as exc:
        print(f"  [macro_calendar] FOMC fetch failed: {exc}")
        return []


def _fetch_bls_dates(url: str, label: str) -> list:
    """
    Scrape release dates from a BLS schedule page.
    BLS pages list dates as 'January 14, 2026' inside table cells.
    """
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
        r.raise_for_status()
        pattern = re.compile(r'(\w+)\s+(\d{1,2}),\s*(202\d)', re.IGNORECASE)
        seen, dates = set(), []
        for m in pattern.finditer(r.text):
            month_str, day, year = m.group(1), m.group(2), m.group(3)
            month = _MONTHS.get(month_str.lower())
            if not month:
                continue
            try:
                d = date(int(year), month, int(day)).isoformat()
                if d not in seen:
                    seen.add(d)
                    dates.append(d)
            except ValueError:
                pass
        result = sorted(dates)
        if result:
            print(f"  [macro_calendar] Fetched {len(result)} {label} dates from BLS website")
        return result
    except Exception as exc:
        print(f"  [macro_calendar] {label} fetch failed: {exc}")
        return []


# ── Cache management ───────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
        fetched_str = data.get('fetched', '2000-01-01T00:00:00')
        fetched = datetime.fromisoformat(fetched_str)
        age_days = (datetime.now() - fetched).days
        if age_days < CACHE_TTL_DAYS:
            return data
        print(f"  [macro_calendar] Cache is {age_days}d old — refreshing")
    except Exception:
        pass
    return {}


def _save_cache(calendar: dict):
    try:
        calendar['fetched'] = datetime.now().isoformat()
        CACHE_PATH.write_text(json.dumps(calendar, indent=2), encoding='utf-8')
    except Exception as exc:
        print(f"  [macro_calendar] Could not write cache: {exc}")


# ── Fallback (last-resort) ─────────────────────────────────────────────────

def _fallback_dates() -> dict:
    """Known 2026 schedule — used only when both network and cache are unavailable."""
    return {
        'fomc': [
            "2026-01-28", "2026-03-18", "2026-04-29",
            "2026-06-17", "2026-07-29", "2026-09-16",
            "2026-10-28", "2026-12-16",
        ],
        'cpi': [
            "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
            "2026-05-13", "2026-06-10", "2026-07-14", "2026-08-12",
            "2026-09-11", "2026-10-14", "2026-11-12", "2026-12-11",
        ],
        'nfp': [
            "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
            "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
            "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
        ],
    }


# ── Public API ─────────────────────────────────────────────────────────────

def get_macro_dates(force_refresh: bool = False) -> dict:
    """
    Return {'fomc': [...], 'cpi': [...], 'nfp': [...]} with ISO date strings.
    Load order: cache → live web fetch → fallback hardcoded dates.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached.get('fomc') and cached.get('cpi') and cached.get('nfp'):
            return {k: cached[k] for k in ('fomc', 'cpi', 'nfp')}

    print("  [macro_calendar] Fetching macro event calendar from official sources...")
    fomc = _fetch_fomc_dates()
    cpi  = _fetch_bls_dates(
        "https://www.bls.gov/schedule/news_release/cpi.htm", "CPI"
    )
    nfp  = _fetch_bls_dates(
        "https://www.bls.gov/schedule/news_release/empsit.htm", "NFP"
    )

    fallback = _fallback_dates()
    result = {
        'fomc': fomc if fomc else fallback['fomc'],
        'cpi':  cpi  if cpi  else fallback['cpi'],
        'nfp':  nfp  if nfp  else fallback['nfp'],
    }
    _save_cache(result)
    return result


def is_macro_event_day(check_date: date = None, lookahead_days: int = 1) -> tuple:
    """
    Return (True, 'EVENT_TYPE YYYY-MM-DD') if check_date (default: today) or
    any of the next lookahead_days calendar days falls on a macro event.
    Returns (False, '') when the window is clear.
    """
    if check_date is None:
        check_date = date.today()

    calendar = get_macro_dates()
    event_list = (
        [('FOMC', d) for d in calendar.get('fomc', [])] +
        [('CPI',  d) for d in calendar.get('cpi',  [])] +
        [('NFP',  d) for d in calendar.get('nfp',  [])]
    )

    for event_type, ds in event_list:
        try:
            event_date = date.fromisoformat(ds)
        except ValueError:
            continue
        delta = (event_date - check_date).days
        if 0 <= delta <= lookahead_days:
            label = 'today' if delta == 0 else f'tomorrow ({ds})'
            return True, f"{event_type} {label}"

    return False, ''
