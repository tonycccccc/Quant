"""SQLite schema definitions and CRUD helpers."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import DB_PATH, WATCHLIST


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_db():
    """Create all tables (idempotent) and seed the watchlist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker       TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                sector       TEXT DEFAULT 'Technology'
            );

            CREATE TABLE IF NOT EXISTS daily_clearance (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT    NOT NULL,
                date          TEXT    NOT NULL,
                clearance     INTEGER DEFAULT 0,
                summary       TEXT,
                raw_research  TEXT,
                regime_bias   TEXT,
                confidence    REAL,
                key_catalyst  TEXT,
                earnings_risk INTEGER DEFAULT 0,
                created_at    TEXT    DEFAULT (datetime('now')),
                UNIQUE(ticker, date)
            );

            CREATE TABLE IF NOT EXISTS active_positions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                alpaca_order_id   TEXT UNIQUE NOT NULL,
                ticker            TEXT NOT NULL,
                shares            REAL NOT NULL,
                entry_price       REAL,
                stop_loss_price   REAL NOT NULL,
                take_profit_price REAL NOT NULL,
                status            TEXT DEFAULT 'pending',
                signal_score      REAL,
                regime_bias       TEXT,
                entry_time        TEXT,
                exit_time         TEXT,
                exit_price        REAL,
                realized_pnl      REAL,
                created_at        TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS trade_signals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT NOT NULL,
                signal_time  TEXT NOT NULL,
                signal_score REAL,
                clearance    INTEGER DEFAULT 0,
                executed     INTEGER DEFAULT 0,
                skip_reason  TEXT,
                close_price  REAL,
                ema20        REAL,
                ema50        REAL,
                vwap         REAL,
                atr          REAL,
                volume       REAL,
                volume_avg   REAL,
                rs_vs_qqq    REAL,
                regime_bias  TEXT
            );
        """)

        for ticker, name in WATCHLIST.items():
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (ticker, company_name) VALUES (?, ?)",
                (ticker, name),
            )
        conn.commit()

    print(f"[DB] Initialized at {DB_PATH}")


# ── Clearance ──────────────────────────────────────────────────────────────

def upsert_clearance(ticker: str, date: str, clearance: int, summary: str,
                     raw_research: str, regime_bias: str, confidence: float,
                     key_catalyst: str, earnings_risk: bool):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO daily_clearance
                (ticker, date, clearance, summary, raw_research, regime_bias,
                 confidence, key_catalyst, earnings_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                clearance     = excluded.clearance,
                summary       = excluded.summary,
                raw_research  = excluded.raw_research,
                regime_bias   = excluded.regime_bias,
                confidence    = excluded.confidence,
                key_catalyst  = excluded.key_catalyst,
                earnings_risk = excluded.earnings_risk,
                created_at    = datetime('now')
        """, (ticker, date, clearance, summary, raw_research,
              regime_bias, confidence, key_catalyst, int(earnings_risk)))
        conn.commit()


def get_today_clearance(ticker: str) -> Optional[dict]:
    today = datetime.now().strftime('%Y-%m-%d')
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM daily_clearance WHERE ticker = ? AND date = ?",
            (ticker, today),
        ).fetchone()
        return dict(row) if row else None


def get_all_today_clearances() -> list:
    today = datetime.now().strftime('%Y-%m-%d')
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_clearance WHERE date = ? ORDER BY ticker",
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Signals ────────────────────────────────────────────────────────────────

def save_signal(ticker: str, signal_score: float, clearance: int,
                executed: bool, skip_reason: Optional[str], indicators: dict):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO trade_signals
                (ticker, signal_time, signal_score, clearance, executed, skip_reason,
                 close_price, ema20, ema50, vwap, atr, volume, volume_avg, rs_vs_qqq, regime_bias)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, signal_score, clearance, int(executed), skip_reason,
            indicators.get('close'), indicators.get('ema20'), indicators.get('ema50'),
            indicators.get('vwap'), indicators.get('atr'), indicators.get('volume'),
            indicators.get('volume_avg'), indicators.get('rs_vs_qqq'),
            indicators.get('regime_bias'),
        ))
        conn.commit()


# ── Positions ──────────────────────────────────────────────────────────────

def save_position(alpaca_order_id: str, ticker: str, shares: float,
                  stop_loss_price: float, take_profit_price: float,
                  signal_score: float, regime_bias: str):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO active_positions
                (alpaca_order_id, ticker, shares, stop_loss_price,
                 take_profit_price, signal_score, regime_bias)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (alpaca_order_id, ticker, shares, stop_loss_price,
              take_profit_price, signal_score, regime_bias))
        conn.commit()


def update_position_fill(alpaca_order_id: str, entry_price: float):
    with get_connection() as conn:
        conn.execute("""
            UPDATE active_positions
               SET status = 'open', entry_price = ?, entry_time = datetime('now')
             WHERE alpaca_order_id = ?
        """, (entry_price, alpaca_order_id))
        conn.commit()


def close_position(alpaca_order_id: str, exit_price: float, status: str):
    with get_connection() as conn:
        pos = conn.execute(
            "SELECT * FROM active_positions WHERE alpaca_order_id = ?",
            (alpaca_order_id,),
        ).fetchone()
        if pos and pos['entry_price']:
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            conn.execute("""
                UPDATE active_positions
                   SET status = ?, exit_price = ?,
                       exit_time = datetime('now'), realized_pnl = ?
                 WHERE alpaca_order_id = ?
            """, (status, exit_price, pnl, alpaca_order_id))
            conn.commit()


def get_open_positions() -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM active_positions WHERE status IN ('pending', 'open')",
        ).fetchall()
        return [dict(r) for r in rows]


def get_closed_positions(limit: int = 20) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM active_positions
               WHERE status NOT IN ('pending', 'open')
               ORDER BY exit_time DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def has_open_position_for_ticker(ticker: str) -> bool:
    """Return True if there is already a pending or open position for this ticker."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM active_positions WHERE ticker = ? AND status IN ('pending', 'open') LIMIT 1",
            (ticker,),
        ).fetchone()
        return row is not None


def get_today_realized_pnl() -> float:
    """Return total realized P&L from trades closed today."""
    today = datetime.now().strftime('%Y-%m-%d')
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(realized_pnl), 0.0) AS total
               FROM active_positions
               WHERE date(exit_time) = ? AND realized_pnl IS NOT NULL""",
            (today,),
        ).fetchone()
        return float(row['total']) if row else 0.0
