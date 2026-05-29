"""Discord webhook notifications for trades, battle plans, and alerts."""
import httpx
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import DISCORD_WEBHOOK_URL


def _post(payload: dict):
    if not DISCORD_WEBHOOK_URL:
        print("[Discord] Webhook URL not configured — skipping")
        return
    try:
        r = httpx.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[Discord] Failed: {e}")


def send_battle_plan(clearances: list):
    """Morning 'Daily Battle Plan' grid embed."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M EST')
    rows = []
    for c in clearances:
        icon = "🟢" if c['clearance'] == 1 else "🔴"
        bias = (c.get('regime_bias') or 'N/A').capitalize()
        conf = c.get('confidence') or 0.0
        summary = (c.get('summary') or '')[:80]
        rows.append(f"{icon} **{c['ticker']}** — {bias} ({conf:.0%}) | {summary}")

    cleared = sum(1 for c in clearances if c['clearance'] == 1)
    payload = {
        "embeds": [{
            "title": f"📋 Daily Battle Plan — {now}",
            "description": "\n".join(rows) or "No clearance data.",
            "color": 0x00FF88,
            "footer": {"text": f"{cleared}/{len(clearances)} stocks cleared for trading"},
        }]
    }
    _post(payload)
    print(f"[Discord] Battle plan sent ({cleared}/{len(clearances)} cleared)")


def send_trade_entry(ticker: str, shares: float, entry_price: float,
                     stop_loss: float, take_profit: float,
                     signal_score: float, order_id: str):
    """Bracket order submitted notification."""
    risk_pct   = (entry_price - stop_loss) / entry_price * 100
    reward_pct = (take_profit - entry_price) / entry_price * 100
    payload = {
        "embeds": [{
            "title": f"🚀 TRADE ENTERED — {ticker}",
            "color": 0x00BFFF,
            "fields": [
                {"name": "Shares",       "value": str(shares),                                       "inline": True},
                {"name": "Entry",        "value": f"${entry_price:.2f}",                             "inline": True},
                {"name": "Signal Score", "value": f"{signal_score:.1f}/100",                         "inline": True},
                {"name": "Stop Loss",    "value": f"${stop_loss:.2f}  (-{risk_pct:.1f}%)",           "inline": True},
                {"name": "Take Profit",  "value": f"${take_profit:.2f}  (+{reward_pct:.1f}%)",       "inline": True},
                {"name": "Order ID",     "value": order_id[:20] + "...",                             "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    _post(payload)


def send_trade_exit(ticker: str, shares: float, entry_price: float,
                    exit_price: float, pnl: float, reason: str):
    """Position closed notification."""
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    color   = 0x00FF88 if pnl >= 0 else 0xFF4444
    icon    = "✅" if pnl >= 0 else "❌"
    payload = {
        "embeds": [{
            "title": f"{icon} TRADE CLOSED — {ticker}",
            "color": color,
            "fields": [
                {"name": "Reason",  "value": reason,                                    "inline": False},
                {"name": "Entry",   "value": f"${entry_price:.2f}",                     "inline": True},
                {"name": "Exit",    "value": f"${exit_price:.2f}",                      "inline": True},
                {"name": "P&L",     "value": f"${pnl:+,.2f}  ({pnl_pct:+.1f}%)",       "inline": True},
                {"name": "Shares",  "value": str(shares),                               "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    _post(payload)


def send_error(message: str):
    """System error / warning alert."""
    payload = {
        "embeds": [{
            "title": "⚠️ System Alert",
            "description": message,
            "color": 0xFF8800,
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    _post(payload)
