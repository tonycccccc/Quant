"""
Phase 3: Trailing Maintenance & Portfolio Tracking

- Connects to Alpaca TradingStream WebSocket (paper=True)
- On every 'fill' event:
    - Parent buy fill  → marks position 'open', updates entry_price
    - Bracket leg fill → closes the position, updates CSV, notifies Discord
      TP/SL classification uses recorded take_profit_price / stop_loss_price
      from the DB (not P&L sign) to avoid misclassification.
- After each close, logs a one-line observation to trade_observations.md
  (LLM reflection and Strategy.md auto-write are disabled).
- Maintains Metholody/portfolio_status.md after every state change
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    PORTFOLIO_STATUS_PATH,
)
import database as db
import discord_client as discord


# ── Portfolio status writer ────────────────────────────────────────────────

def update_portfolio_status():
    """Regenerate Metholody/portfolio_status.md from live Alpaca + DB state."""
    try:
        from alpaca.trading.client import TradingClient
        client    = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        account   = client.get_account()
        equity    = float(account.equity)
        cash      = float(account.cash)
        positions = client.get_all_positions()
    except Exception as e:
        print(f'[Phase 3] Alpaca account fetch failed: {e}')
        equity, cash, positions = 0.0, 0.0, []

    open_pnl     = sum(float(p.unrealized_pl) for p in positions)
    open_pnl_pct = (open_pnl / equity * 100) if equity else 0.0
    now_str      = datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')

    lines = [
        '# Portfolio Status',
        f'*Last Updated: {now_str}*',
        '',
        '## Account Overview',
        '| Metric | Value |',
        '|--------|-------|',
        f'| Equity | ${equity:,.2f} |',
        f'| Open P&L | ${open_pnl:+,.2f} ({open_pnl_pct:+.2f}%) |',
        f'| Cash | ${cash:,.2f} |',
        f'| Open Positions | {len(positions)}/5 |',
        '',
    ]

    if positions:
        lines += [
            '## Open Positions',
            '| Ticker | Shares | Avg Entry | Current | Unrealized P&L | % |',
            '|--------|--------|-----------|---------|----------------|---|',
        ]
        for p in positions:
            upnl     = float(p.unrealized_pl)
            upnl_pct = float(p.unrealized_plpc) * 100
            icon     = '🟢' if upnl >= 0 else '🔴'
            lines.append(
                f'| {p.symbol} | {p.qty} | ${float(p.avg_entry_price):.2f} '
                f'| ${float(p.current_price):.2f} '
                f'| {icon} ${upnl:+,.2f} | {upnl_pct:+.1f}% |'
            )
        lines.append('')

    closed = db.get_closed_positions(limit=10)
    if closed:
        lines += [
            '## Recent Closed Trades',
            '| Ticker | Entry | Exit | P&L | Status |',
            '|--------|-------|------|-----|--------|',
        ]
        for t in closed:
            pnl    = t.get('realized_pnl') or 0
            icon   = '✅' if pnl >= 0 else '❌'
            ep     = t.get('entry_price') or 0
            xp     = t.get('exit_price')  or 0
            lines.append(
                f"| {t['ticker']} | ${ep:.2f} | ${xp:.2f} "
                f'| {icon} ${pnl:+,.2f} | {t["status"]} |'
            )
        lines.append('')

    all_closed = db.get_closed_positions(limit=200)
    if all_closed:
        wins     = [t for t in all_closed if (t.get('realized_pnl') or 0) > 0]
        losses   = [t for t in all_closed if (t.get('realized_pnl') or 0) <= 0]
        wr       = len(wins) / len(all_closed) * 100
        avg_win  = sum(t['realized_pnl'] for t in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(t['realized_pnl'] for t in losses) / len(losses) if losses else 0
        total    = sum(t.get('realized_pnl') or 0 for t in all_closed)
        lines += [
            '## Performance Summary',
            f'- **Win Rate**: {wr:.1f}%  ({len(wins)}W / {len(losses)}L)',
            f'- **Avg Win**: ${avg_win:+,.2f}',
            f'- **Avg Loss**: ${avg_loss:+,.2f}',
            f'- **Total Realized P&L**: ${total:+,.2f}',
            '',
        ]

    PORTFOLIO_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_STATUS_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print('[Phase 3] portfolio_status.md updated')


# ── Trade observation log (replaces LLM reflection) ───────────────────────

def _log_trade_observation(trade: dict):
    """
    Append a one-line trade result to trade_observations.md.
    No LLM calls; Strategy.md is never auto-modified.
    """
    obs_path = PORTFOLIO_STATUS_PATH.parent / 'trade_observations.md'
    pnl    = trade.get('realized_pnl', 0) or 0
    result = trade.get('status', 'closed').replace('closed_', '').upper()
    ts     = datetime.now().strftime('%Y-%m-%d %H:%M')
    line   = (
        f'| {ts} | {trade["ticker"]} | {result} | ${pnl:+.2f} | '
        f'Score={trade.get("signal_score", "N/A")} | '
        f'Regime={trade.get("regime_bias", "N/A")} |\n'
    )
    try:
        if not obs_path.exists():
            obs_path.write_text(
                '# Trade Observations\n'
                '| Time | Ticker | Exit | P&L | Score | Regime |\n'
                '|------|--------|------|-----|-------|--------|\n',
                encoding='utf-8',
            )
        with open(obs_path, 'a', encoding='utf-8') as f:
            f.write(line)
    except Exception as e:
        print(f'[Phase 3] Could not write trade_observations.md: {e}')


# ── WebSocket stream ───────────────────────────────────────────────────────

async def run_tracking():
    """
    Start the Alpaca TradingStream WebSocket.
    Blocks until Ctrl+C or stream disconnect.
    """
    from alpaca.trading.stream import TradingStream

    print(f"\n{'='*60}")
    print('  PHASE 3 — Trade Tracking Stream')
    print(f"{'='*60}")
    print('[Phase 3] Connecting to Alpaca TradingStream (paper=True)...')

    stream = TradingStream(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    @stream.on_trade_update
    async def handle(data):
        event    = data.event
        order    = data.order
        order_id = str(order.id)

        print(f'[Stream] {event.upper()} — {order.symbol}  order_id={order_id[:16]}')

        if event == 'fill':
            fill_price = float(order.filled_avg_price or 0)

            if order.side == 'buy':
                # Primary entry has filled
                db.update_position_fill(order_id, fill_price)
                print(f'  Entry filled: {order.symbol} @ ${fill_price:.2f}')
                update_portfolio_status()

            elif order.side == 'sell':
                # A bracket leg (SL or TP) has triggered
                open_pos = db.get_open_positions()
                pos = next(
                    (p for p in open_pos if p['ticker'] == order.symbol
                     and p['status'] == 'open'),
                    None,
                )
                if pos:
                    entry_p  = pos.get('entry_price') or fill_price
                    pnl      = (fill_price - entry_p) * pos['shares']
                    tp_price = pos.get('take_profit_price') or float('inf')
                    sl_price = pos.get('stop_loss_price')   or 0.0

                    # Classify exit using recorded bracket prices (1% tolerance)
                    if fill_price >= tp_price * 0.999:
                        status = 'closed_tp'
                        reason = 'Take Profit Hit'
                    elif fill_price <= sl_price * 1.001:
                        status = 'closed_sl'
                        reason = 'Stop Loss Hit'
                    else:
                        # Ambiguous (manual close / partial) — fall back to P&L sign
                        status = 'closed_tp' if pnl >= 0 else 'closed_sl'
                        reason = 'Take Profit Hit' if pnl >= 0 else 'Stop Loss Hit'

                    db.close_position(pos['alpaca_order_id'], fill_price, status)
                    print(f'  Closed {order.symbol} via {reason} '
                          f'@ ${fill_price:.2f}  P&L=${pnl:+.2f}')

                    discord.send_trade_exit(
                        ticker=order.symbol, shares=pos['shares'],
                        entry_price=entry_p, exit_price=fill_price,
                        pnl=pnl, reason=reason,
                    )

                    from phase2_execution import log_exit_to_csv
                    log_exit_to_csv(
                        trade_id=pos['alpaca_order_id'],
                        exit_price=fill_price,
                        exit_time=datetime.now().isoformat(),
                        exit_reason=reason,
                        realized_pnl=pnl,
                    )

                    # Append trade observation (no LLM; Strategy.md is not modified)
                    closed = db.get_closed_positions(limit=1)
                    if closed:
                        _log_trade_observation(closed[0])

                    update_portfolio_status()

        elif event in ('canceled', 'expired'):
            # Clean up any pending position that never got an entry fill
            for p in db.get_open_positions():
                if p['alpaca_order_id'] == order_id and p['status'] == 'pending':
                    with db.get_connection() as conn:
                        conn.execute(
                            "UPDATE active_positions SET status='canceled' "
                            "WHERE alpaca_order_id=?", (order_id,)
                        )
                        conn.commit()
                    print(f'  Pending position for {order.symbol} marked canceled')

    # Snapshot portfolio on startup
    update_portfolio_status()
    print('[Phase 3] Stream running. Press Ctrl+C to stop.\n')
    stream.run()
