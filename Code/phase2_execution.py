"""
Phase 2: Alpaca Execution & Server-Side Bracket Configuration

PortfolioManager validates pre-conditions, sizes the position using the
1.5%-risk model, submits TWO BRACKET orders per trade (TP1 at +3% for 30%
of shares, TP2 at +7% for 70% of shares), persists to SQLite + CSV, and
notifies Discord.

Structural stop: VWAP is used as the stop level when it is at least 1%
below entry; otherwise a 1% minimum-distance stop is applied. The -3.5%
hard cap is always enforced as a floor.

Guards: per-ticker duplicate check, daily circuit breaker (-3% equity).
"""
import csv
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    RISK_PER_TRADE, MAX_POSITIONS, MAX_STOCK_CONCENTRATION,
    HARD_STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TP1_PCT, TP1_SHARE_FRACTION,
    STRUCTURAL_STOP_MIN_DISTANCE,
    DAILY_LOSS_LIMIT_PCT,
    ML_ENABLED, ML_CONFIDENCE_THRESHOLD,
    TRADING_STATS_CSV, LOGS_DIR,
)
import database as db
import discord_client as discord

# CSV column order — entry fields populated at buy, exit fields filled by Phase 3
STATS_COLUMNS = [
    'trade_id', 'ticker', 'entry_time', 'exit_time',
    'entry_price', 'exit_price', 'shares',
    'stop_loss_price', 'take_profit_price',
    'realized_pnl', 'pnl_pct', 'exit_reason',
    'signal_score', 'ema20', 'ema50', 'vwap_at_entry',
    'atr_at_entry', 'volume_at_entry', 'volume_avg',
    'rs_vs_qqq', 'regime_bias',
    'breakout_vol_ratio', 'atr_contraction_ratio',
    'rsi_at_entry', 'macd_histogram_at_entry',
]


# ── CSV helpers ────────────────────────────────────────────────────────────

def _ensure_csv_header():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not TRADING_STATS_CSV.exists():
        with open(TRADING_STATS_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=STATS_COLUMNS).writeheader()


def _log_entry_to_csv(row: dict):
    _ensure_csv_header()
    with open(TRADING_STATS_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=STATS_COLUMNS, extrasaction='ignore').writerow(row)


def log_exit_to_csv(trade_id: str, exit_price: float,
                    exit_time: str, exit_reason: str, realized_pnl: float):
    """Update exit fields for a completed trade (called from Phase 3)."""
    if not TRADING_STATS_CSV.exists():
        return
    rows = []
    updated = False
    with open(TRADING_STATS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('trade_id') == trade_id:
                ep = float(row['entry_price']) if row.get('entry_price') else 0
                row['exit_price']   = exit_price
                row['exit_time']    = exit_time
                row['exit_reason']  = exit_reason
                row['realized_pnl'] = realized_pnl
                row['pnl_pct']      = round((exit_price - ep) / ep, 4) if ep else 0
                updated = True
            rows.append(row)
    if updated:
        with open(TRADING_STATS_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=STATS_COLUMNS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)


# ── Portfolio manager ──────────────────────────────────────────────────────

class PortfolioManager:
    def __init__(self, dry_run: bool = False):
        """
        dry_run=True  → logs + notifies but never submits a real Alpaca order.
        dry_run=False → submits live paper orders to Alpaca.
        """
        from alpaca.trading.client import TradingClient
        self.dry_run = dry_run
        self.client  = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

    # ── Read-only helpers ──────────────────────────────────────────────────

    def get_equity(self) -> float:
        return float(self.client.get_account().equity)

    def get_open_count(self) -> int:
        """Number of unique ticker positions currently held (Alpaca aggregates by ticker)."""
        return len(self.client.get_all_positions())

    def get_today_total_pnl(self) -> float:
        """Realized P&L today plus current unrealized P&L across all open positions."""
        realized = db.get_today_realized_pnl()
        try:
            positions  = self.client.get_all_positions()
            unrealized = sum(float(p.unrealized_pl) for p in positions)
        except Exception:
            unrealized = 0.0
        return realized + unrealized

    # ── Stop calculation ───────────────────────────────────────────────────

    def _compute_stop(self, entry_price: float, vwap: float) -> float:
        """
        Structural stop: use VWAP when it is >= STRUCTURAL_STOP_MIN_DISTANCE (1%)
        below entry. Otherwise fall back to the minimum 1% distance.
        Always clamps to the HARD_STOP_LOSS_PCT (-3.5%) floor.
        """
        hard_floor = entry_price * (1 - HARD_STOP_LOSS_PCT)
        min_stop   = entry_price * (1 - STRUCTURAL_STOP_MIN_DISTANCE)

        if vwap > 0 and vwap <= min_stop:
            # VWAP is at least 1% below entry — use it (respect hard floor)
            return round(max(vwap, hard_floor), 2)
        # VWAP too close or unavailable — use minimum structural distance
        return round(max(min_stop, hard_floor), 2)

    # ── Main execution entry point ─────────────────────────────────────────

    def execute_signal(self, ticker: str, indicators: dict,
                       signal_score: float, regime_bias: str,
                       bar_df=None, timestamp=None) -> bool:
        """
        Full execution pipeline for one confirmed buy signal.

        Guards  : duplicate ticker, max-position cap, daily circuit breaker, ML confidence
        Sizing  : 1.5%-risk formula using actual stop distance; 20% concentration cap
        Orders  : two bracket orders — TP1 (+3%, 30% shares) + TP2 (+7%, 70% shares)

        bar_df / timestamp are passed from Phase 1 for ML feature computation.
        Returns True if at least one order was submitted (or dry-run succeeded).
        """
        from alpaca.trading.requests import (
            MarketOrderRequest, TakeProfitRequest, StopLossRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        print(f'\n[Phase 2] Processing signal: {ticker}')

        equity = self.get_equity()

        # ── Guard: duplicate ticker ────────────────────────────────────────
        if db.has_open_position_for_ticker(ticker):
            print(f'  [SKIP] Already holding an open/pending position for {ticker}')
            return False

        # ── Guard: position cap ────────────────────────────────────────────
        open_count = self.get_open_count()
        if open_count >= MAX_POSITIONS:
            print(f'  [SKIP] Max positions reached ({open_count}/{MAX_POSITIONS})')
            return False

        # ── Guard: daily circuit breaker ───────────────────────────────────
        today_pnl = self.get_today_total_pnl()
        loss_limit = -(equity * DAILY_LOSS_LIMIT_PCT)
        if today_pnl < loss_limit:
            print(f'  [CIRCUIT BREAKER] Today P&L ${today_pnl:+.2f} < '
                  f'-3% equity (${loss_limit:.2f}) — no new trades today.')
            return False

        # ── Guard: ML confidence gate ──────────────────────────────────────
        if ML_ENABLED:
            try:
                from ml.predict import predict_success_prob, model_info
                from ml.features import indicators_to_feature_row
                if model_info() is None:
                    print('  [ML] No trained model found — ML gate skipped')
                else:
                    rs          = indicators.get('rs_vs_qqq', 0.0)
                    ts          = timestamp or datetime.now()
                    feature_row = indicators_to_feature_row(indicators, bar_df, ts,
                                                            rs_vs_qqq=rs)
                    prob = predict_success_prob(feature_row)
                    if prob < ML_CONFIDENCE_THRESHOLD:
                        print(f'  [ML] P(TP)={prob:.1%} < '
                              f'{ML_CONFIDENCE_THRESHOLD:.0%} threshold — skipping')
                        return False
                    print(f'  [ML] P(TP)={prob:.1%} — gate passed')
            except Exception as e:
                print(f'  [ML] Gate error: {e} — proceeding without ML filter')

        # ── Position sizing ────────────────────────────────────────────────
        entry_price    = indicators['close']
        vwap           = indicators.get('vwap', 0.0)
        stop_loss_price = self._compute_stop(entry_price, vwap)
        stop_distance  = entry_price - stop_loss_price

        if stop_distance <= 0:
            print(f'  [SKIP] Invalid stop distance ({stop_distance:.4f}) — skipping')
            return False

        total_shares = int((equity * RISK_PER_TRADE) / stop_distance)

        # Concentration cap
        max_dollars = equity * MAX_STOCK_CONCENTRATION
        if total_shares * entry_price > max_dollars:
            total_shares = int(max_dollars / entry_price)
            print(f'  [SIZE] Capped at 20% concentration → {total_shares} shares')

        if total_shares <= 0:
            print(f'  [SKIP] Position size computed as 0 shares — skipping')
            return False

        # Split into two legs
        tp1_shares = int(total_shares * TP1_SHARE_FRACTION)
        tp2_shares = total_shares - tp1_shares
        if tp1_shares <= 0:
            tp1_shares = 0
            tp2_shares = total_shares

        tp1_price = round(entry_price * (1 + TP1_PCT),        2)
        tp2_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

        print(f'  Equity=${equity:,.2f} | Shares={total_shares} '
              f'(TP1:{tp1_shares} @ +{TP1_PCT:.0%} / TP2:{tp2_shares} @ +{TAKE_PROFIT_PCT:.0%}) | '
              f'Entry≈${entry_price:.2f} | SL=${stop_loss_price:.2f} '
              f'({(1 - stop_loss_price/entry_price):.1%} below)')

        # ── Shared entry-row template ──────────────────────────────────────
        base_row = {
            'trade_id':             None,
            'ticker':               ticker,
            'entry_time':           datetime.now().isoformat(),
            'exit_time':            '',
            'entry_price':          entry_price,
            'exit_price':           '',
            'shares':               None,            # per-leg
            'stop_loss_price':      stop_loss_price,
            'take_profit_price':    None,            # per-leg
            'realized_pnl':         '',
            'pnl_pct':              '',
            'exit_reason':          '',
            'signal_score':         signal_score,
            'ema20':                indicators.get('ema20'),
            'ema50':                indicators.get('ema50'),
            'vwap_at_entry':        vwap,
            'atr_at_entry':         indicators.get('atr'),
            'volume_at_entry':      indicators.get('volume'),
            'volume_avg':           indicators.get('volume_avg'),
            'rs_vs_qqq':            indicators.get('rs_vs_qqq'),
            'regime_bias':          regime_bias,
            'breakout_vol_ratio':   round(
                                        (indicators.get('volume') or 0) /
                                        (indicators.get('volume_avg') or 1), 3
                                    ),
            'atr_contraction_ratio':   indicators.get('atr_contraction_ratio'),
            'rsi_at_entry':            indicators.get('rsi'),
            'macd_histogram_at_entry': indicators.get('macd_histogram'),
        }

        # ── Dry-run path ───────────────────────────────────────────────────
        if self.dry_run:
            suffix = datetime.now().strftime('%H%M%S')
            if tp1_shares > 0:
                oid1 = f'DRYRUN-{ticker}-TP1-{suffix}'
                print(f'  [DRY RUN] Leg 1: {tp1_shares}× {ticker}  TP=${tp1_price}  id={oid1}')
                db.save_position(oid1, ticker, tp1_shares, stop_loss_price,
                                 tp1_price, signal_score, regime_bias)
                _log_entry_to_csv({**base_row, 'trade_id': oid1,
                                   'shares': tp1_shares, 'take_profit_price': tp1_price})

            oid2 = f'DRYRUN-{ticker}-TP2-{suffix}'
            print(f'  [DRY RUN] Leg 2: {tp2_shares}× {ticker}  TP=${tp2_price}  id={oid2}')
            db.save_position(oid2, ticker, tp2_shares, stop_loss_price,
                             tp2_price, signal_score, regime_bias)
            _log_entry_to_csv({**base_row, 'trade_id': oid2,
                               'shares': tp2_shares, 'take_profit_price': tp2_price})

            discord.send_trade_entry(ticker, total_shares, entry_price,
                                     stop_loss_price, tp2_price, signal_score, oid2)
            return True

        # ── Live order submission ──────────────────────────────────────────
        submitted_ids = []
        try:
            if tp1_shares > 0:
                req1 = MarketOrderRequest(
                    symbol=ticker, qty=tp1_shares,
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp1_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_price),
                )
                order1 = self.client.submit_order(req1)
                oid1   = str(order1.id)
                submitted_ids.append(oid1)
                db.save_position(oid1, ticker, tp1_shares, stop_loss_price,
                                 tp1_price, signal_score, regime_bias)
                _log_entry_to_csv({**base_row, 'trade_id': oid1,
                                   'shares': tp1_shares, 'take_profit_price': tp1_price})
                print(f'  TP1 bracket submitted: {oid1}')

            req2 = MarketOrderRequest(
                symbol=ticker, qty=tp2_shares,
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp2_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price),
            )
            order2 = self.client.submit_order(req2)
            oid2   = str(order2.id)
            submitted_ids.append(oid2)
            db.save_position(oid2, ticker, tp2_shares, stop_loss_price,
                             tp2_price, signal_score, regime_bias)
            _log_entry_to_csv({**base_row, 'trade_id': oid2,
                               'shares': tp2_shares, 'take_profit_price': tp2_price})
            print(f'  TP2 bracket submitted: {oid2}  ✅')

        except Exception as e:
            print(f'  [ERROR] Order failed: {e}')
            discord.send_error(f'Order submission failed for {ticker}: {e}')
            return bool(submitted_ids)  # True if at least TP1 was submitted

        discord.send_trade_entry(ticker, total_shares, entry_price,
                                 stop_loss_price, tp2_price,
                                 signal_score, ', '.join(submitted_ids))
        return True
