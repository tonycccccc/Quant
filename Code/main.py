"""
AI Quant Trader — RATMB Strategy Engine
========================================
Usage
-----
  python main.py setup-db                            Init / reset DB and CSV
  python main.py phase0 [--dry-run]                  Pre-market clearance
  python main.py phase1 [--test] [--dry-run]         One polling cycle
  python main.py phase2 [--ticker NVDA] [--live]     Test execution signal
  python main.py phase3                              Start WebSocket tracker
  python main.py status                              Print portfolio_status.md
  python main.py run [--test] [--dry-run]            Full pipeline
  python main.py build-dataset [--months 12] [--force]   Fetch bars + compute features + labels
  python main.py train-model [--min-precision 0.50]       Walk-forward CV + train final model
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 output on Windows so emoji in print() don't crash
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ── Sub-command handlers ───────────────────────────────────────────────────

def cmd_setup_db(_args):
    import database as db
    from phase2_execution import _ensure_csv_header
    db.initialize_db()
    _ensure_csv_header()
    print('[Setup] DB tables created, CSV header written.')


def cmd_phase0(args):
    import database as db
    db.initialize_db()
    from phase0_research import run_clearance_engine
    run_clearance_engine(dry_run=args.dry_run)


def cmd_phase1(args):
    import database as db
    db.initialize_db()
    from phase1_polling   import run_polling_cycle
    from phase2_execution import PortfolioManager

    pm = PortfolioManager(dry_run=args.dry_run)
    run_polling_cycle(
        execution_callback=pm.execute_signal,
        test_mode=args.test,
        force_run=args.force,
    )


def cmd_phase2(args):
    """
    Inject a mock buy signal for --ticker and run it through Phase 2.
    Fetches real current Alpaca data for the ticker so indicators are live.
    Dry-run by default; pass --live to submit an actual paper order.
    """
    import database as db
    import pandas as pd
    import technicals as ta
    from phase1_polling   import fetch_bars
    from phase2_execution import PortfolioManager

    db.initialize_db()

    ticker   = args.ticker.upper()
    dry_run  = not args.live
    print(f'[Phase 2 Test] Injecting signal for {ticker}  '
          f'({"DRY RUN" if dry_run else "LIVE PAPER ORDER"})')

    # Fetch real bars so indicators reflect actual market conditions
    all_bars = fetch_bars([ticker, 'QQQ'], days_back=7)
    df       = all_bars.get(ticker, pd.DataFrame())
    qqq_df   = all_bars.get('QQQ',   pd.DataFrame())

    if len(df) < 55:
        print(f'  [!] Only {len(df)} bars for {ticker} — using placeholder indicators')
        indicators = {
            'close': 500.0, 'high': 502.0, 'low': 498.0, 'volume': 5_000_000,
            'ema20': 490.0, 'ema50': 480.0, 'atr': 8.0,  'vwap': 495.0,
            'volume_avg': 2_500_000, 'atr_contraction_ratio': 0.80,
            'higher_highs': True,   'higher_lows': True,
            'resistance': 498.0,    'vwap_hold': True,
            'rs_vs_qqq': 0.015,     'regime_bias': 'bullish',
            'signal_score': 85.0,
        }
    else:
        indicators = ta.compute_indicators(df)
        if indicators is None:
            print(f'[ERROR] Could not compute indicators for {ticker}')
            return
        rs = ta.compute_rs_vs_qqq(df, qqq_df) if len(qqq_df) >= 10 else 0.01
        indicators.update({
            'rs_vs_qqq':    round(rs, 4),
            'regime_bias':  'bullish',
            'signal_score': 85.0,
        })
        print(f'  Close=${indicators["close"]:.2f}  EMA20=${indicators["ema20"]:.2f}  '
              f'VWAP=${indicators["vwap"]:.2f}  RS={rs:+.2%}')

    pm     = PortfolioManager(dry_run=dry_run)
    result = pm.execute_signal(ticker, indicators,
                               signal_score=85.0, regime_bias='bullish')
    print(f'\n[Phase 2 Test] Result: {"✅ SUCCESS" if result else "❌ SKIPPED/FAILED"}')


def cmd_phase3(_args):
    import asyncio
    import database as db
    db.initialize_db()
    from phase3_tracking import run_tracking
    asyncio.run(run_tracking())


def cmd_status(_args):
    import database as db
    db.initialize_db()
    from phase3_tracking import update_portfolio_status
    from config import PORTFOLIO_STATUS_PATH
    update_portfolio_status()
    if PORTFOLIO_STATUS_PATH.exists():
        print(PORTFOLIO_STATUS_PATH.read_text())


def cmd_build_dataset(args):
    """Fetch historical bars then build feature + label parquet."""
    from ml.collect  import fetch_bars
    from ml.features import build_all_features
    raw_bars = fetch_bars(months_back=args.months, force=args.force)
    build_all_features(raw_bars, save=True)


def cmd_train_model(args):
    """Walk-forward CV then train final model if precision threshold is met."""
    from ml.train import run_training_pipeline
    run_training_pipeline(min_precision=args.min_precision)


def cmd_backtest(args):
    """Simulate rule-only vs rule+ML against SPY/QQQ buy-and-hold."""
    from ml.backtest import run_backtest
    run_backtest(use_ml=not args.no_ml, starting_equity=args.equity)


def cmd_run(args):
    """
    Full pipeline:
      - Phase 0 clearance engine (once)
      - Phase 1 polling loop in a background thread (every 30 min)
      - Phase 3 WebSocket tracker in the main thread
    """
    import asyncio
    import threading
    import time
    import database as db

    db.initialize_db()

    from phase0_research  import run_clearance_engine
    from phase1_polling   import run_polling_cycle
    from phase2_execution import PortfolioManager
    from phase3_tracking  import run_tracking

    run_clearance_engine(dry_run=args.dry_run)

    pm = PortfolioManager(dry_run=args.dry_run)

    def polling_loop():
        while True:
            run_polling_cycle(
                execution_callback=pm.execute_signal,
                test_mode=args.test,
                force_run=args.test,
            )
            print('[Runner] Sleeping 30 minutes until next poll...')
            time.sleep(30 * 60)

    t = threading.Thread(target=polling_loop, daemon=True)
    t.start()

    asyncio.run(run_tracking())


# ── Argument parser ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description='AI Quant Trader — RATMB Strategy Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest='command')

    sub.add_parser('setup-db', help='Init / reset SQLite DB and trading_stats.csv')

    p0 = sub.add_parser('phase0', help='Pre-market clearance engine')
    p0.add_argument('--dry-run', action='store_true',
                    help='Skip LLM/Perplexity calls; use stub data (good for testing DB)')

    p1 = sub.add_parser('phase1', help='One 30-min polling cycle')
    p1.add_argument('--test',    action='store_true',
                    help='Skip market-hours check; use 7 days of historical bars')
    p1.add_argument('--force',   action='store_true',
                    help='Force run regardless of time or market status')
    p1.add_argument('--dry-run', action='store_true',
                    help='Run signals through scoring but do not submit orders')

    p2 = sub.add_parser('phase2', help='Inject a test buy signal through execution')
    p2.add_argument('--ticker', default='NVDA', metavar='TICKER',
                    help='Ticker to inject (default: NVDA)')
    p2.add_argument('--live',   action='store_true',
                    help='Actually submit a paper order (default is dry-run)')

    sub.add_parser('phase3', help='Start WebSocket trade tracker')
    sub.add_parser('status',  help='Print current portfolio_status.md')

    pr = sub.add_parser('run', help='Full pipeline (ph0 + polling loop + ph3 stream)')
    pr.add_argument('--test',    action='store_true',
                    help='Use historical data for polling (no market-hours gate)')
    pr.add_argument('--dry-run', action='store_true',
                    help='Do not submit real orders')

    pds = sub.add_parser('build-dataset',
                         help='Fetch 30-min bars + compute features + attach labels')
    pds.add_argument('--months', type=int, default=12,
                     help='Months of history to fetch (default: 12)')
    pds.add_argument('--force', action='store_true',
                     help='Re-fetch even if raw_bars.parquet already exists')

    ptm = sub.add_parser('train-model',
                         help='Walk-forward CV + train final LightGBM/RF model')
    ptm.add_argument('--min-precision', type=float, default=0.50, dest='min_precision',
                     help='Minimum CV precision to allow saving the model (default: 0.50)')

    pbt = sub.add_parser('backtest',
                         help='Simulate rule-only and rule+ML vs SPY/QQQ buy-and-hold')
    pbt.add_argument('--equity', type=float, default=10_000.0,
                     help='Starting equity for the simulation (default: $10,000)')
    pbt.add_argument('--no-ml', action='store_true',
                     help='Skip the rule+ML variant (rule-only baseline)')

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        'setup-db':      cmd_setup_db,
        'phase0':        cmd_phase0,
        'phase1':        cmd_phase1,
        'phase2':        cmd_phase2,
        'phase3':        cmd_phase3,
        'status':        cmd_status,
        'run':           cmd_run,
        'build-dataset': cmd_build_dataset,
        'train-model':   cmd_train_model,
        'backtest':      cmd_backtest,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
