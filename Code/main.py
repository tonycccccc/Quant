"""
AI Quant Trader — RATMB momentum model + fundamentals advisor
=============================================================
Daily
  python main.py daily-update                  Fetch new bars, rebuild features, refresh fundamentals, retrain if stale
  python main.py scan-momentum [--only-buys]   Score every ticker; BUY/WATCH/SKIP + fair-value context
  python main.py advise --ticker NVDA          Full report for one ticker: signal, entry plan, fundamentals
  python main.py scan-fundamentals             Fundamentals leaderboard

Model
  python main.py build-dataset [--months 12] [--force]   Fetch bars + compute features + labels
  python main.py train-model                              Walk-forward CV + train final model
  python main.py backtest | oos-backtest | walk-forward-oos
  python main.py tune-bracket | ablate-features
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


def cmd_oos_backtest(args):
    """Train on first N months, backtest on the remainder. True OOS read."""
    from ml.backtest import run_oos_backtest
    run_oos_backtest(train_months=args.train_months, starting_equity=args.equity)


def cmd_walk_forward_oos(args):
    """Multi-fold walk-forward OOS — averages over N test windows for robustness."""
    from ml.backtest import run_walk_forward_oos
    run_walk_forward_oos(n_folds=args.folds, test_months=args.test_months,
                          starting_equity=args.equity)


def cmd_daily_update(args):
    """
    Daily update pipeline (run once per trading day, e.g., 08:00 ET pre-market):
      1. Fetch incremental bars (only new since last cache) — ~30s
      2. Rebuild features.parquet with fresh data                — ~5 min
      3. Refresh macro features (24h cache auto-handled)         — <10s
      4. Retrain model IF older than --retrain-days               — ~5 min

    Designed for cron / Windows Task Scheduler / AWS EventBridge.
    Idempotent — safe to run multiple times per day.
    """
    from datetime import datetime
    import time
    t0 = time.time()

    log_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n{"="*66}')
    print(f'  DAILY UPDATE — {log_stamp}')
    print(f'{"="*66}')

    # Step 1: incremental bar fetch
    print('\n[1/4] Fetching incremental bars...')
    from ml.collect import fetch_incremental
    fetch_incremental(lookback_days=args.lookback)

    # Step 2: rebuild features
    print('\n[2/4] Rebuilding features (30-min + 4H + daily + macro + ranks + labels)...')
    from ml.features import build_all_features
    build_all_features(save=True)

    # Step 2b: refresh fundamentals (yfinance, 24h TTL, non-critical)
    print('\n[2b/4] Refreshing fundamentals (yfinance)...')
    try:
        from ml.fundamentals import refresh_all
        refresh_all(force=False)
    except Exception as e:
        print(f'  [fundamentals] refresh failed: {e} — advisor will use stale/missing data')

    # Step 3: check model age
    print('\n[3/4] Checking model age...')
    from config import ML_MODEL_PATH
    should_retrain = False
    if not ML_MODEL_PATH.exists():
        print('  No model found — will retrain')
        should_retrain = True
    else:
        model_age_days = (datetime.now().timestamp() - ML_MODEL_PATH.stat().st_mtime) / 86400
        print(f'  Model age: {model_age_days:.1f} days (retrain threshold: {args.retrain_days})')
        if args.force_retrain or model_age_days >= args.retrain_days:
            should_retrain = True

    # Step 4: optional retrain
    if should_retrain:
        print(f'\n[4/4] Retraining model...')
        from ml.train import run_training_pipeline
        from config import ML_MIN_PRECISION
        run_training_pipeline(min_precision=ML_MIN_PRECISION)
    else:
        print(f'\n[4/4] Skipping retrain (model still fresh; use --force-retrain to override)')

    elapsed = time.time() - t0
    print(f'\n[daily-update] Complete in {elapsed:.1f}s at {datetime.now().strftime("%H:%M:%S")}')


def cmd_tune_bracket(args):
    """Sweep TP/SL/timeout combos and exit-reason histogram."""
    from ml.tune_bracket import run_tp_sl_grid, analyze_exit_reasons
    print('\n[tune-bracket] Step 1: exit-reason histogram from existing walk-forward trades')
    try:
        analyze_exit_reasons()
    except FileNotFoundError as e:
        print(f'  Skipped: {e}')
    print('\n[tune-bracket] Step 2: TP/SL/timeout grid sweep')
    run_tp_sl_grid(quick=args.quick)


def cmd_ablate_features(_args):
    """Feature ablation: find the minimum feature set that captures most performance."""
    from ml.feature_ablation import run_ablation
    run_ablation()


def cmd_advise(args):
    """Advisor mode: run the full pipeline for one ticker and report recommendation."""
    from advisor import advise
    advise(ticker=args.ticker, equity=args.equity,
            refresh=args.refresh, verbose=args.verbose)


def cmd_scan_fundamentals(args):
    """Print a fundamentals leaderboard sorted by quality score."""
    from ml.fundamentals import leaderboard
    leaderboard(filter_mode=args.filter)


def cmd_scan_momentum(args):
    """Run the momentum model across the whole universe. Show BUY candidates."""
    from scan_momentum import scan_momentum
    scan_momentum(min_score=args.min_score, only_buys=args.only_buys)


# ── Argument parser ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description='AI Quant Trader — RATMB Strategy Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest='command')

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

    poos = sub.add_parser('oos-backtest',
                          help='True OOS: train on first N months, backtest on the rest')
    poos.add_argument('--train-months', type=int, default=18, dest='train_months',
                      help='Months of history to train on; rest becomes the OOS test (default: 18)')
    poos.add_argument('--equity', type=float, default=10_000.0,
                      help='Starting equity for the simulation (default: $10,000)')

    pwf = sub.add_parser('walk-forward-oos',
                         help='Multi-fold OOS backtest — most reliable read on real-world performance')
    pwf.add_argument('--folds', type=int, default=4,
                     help='Number of rolling OOS folds (default: 4)')
    pwf.add_argument('--test-months', type=int, default=3, dest='test_months',
                     help='Months per OOS test window (default: 3)')
    pwf.add_argument('--equity', type=float, default=10_000.0,
                     help='Starting equity per fold (default: $10,000)')

    pdu = sub.add_parser('daily-update',
                          help='Incremental bar fetch + feature rebuild + optional retrain (schedule this)')
    pdu.add_argument('--lookback', type=int, default=2,
                     help='Days of overlap on incremental fetch (default: 2)')
    pdu.add_argument('--retrain-days', type=int, default=7, dest='retrain_days',
                     help='Retrain if model is this many days old or older (default: 7)')
    pdu.add_argument('--force-retrain', action='store_true',
                     help='Retrain regardless of model age')

    ptun = sub.add_parser('tune-bracket',
                          help='Grid-search TP/SL/timeout combos for best risk-adjusted return')
    ptun.add_argument('--quick', action='store_true',
                      help='Use a small 3x3x2 grid (faster) instead of 5x4x3')

    sub.add_parser('ablate-features',
                    help='Feature ablation: cumulative subsets to find minimum viable feature set')

    psf = sub.add_parser('scan-fundamentals',
                          help='Leaderboard of all tickers sorted by fundamental quality score')
    psf.add_argument('--filter', choices=['all', 'strong'], default='all',
                     help='"strong": quality score >= 70')

    psm = sub.add_parser('scan-momentum',
                          help='Run the momentum model across the whole universe; find BUY setups')
    psm.add_argument('--min-score', type=float, default=0,
                     help='Filter to tickers with rule score >= this (default: 0 = all)')
    psm.add_argument('--only-buys', action='store_true',
                     help='Show only BUY verdicts')

    pad = sub.add_parser('advise',
                          help='Advisor mode: full recommendation for one ticker (BUY/WATCH/SKIP + prices)')
    pad.add_argument('--ticker', required=True,
                     help='Ticker to analyze (e.g., AVGO)')
    pad.add_argument('--equity', type=float, default=10_000.0,
                     help='Assumed account equity for position sizing (default: $10,000)')
    pad.add_argument('--refresh', action='store_true',
                     help='Force live Alpaca fetch instead of using cached bars')
    pad.add_argument('--verbose', action='store_true',
                     help='Print all raw feature values for debugging')

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        'build-dataset': cmd_build_dataset,
        'train-model':   cmd_train_model,
        'backtest':         cmd_backtest,
        'oos-backtest':     cmd_oos_backtest,
        'walk-forward-oos': cmd_walk_forward_oos,
        'tune-bracket':     cmd_tune_bracket,
        'ablate-features':   cmd_ablate_features,
        'advise':            cmd_advise,
        'scan-fundamentals': cmd_scan_fundamentals,
        'scan-momentum':     cmd_scan_momentum,
        'daily-update':      cmd_daily_update,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
