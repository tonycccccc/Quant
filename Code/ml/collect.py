"""
Step 1 — Fetch historical 30-min OHLCV bars from Alpaca (IEX feed).

Saves a MultiIndex parquet (symbol, timestamp) to Models/raw_bars.parquet.
Skips re-fetching if the file already exists and --force is not set.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, WATCHLIST, MODELS_DIR,
    ML_RAW_BARS_PATH, ML_HISTORY_MONTHS, ML_EXTRA_TRAINING_SYMBOLS,
)

# QQQ is used for relative-strength features; SPY for macro regime context
_INDEX_TICKERS = ['QQQ', 'SPY']


def fetch_bars(months_back: int = ML_HISTORY_MONTHS, force: bool = False):
    """
    Fetch 30-min bars for all watchlist stocks + ML_EXTRA_TRAINING_SYMBOLS + QQQ/SPY.

    The extra training symbols are used only for ML training to broaden regime
    coverage. They are not traded live.

    Returns a pd.DataFrame with MultiIndex (symbol, timestamp) and
    columns [open, high, low, close, volume].

    If raw_bars.parquet already exists and force=False, loads from disk.
    """
    import pandas as pd
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    if not force and ML_RAW_BARS_PATH.exists():
        print(f'[collect] Loading cached bars from {ML_RAW_BARS_PATH}')
        return pd.read_parquet(ML_RAW_BARS_PATH)

    end   = datetime.now()
    start = end - timedelta(days=months_back * 31)   # 31 days/month buffer

    # Deduplicate in case of overlap between watchlist and extra symbols
    all_trade_symbols = list(WATCHLIST.keys())
    extra = [s for s in ML_EXTRA_TRAINING_SYMBOLS if s not in all_trade_symbols]
    symbols = all_trade_symbols + extra + _INDEX_TICKERS
    print(f'[collect] Fetching {months_back}mo of 30-min bars for {len(symbols)} symbols '
          f'({len(all_trade_symbols)} watchlist + {len(extra)} training-only + {len(_INDEX_TICKERS)} indexes) ...')

    client  = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request)
    df   = bars.df   # MultiIndex: (symbol, timestamp), already UTC tz-aware

    # Keep only the OHLCV columns we need
    keep = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df   = df[keep]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ML_RAW_BARS_PATH)
    fetched = df.index.get_level_values(0).nunique()
    print(f'[collect] Saved {len(df):,} bars across {fetched} symbols -> {ML_RAW_BARS_PATH}')
    return df
