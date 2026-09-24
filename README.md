# RATMB — momentum model + fundamentals advisor

Captures momentum breakouts in US tech stocks (rule score + LightGBM gate on
30-min Alpaca bars) and adds a fundamentals layer that estimates fair value
and a margin-of-safety buy price. Advisory only: no orders are placed.

```
# Once per trading day (pre-market): new bars, features, fundamentals, retrain if stale
python Code/main.py daily-update

# Which tickers have a momentum setup right now?
python Code/main.py scan-momentum --only-buys

# Full report for one ticker: signal, entry/TP/SL plan, sizing, fair value
python Code/main.py advise --ticker NVDA --equity 25000

# Fundamentals leaderboard
python Code/main.py scan-fundamentals
```

Model maintenance: `build-dataset`, `train-model`, `backtest`, `oos-backtest`,
`walk-forward-oos`, `tune-bracket`, `ablate-features` (see `python Code/main.py -h`).

Setup: `pip install -r requirements.txt`, then put `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY` in `.env`. Segment data for multi-segment companies
(e.g. MSFT) goes in `Code/ml/segment_disclosures.json`.
