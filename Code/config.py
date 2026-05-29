import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / '.env')

# ── API Keys ───────────────────────────────────────────────────────────────
ALPACA_API_KEY     = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY  = os.getenv('ALPACA_SECRET_KEY', '')
DISCORD_WEBHOOK_URL = os.getenv('Discord_Webhook', '')
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY', '')

# ── Watchlist ──────────────────────────────────────────────────────────────
WATCHLIST = {
    'NVDA':  'Nvidia',
    'AMD':   'AMD',
    'PLTR':  'Palantir',
    'MSFT':  'Microsoft',
    'META':  'Meta Platforms',
    'AMZN':  'Amazon',
    'GOOGL': 'Alphabet (Google)',
    'TSLA':  'Tesla',
    'AAPL':  'Apple',
    'ORCL':  'Oracle',
    'NFLX':  'Netflix',
}

# Additional symbols used for ML training only (not traded live).
# More diverse market regimes = better model generalization.
ML_EXTRA_TRAINING_SYMBOLS = [
    'AVGO',  # Broadcom — semis, similar momentum profile to NVDA/AMD
    'QCOM',  # Qualcomm — semis / mobile chips
    'MU',    # Micron — memory semis, high beta
    'CRM',   # Salesforce — enterprise SaaS momentum
    'NOW',   # ServiceNow — enterprise SaaS, strong trend behavior
    'CRWD',  # CrowdStrike — cybersecurity growth stock
    'UBER',  # Uber — consumer tech, liquid and volatile
    'SHOP',  # Shopify — e-commerce growth, similar breakout patterns
]

MARKET_INDEXES = ['SPY', 'QQQ']

# ── Risk & Portfolio ───────────────────────────────────────────────────────
RISK_PER_TRADE          = 0.015   # 1.5% of equity max loss per position
MAX_POSITIONS           = 5
MAX_STOCK_CONCENTRATION = 0.20    # 20% of portfolio per stock
HARD_STOP_LOSS_PCT      = 0.035   # 3.5% absolute hard cap
TAKE_PROFIT_PCT         = 0.07    # +7%  — second (full) TP leg

# ── Two-Leg Take Profit ────────────────────────────────────────────────────
TP1_PCT            = 0.03   # first exit at +3% (30% of shares)
TP1_SHARE_FRACTION = 0.30

# ── Structural Stop ────────────────────────────────────────────────────────
STRUCTURAL_STOP_MIN_DISTANCE = 0.01  # minimum 1% below entry; VWAP used if deeper

# ── Volatility / Macro Gates ───────────────────────────────────────────────
VIX_HARD_BLOCK       = 30    # no new trades when spot VIX >= this
DAILY_LOSS_LIMIT_PCT = 0.03  # circuit breaker: halt trading if total P&L < -3% equity

# ── Signal Scoring ─────────────────────────────────────────────────────────
# Max base score is now 135 (added RSI=20 + MACD=15 on top of original 100).
# With bullish regime multiplier (×1.10) ceiling is ~148.
SIGNAL_BUY_THRESHOLD   = 100
SIGNAL_WATCH_THRESHOLD = 80

# ── Technical Indicator Periods ────────────────────────────────────────────
EMA_SHORT_PERIOD    = 20
EMA_LONG_PERIOD     = 50
ATR_PERIOD          = 14
VOLUME_MA_PERIOD    = 20
RESISTANCE_LOOKBACK = 20
VWAP_HOLD_BARS      = 2

RSI_PERIOD          = 14
RSI_OVERBOUGHT      = 80    # hard block — no entry when RSI >= this value
MACD_FAST_PERIOD    = 12
MACD_SLOW_PERIOD    = 26
MACD_SIGNAL_PERIOD  = 9

# ── Volume Thresholds ──────────────────────────────────────────────────────
VOLUME_SPIKE_MULTIPLIER    = 1.5   # min spike indicating compression exit
BREAKOUT_VOLUME_MULTIPLIER = 1.8   # confirms breakout
ATR_CONTRACTION_THRESHOLD  = 0.85  # ATR < 85% of reference = compression

# ── Relative Strength ──────────────────────────────────────────────────────
RS_PERIOD_DAYS = 5                 # 5-day return vs QQQ

# ── LLM (OpenRouter free tier only) ───────────────────────────────────────
LLM_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'

# ── File Paths ─────────────────────────────────────────────────────────────
DB_PATH               = BASE_DIR / 'trading.db'
LOGS_DIR              = BASE_DIR / 'Logs'
RESEARCH_DIR          = BASE_DIR / 'Metholody' / 'Research'
PORTFOLIO_STATUS_PATH = BASE_DIR / 'Metholody' / 'portfolio_status.md'
STRATEGY_PATH         = BASE_DIR / 'Metholody' / 'Rules' / 'Strategy.md'
TRADING_STATS_CSV     = LOGS_DIR / 'trading_stats.csv'

# ── ML Model ───────────────────────────────────────────────────────────────
MODELS_DIR            = BASE_DIR / 'Models'
ML_RAW_BARS_PATH      = MODELS_DIR / 'raw_bars.parquet'
ML_FEATURES_PATH      = MODELS_DIR / 'features.parquet'
ML_MODEL_PATH         = MODELS_DIR / 'quant_model.pkl'

ML_HISTORY_MONTHS     = 24          # months of 30-min bar history to fetch
ML_LABEL_TP_PCT       = 0.05        # forward label: +5% = TP hit (used in training only)
ML_LABEL_TIMEOUT_DAYS = 5           # trading days to wait for TP resolution
ML_LABEL_TIMEOUT_BARS = ML_LABEL_TIMEOUT_DAYS * 13  # 65 30-min bars
ML_CONFIDENCE_THRESHOLD = 0.55      # P(TP_hit) gate: must exceed this to trade
ML_MIN_PRECISION      = 0.30        # walk-forward CV precision floor; raise RuntimeError if below
ML_SIGNAL_SCORE_THRESHOLD = 90      # only train on bars that score >= this (aligns training with inference)
ML_ENABLED            = True        # set False to bypass ML gate entirely
