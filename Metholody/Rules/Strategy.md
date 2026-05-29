“Regime-Adaptive Tech Momentum Breakout System (RATMB v1)”
Goal: Not be an aggressiver trader. Buy smartly, beat SPY & QQQ.

1. Strategy Overview

This system trades high-liquidity US tech equities using a combination of:

Trend continuation (primary edge)
Volatility expansion (timing filter)
VWAP structural reclaim (execution trigger)
Relative strength vs QQQ (selection filter)
Market regime gating (risk filter)
Holding Period:
2 days to 3 weeks (swing / short momentum)
Universe:
Fixed 11-stock high liquidity tech basket (see Section 1a)
1a. Trading Universe

| Ticker | Company |
|---|---|
| NVDA | Nvidia |
| AMD | AMD |
| PLTR | Palantir |
| MSFT | Microsoft |
| META | Meta Platforms |
| AMZN | Amazon |
| GOOGL | Alphabet (Google) |
| TSLA | Tesla |
| AAPL | Apple |
| ORCL | Oracle |
| NFLX | Netflix |

All 11 are US large-cap tech/growth equities with high intraday liquidity and strong momentum characteristics.

2. Core Principle

“We do not predict price. We detect structural imbalance and ride forced continuation.”

The strategy assumes:

price moves in bursts, not linear trends
liquidity clusters create predictable breakouts
most alpha comes from timing entry in expansion phases
3. Market Regime Filter (GLOBAL GATE)

No trade is allowed unless regime is favorable.

Compute:
SPY trend (20EMA > 50EMA = bullish bias)
QQQ relative strength
VIX level + slope
market breadth (advancers/decliners)
Regime States:
🟢 Risk-On (Trade allowed)
SPY trending up
VIX stable or declining
tech leading
🟡 Neutral (reduced size)
mixed signals
choppy breadth
🔴 Risk-Off (NO TRADE)
VIX spike
breakdown in SPY
macro shock
4. Stock Selection Filter (RELATIVE STRENGTH CORE)

Each stock is scored vs QQQ:

Relative Strength Score:
RS = (Stock 5D return - QQQ 5D return)

Only trade if:

RS > 0 (must outperform index)
5. Setup Condition (PRE-ENTRY STRUCTURE)

A trade candidate must satisfy ALL:

5.1 Trend Structure
price above 20 EMA
20 EMA above 50 EMA
higher highs / higher lows structure
5.2 Volatility Compression → Expansion

Detect:

low ATR compression phase
followed by volume expansion

Condition:

ATR contraction ratio < threshold
AND volume spike > 1.5x avg
5.3 VWAP Structural Reclaim

Entry only valid if:

price reclaims VWAP
holds above VWAP for N bars
5.4 Volume Confirmation
breakout volume ≥ 1.8x 20-bar average
6. Entry Trigger (EXECUTION SIGNAL)

Entry occurs only when ALL are true:

Breakout Condition:
Close > previous resistance high
AND volume expansion confirmed
AND VWAP support holds
AND regime != risk_off
7. Entry Timing Logic

We do NOT buy breakouts blindly.

We prefer:

Preferred Entry Types:
A. VWAP Reclaim Pullback Entry
breakout happens
price retests VWAP
holds → entry
B. Micro Break + Hold Entry
breakout candle closes strong
next candle confirms continuation
8. Position Sizing (RISK-FIRST MODEL)

Risk per trade:

1.5% of total equity max loss per position
Position Size Formula:
shares = (equity * 0.015) / stop_loss_distance
9. Stop Loss Logic (STRUCTURAL, NOT FIXED %)

Stop is placed at:

below VWAP reclaim zone OR
below breakout base OR
-3.5% max hard cap
Rule:

whichever is tighter but not too tight to cause noise stop-outs

10. Take Profit Logic (SCALING EXIT)
Tiered Exit:
TP1: +3%
take 30% position off
TP2: +7%
take 50% off
Runner:
trail with 10 EMA or VWAP
11. Trailing Stop Logic

Trailing activates only after:

trade is +3% in profit

Trail method:

max(high since entry) - ATR(14) * 1.5
12. Exit Conditions

Exit immediately if:

VWAP lost on high volume
market regime flips risk-off
sector breakdown occurs
stop loss hit (hard exit via Alpaca)
13. Regime Multiplier (LLM ROLE)

LLM is ONLY allowed to output:

{
  "regime_bias": "bullish | neutral | bearish",
  "confidence": 0.0-1.0
}
Impact:
final_signal_score *= (0.9 to 1.1)

NOT a buy/sell decision.

14. No Trade Conditions (IMPORTANT FILTER)

Do NOT trade if:

earnings within 3 trading days
abnormal macro event pending (CPI/FOMC)
VIX > extreme threshold
low liquidity anomaly detected
15. Time Filters
No trades first 10 min of market open
No trades last 20 min of close
Avoid lunch chop window (optional)
16. Portfolio Constraints
max 5 positions
max 40% exposure per sector
no correlated overexposure cluster

Example forbidden cluster:

NVDA + AMD + AVGO + TSM
17. Signal Scoring Model

Final score:

Score =
  trend_score        (max 25 — EMA alignment, price above EMA20/EMA50)
+ breakout_strength  (max 20 — close vs resistance)
+ volume_quality     (max 20 — relative volume vs 20-bar MA)
+ VWAP_support       (max 20 — price above VWAP, VWAP hold bars)
+ relative_strength  (max 15 — stock 5d return vs QQQ)
+ rsi_score          (max 20 — RSI zone; hard block at RSI >= 80)
+ macd_score         (max 15 — MACD histogram and line)
+ regime_multiplier  (x1.10 both EMAs aligned, x0.90 neither aligned)

Max base score: 135. With bullish regime multiplier: ~148.

Threshold:

>= 100 → BUY signal (passes to ML gate)
80-99  → WATCH
< 80   → ignore

17a. ML Confidence Gate (runs AFTER score >= 100)

The ML model provides a secondary filter calibrated on 5-day outcome probability:

Gate:
  P(TP hit) >= ML_CONFIDENCE_THRESHOLD (0.55)
  Feature input: 31 features across three timeframes + cross-sectional ranks
  Model: Calibrated LightGBM (isotonic), trained on 41 stocks x 24 months

Lopez de Prado methodology (added 2026-05-29):
  - Triple-barrier labels with ATR-scaled TP/SL (adapt to each stock's volatility)
  - Sample-uniqueness weights for overlapping forward-window labels
  - Isotonic probability calibration (CalibratedClassifierCV)
  - Cross-sectional rank features (rs_rank_5d, rsi_rank, momentum_rank_20d, vol_ratio_rank)
  - Meta-labeling primary_score feature (lets ML see rule-based system's verdict)
  - Signal-quality filter (only train on bars scoring >= 90) for distribution alignment

Why multi-timeframe + cross-sectional:
  Feature importance analysis shows that daily-timeframe features dominate
  prediction of 3-5 day outcomes. Cross-sectional ranks capture relative
  strength dynamics that absolute indicator values miss.

Score adjustment:
  ml_multiplier = 0.8 + 0.4 * P(TP_hit)
  P=0.55 -> x1.02 (slight boost)   P=0.70 -> x1.08   P=0.35 -> x0.94

Fallback: if quant_model.pkl is absent, ML gate is skipped (rule-based only).

17b. Alert Logging (production training-data capture, added 2026-05-29)

Every Phase 1 polling cycle, each watchlist stock that scores >= SIGNAL_WATCH_THRESHOLD (80)
is written to Logs/alerts.parquet with:

  - Full feature snapshot (all 31 ML features) at the moment of the alert
  - Rule-based score breakdown (base, regime multiplier, final)
  - ML probability (or NaN if model absent)
  - Clearance flag and would-have-traded flag
  - Entry price (close at alert time) and current VIX

Outcome columns (tp_hit, sl_hit, max_favorable, max_adverse, exit_reason) are
backfilled by a periodic job once the 5-day forward window resolves.

Purpose: builds an ongoing dataset that mirrors LIVE inference distribution
exactly — same bars, same indicator computations, same regime context. Monthly
retrain merges these alerts with historical Alpaca backfill to grow the
training set with real-world data.

18. Model Retraining Schedule

Retrain monthly using the most recent 24 months of data + accumulated alerts log. Monitor live P(TP_hit) vs actual TP hit rate — if they diverge by > 10pp for 30+ consecutive trading days, retrain immediately.

Training-only symbols (30 total — see Section 1a / quantModel.md Section 2c) are used to broaden regime coverage during training and are NOT traded live. They include semi equipment (AMAT, LRCX, ASML), cybersecurity (CRWD, PANW), and non-tech diversifiers (JPM, V, LLY) to expose the model to wider behavior than mega-cap tech alone.

20. Strategy Philosophy
Edge comes from:
not missing expansions
avoiding chop
entering AFTER confirmation
respecting regime
controlling downside first
21. What This Strategy is NOT
not prediction-based
not LLM-driven
not news-driven scalping
not high-frequency trading
not “AI thinks stock goes up”
22. Expected Behavior Across Market Regimes
Bull market:
high win rate
momentum continuation works well
Choppy market:
fewer trades
stricter filters reduce noise
Bear market:
system mostly inactive (by design)
Final Statement

“This system does not try to be right. It tries to lose small and capture structural momentum when it appears.”