# Quant ML Model — Design & Planning

**Status:** Planning  
**Purpose:** Add a trained ML confidence score to Phase 1 signal scoring without replacing the rule-based system.

---

## 1. What Problem Are We Actually Solving?

Our current `score_signal()` function assigns points to each indicator component using **manually tuned weights** (e.g., "RSI 55-74 = 20 pts"). These weights are educated guesses — they haven't been validated against actual outcomes on our specific watchlist.

The ML model solves one narrow problem:

> **Given the full indicator state at bar T, what is the empirical probability that a bracket order entered here (SL = -3.5%, TP = +7%) will hit TP before SL?**

We call this `P(success | features)`. The model replaces manual weight guessing with a probability learned from historical outcomes on the actual stocks we trade.

**What ML does NOT do:**
- It does not replace Phase 0 clearance (earnings, macro vetoes stay as hard rules)
- It does not pick which stock to buy (rule-based filtering still applies)
- It does not set position sizing (that stays in Phase 2)
- It does not predict price direction — only outcome probability for our specific TP/SL parameters

---

## 2. Data Strategy

### 2a. Bar granularity — **Multi-Timeframe (30-min + 4H proxy + Daily)**

**Primary bar: 30-minute**, matching live trading exactly.  
**Overlaid features: 4-hour proxy and daily**, computed from the same 30-min data.

Rationale:
- Core indicators (EMA, RSI, MACD, VWAP) are all computed at 30-min resolution — no training/inference mismatch.
- 30-min bars alone are too noisy to reliably predict a 5-day trend outcome; adding higher-timeframe context dramatically improves signal quality.
- 4-hour features (8-bar rolling window) capture the intraday trend structure without introducing a new data source.
- Daily features (business-day resample with `shift(1)` lookahead guard) capture multi-day momentum and ATR regime.

Feature importance validation: after training with all three timeframes, **all top-5 features were daily** (`d_vol_ratio`, `d_return_20d`, `d_atr_pct`, `d_rsi`, `d_close_ema20_ratio`), confirming the multi-timeframe hypothesis. 30-min features contribute from position 6 onward.

**Lookahead guard for daily features:** each intraday bar uses the *previous* day's daily data via `shift(1)` before reindexing — the current day's close is never visible at inference time.

### 2b. History window

**Minimum: 12 months. Target: 24 months.**

With 24 months across 19 stocks:
- ~129,800 labeled rows total before quality filtering
- After signal-score filter (score ≥ 90): ~24,500 rows — training distribution matches inference distribution
- Covers at least 2 full bull/bear/sideways cycles for these tech names
- Sufficient for LightGBM with proper regularization

### 2c. Training universe (expanded 2026-05-29)

**41 stocks total: 11 trading stocks + 30 training-only stocks (via `ML_EXTRA_TRAINING_SYMBOLS`).**

| Role | Stocks | Count |
|---|---|---|
| Trading + training | NVDA, AMD, PLTR, MSFT, META, AMZN, GOOGL, TSLA, AAPL, ORCL, NFLX | 11 |
| Training only — semis | AVGO, QCOM, MU, TSM, ASML, AMAT, LRCX, MRVL | 8 |
| Training only — cloud/SaaS/AI infra | CRM, NOW, CRWD, SNOW, DDOG, MDB, NET, PANW | 8 |
| Training only — consumer tech | UBER, SHOP, ABNB, SPOT, ROKU | 5 |
| Training only — fintech | COIN, SQ, HOOD, PYPL | 4 |
| Training only — sector diversifiers | JPM, V, MA, LLY, COST | 5 |
| Index context (always fetched) | QQQ, SPY | 2 |

**Why the expansion (2026-05-29):**
- Earlier 19-stock universe was mostly mega-cap tech — the model couldn't learn cross-sector regime variation.
- Adding semi equipment (AMAT/LRCX/ASML), cybersecurity (CRWD/PANW), and non-tech sectors (JPM/V/LLY) exposes the model to wider behavior:
  - Different volatility profiles (LLY's GLP-1 momentum vs JPM's rate-driven moves)
  - Different breakout patterns (semis cycle differently from SaaS)
  - Different liquidity/volume signatures
- Training-only symbols are excluded from inference by not being in `WATCHLIST`. They contribute to the model's understanding of "what a successful breakout looks like" without touching live trading risk.

### 2d. Data source

**Alpaca historical bars via IEX feed** — free, no additional API keys needed, same source as Phase 1.

```python
StockBarsRequest(
    symbol_or_symbols=all_watchlist + ML_EXTRA_TRAINING_SYMBOLS + ['QQQ', 'SPY'],
    timeframe=TimeFrame.Minute30,
    start=datetime.now() - timedelta(days=730),  # 24 months
    feed=DataFeed.IEX,
)
```

Expected data volume: ~6,500 bars/stock/year × 21 symbols × 2 years = ~273,000 bars. Fits comfortably in memory.

---

## 3. Feature Engineering

Every feature must be available at inference time in Phase 1 (no lookahead). All features are price-scale invariant (ratios or normalized values) so the model generalizes across stocks at different price levels.

### 3a. Complete FEATURE_COLS (26 features)

**30-minute timeframe (17 features) — intraday structure**

| # | Feature | Description |
|---|---|---|
| 1 | `close_ema20_ratio` | `(close/EMA20) − 1` — price distance from EMA20 |
| 2 | `ema20_ema50_ratio` | `(EMA20/EMA50) − 1` — EMA alignment / trend direction |
| 3 | `close_vwap_ratio` | `(close−VWAP)/VWAP` — price above/below intraday VWAP |
| 4 | `close_resist_ratio` | `(close−resistance)/resistance` — breakout distance |
| 5 | `atr_pct` | `ATR/close` — normalized volatility |
| 6 | `atr_contraction` | `ATR / ATR[−10 bars]` — compression ratio (<1 = squeezing) |
| 7 | `rsi` | RSI-14 (0–100) |
| 8 | `macd_hist_pct` | `MACD histogram / close` — momentum acceleration |
| 9 | `macd_line_pct` | `MACD line / close` — trend bias |
| 10 | `vwap_hold` | 1 if last 2 bars both closed above VWAP |
| 11 | `prior_1d_return` | 1-trading-day return (13 bars back) |
| 12 | `prior_5d_return` | 5-trading-day return (65 bars back) |
| 13 | `rs_vs_qqq` | Stock 5d return − QQQ 5d return |
| 14 | `hour` | Decimal hour of day (9.5–16.0 ET) |
| 15 | `day_of_week` | 0 = Mon … 4 = Fri |
| 16 | `spy_ema_aligned` | 1 if SPY EMA20 > EMA50 (macro trend) |
| 17 | `qqq_ema_aligned` | 1 if QQQ EMA20 > EMA50 (tech sector trend) |

**4-hour proxy (3 features) — intraday swing context**

Computed as 8-bar rolling windows on 30-min data. No separate data fetch required.

| # | Feature | Description |
|---|---|---|
| 18 | `h4_close_ema_ratio` | `(close / EWM8) − 1` — price vs 4H EMA |
| 19 | `h4_rsi` | RSI equivalent over 8-bar EWM window |
| 20 | `h4_return` | `pct_change(8)` — 4-hour return |

**Daily timeframe (6 features) — multi-day trend context**

Resampled from 30-min to business-day resolution with `shift(1)` lookahead guard — each intraday bar uses the *previous* day's data only.

| # | Feature | Description |
|---|---|---|
| 21 | `d_close_ema20_ratio` | Daily `(close/EMA20) − 1` — daily trend position |
| 22 | `d_ema_aligned` | 1 if daily EMA20 > EMA50 — daily trend direction |
| 23 | `d_rsi` | Daily RSI-14 |
| 24 | `d_atr_pct` | Daily `ATR/close` — daily volatility regime |
| 25 | `d_vol_ratio` | Daily `volume / vol_MA20` — institutional activity |
| 26 | `d_return_20d` | 20-trading-day return — medium-term momentum |

**Cross-sectional ranks (5 features) — added 2026-05-29**

Each timestamp's universe is ranked together; rank is in [0,1] where 1 = strongest in the universe at that bar. Captures relative-strength dynamics: a stock's RSI=65 means a very different thing when most peers are at 70 vs when most peers are at 50.

| # | Feature | What is ranked |
|---|---|---|
| 27 | `rs_rank_5d` | `prior_5d_return` across all symbols at bar T |
| 28 | `rsi_rank` | `rsi` across all symbols at bar T |
| 29 | `momentum_rank_20d` | `d_return_20d` across all symbols at bar T |
| 30 | `vol_ratio_rank` | `vol_ratio` across all symbols at bar T |

**Meta-labeling primary-signal feature (added 2026-05-29)**

| # | Feature | Description |
|---|---|---|
| 31 | `primary_score` | Rule-based score (0-148) of the same bar — lets the ML model "see what the primary system already concluded" |

This is meta-labeling per López de Prado: the secondary (ML) model is conditioned on the primary (rule-based) model's output and learns "given that the primary says X, should I actually take this trade?"

**Removed features (vs prior design):** `vol_ratio` (redundant with `d_vol_ratio` and `vol_ratio_rank`), `higher_highs`, `higher_lows` (noisy, low importance).

### 3b. Features explicitly excluded

- Raw price levels (`close`, `high`, `low` absolute values) — not scale invariant
- Sequential bar data beyond indicator window — we're not building an LSTM
- Future bar data — obvious lookahead bias

---

## 4. Label Generation

### 4a. Triple-Barrier Method with ATR-Scaled Barriers (Lopez de Prado)

**Default since 2026-05-29.** Each bar's TP and SL barriers scale with that stock's CURRENT daily volatility regime instead of being fixed percentages.

```
daily_atr_pct = d_atr_pct at bar T   (already in features)

tp_pct[T] = clip(2.0 * daily_atr_pct, 3.0%, 10.0%)   # ML_TB_TP_ATR_MULT=2.0
sl_pct[T] = clip(1.0 * daily_atr_pct, 1.5%, 5.0%)    # ML_TB_SL_ATR_MULT=1.0

TP_price = close[T] * (1 + tp_pct[T])
SL_price = close[T] * (1 - sl_pct[T])

label = 1  if TP_price reached before SL_price within 65 bars (5 trading days)
label = 0  if SL_price reached first, or neither reached within 65 bars
```

**Why ATR-scaled barriers:**
- A fixed +5% TP is easy on high-vol stocks (NVDA at 4% daily ATR) but unhittable on low-vol stocks (V at 1% daily ATR). Labels are inconsistent across the universe.
- ATR scaling produces COMPARABLE labels across regimes — each bar's outcome is measured in units of "how much movement is normal here right now."
- Clipping prevents pathological barriers in extreme volatility (e.g., earnings-day spikes) and in dead-flat regimes.

**Resolution metadata returned per bar:**
- `t1`     — timestamp the label resolved (first barrier touched, or timeout)
- `ret`    — forward return at resolution (signed)
- `tp_pct` — barrier that was used for this bar
- `sl_pct` — barrier that was used for this bar

The `t1` column powers **sample-uniqueness weighting** during training: when two labels' `[t, t1]` intervals overlap, they share part of the same future price path and are correlated. Each sample's weight is the inverse of its average concurrency, so overlapping redundant labels can't dominate training.

Fallback: setting `ML_TRIPLE_BARRIER_ENABLED = False` reverts to fixed +5% / -3.5% barriers.

### 4b. Which bars to label

**Option A — Label all valid bars:** Every bar where indicators can be computed (after warmup). Gives the most data. Model learns when conditions are unfavorable too.

**Option B — Label only signal bars:** Only bars where the rule-based score >= WATCH_THRESHOLD (65). Focuses on situations we'd actually trade.

**Recommendation: Option A.** Label all bars. This gives the model more signal about what NOT to buy. Option B risks the model only seeing "good" setups and not learning to distinguish excellent from merely good.

### 4c. Class balance

Historically, tech breakout setups hit TP ~35-55% of the time in a neutral market. This means ~55-65% of labels will be 0 (loss/timeout). This is a mild class imbalance — handle by:
- Using `class_weight='balanced'` in scikit-learn models
- Or using `scale_pos_weight` in LightGBM
- Do NOT oversample/undersample — it introduces temporal bias in time-series data

---

## 5. Model Selection

### Why NOT a neural network / LLM

- With ~50,000 training rows and ~20 features, deep learning will overfit badly
- Neural nets need millions of examples to generalize on tabular financial data
- Interpretation is impossible (we can't learn why the model prefers certain setups)
- Training time and complexity is not justified by the improvement over tree models on this data size

### Why NOT a linear model (Logistic Regression)

Useful as a baseline and for feature importance analysis. But:
- Cannot capture non-linear interactions (e.g., "RSI > 60 only matters if volume is high too")
- Our scoring formula is already effectively a linear model — LR would just re-derive similar weights

### Recommended model: **LightGBM**

Reasons:
- Consistently best performance on tabular data of this size in academic benchmarks
- Handles class imbalance natively (`scale_pos_weight`)
- Outputs calibrated probabilities with `predict_proba()`
- Fast: trains 50,000-row dataset in seconds
- Interpretable: SHAP values show feature contribution for each prediction
- Handles missing values natively (days where some features can't be computed)
- No feature scaling required

Fallback: **RandomForestClassifier** from scikit-learn if LightGBM is not installed (same performance class, slightly easier to configure).

**Start with:** Logistic Regression to establish a feature importance baseline. Upgrade to LightGBM for the production model.

---

## 6. Walk-Forward Validation (REQUIRED)

Financial data is **not i.i.d.** — you cannot randomly split train/test. A random split leaks future market regimes into training data.

### Walk-forward expanding window

```
Month:  1  2  3  4  5  6  7  8  9  10 11 12 | 13 14 ... 24
        ├───────── train ─────────────────────┤ test │ ...
                   then expand:
        ├─────────── train ────────────────────────────┤ test │ ...
```

Minimum 4 folds with a 3-month test window each.

Evaluation at each fold:
1. Train model on data up to fold boundary
2. Generate predictions on next 3 months
3. Report: Precision, Recall, F1, and **simulated P&L** (did following model signals beat the rule-based baseline?)

Final model is trained on ALL available data after validation confirms it generalizes.

### Minimum performance bar to deploy

The model is only useful if it improves on the rule-based baseline:
- **Precision > 0.50** (model says buy → should hit TP more than 50% of the time)
- **Simulated Sharpe ratio > baseline** (rule-based system running on the same test period)
- If the model does NOT beat these bars after walk-forward validation, **do not deploy it** — continue using the rule-based system only

---

## 7. Integration with Phase 1

### Design principle: ML as a gate, not a replacement

The ML model does not replace the rule-based score. It acts as an additional filter:

```
Stage 1: Rule-based score (current system)
  → Must exceed SIGNAL_WATCH_THRESHOLD (80)
  → Pass to Stage 2

Stage 2: ML confidence gate (new)
  → model.predict_proba(features)[1]  →  P(TP_hit)
  → Must exceed ML_CONFIDENCE_THRESHOLD (e.g., 0.55)
  → Pass to Stage 3: execute_signal()

Final signal score sent to Phase 2:
  ml_adjusted_score = base_score * regime_multiplier * (0.8 + 0.4 * P(TP_hit))
  # P=0.50 → multiplier 1.0 (neutral), P=0.70 → multiplier 1.08, P=0.30 → multiplier 0.92
```

This means:
- Signals that the rule-based system likes but the ML model is uncertain about get a slight penalty
- Signals that both agree on are amplified slightly
- Neither can override the other completely
- The system degrades gracefully if the model file is missing (falls back to rule-based only)

### New config constants

```python
ML_CONFIDENCE_THRESHOLD = 0.55   # minimum P(TP_hit) to allow execution
ML_MODEL_PATH           = BASE_DIR / 'Models' / 'quant_model.pkl'
ML_ENABLED              = True   # set False to disable without changing code
```

---

## 8. Implementation Plan (Phased)

### Phase A — Data collection & labeling infrastructure
Files: `Code/ml/collect.py`, `Code/ml/labels.py`

- Fetch 24 months of 30-min bars for all watchlist stocks + QQQ/SPY
- Apply `compute_indicators()` to rolling 55-bar windows across the full history
- Add new ML-only features (prior_day_return, hour_of_day, etc.)
- Generate forward labels for each bar
- Save to `Models/training_data.parquet` (parquet is ~5x more compact than CSV)

CLI command: `python Code/main.py build-dataset [--months 24]`

### Phase B — Training pipeline
Files: `Code/ml/train.py`

- Load `training_data.parquet`
- Walk-forward cross-validation (4 folds)
- Report validation metrics at each fold
- If validation passes: train final model on full dataset
- Save model to `Models/quant_model.pkl`

CLI command: `python Code/main.py train-model [--months 24] [--min-precision 0.50]`

### Phase C — Inference integration
Files: `Code/ml/predict.py`, updates to `phase1_polling.py`

- Load `quant_model.pkl` once at Phase 1 startup
- For each signal candidate: compute `predict_proba(features)`
- Apply ML gate and score adjustment
- Log ML confidence to `trading_stats.csv`

### Phase D — Backtesting & iteration
Files: `Code/ml/backtest.py`

- Simulate Phase 1 → Phase 2 signal flow against historical data
- Compare: rule-based only vs rule-based + ML gate
- Report: win rate, average P&L per trade, Sharpe ratio, max drawdown

CLI command: `python Code/main.py backtest [--months 12]`

---

## 9. Decisions Log

| Question | Decision | Rationale |
|---|---|---|
| Label timeout window | **5 trading days (65 bars)** | Aligns with 3-5 day trend target; avoids stale labels |
| Label TP target | **+5%** (`ML_LABEL_TP_PCT`) | More achievable; better label balance than +7% |
| Label SL | **-3.5%** (same as live hard stop) | Matches actual trading outcome |
| Training data filter | **Signal score ≥ 90** (`ML_SIGNAL_SCORE_THRESHOLD`) | Aligns training distribution with inference distribution |
| Bar size for training | **30-min + 4H proxy + Daily** | Feature importance validates: daily dominates top 5 |
| Feature count | **26 features** | 17 intraday + 3 four-hour + 6 daily |
| Training universe | **19 stocks** (11 traded + 8 training-only) | Broader regime coverage; extra symbols not traded live |
| ML_CONFIDENCE_THRESHOLD | **0.55** | Start conservative; tune if live precision diverges |
| Model storage | **Single pkl** (`quant_model.pkl`), overwritten on retrain | Simplicity; model metadata stored in `ModelBundle` |
| Retrain frequency | **Monthly** | Market regimes shift; monthly retrain keeps model fresh |

---

## 10. Risks

**Overfitting on regime.** If the training data is mostly bull market (2023-2024), the model will underperform in corrections. Mitigation: include at least one bear/sideways period in training data; monitor Sharpe ratio monthly.

**Lookahead bias in labels.** Generating labels correctly requires only using price data AFTER bar T to determine outcomes. Double-check label code never accesses training row's own future. Use explicit time-index bounds.

**Small-N on individual stocks.** 8 stocks × 24 months gives decent aggregate volume but per-stock patterns may be drowned out. Consider adding a stock-identifier feature so the model can learn stock-specific patterns.

**Model staleness.** Market regimes shift. A model trained on 2023-2024 tech bull market may degrade significantly in 2026 if conditions change. Retrain monthly using rolling or expanding window. Track live P(TP_hit) vs actual TP hit rate in production.

**Integration failure modes.** If `quant_model.pkl` is missing or corrupt, Phase 1 must fall back to rule-based scoring silently (not crash). Wrap all ML calls in try/except with fallback.

---

## 11. Libraries Required

Add to `requirements.txt`:

```
scikit-learn>=1.4.0
lightgbm>=4.3.0
pyarrow>=15.0.0    # for parquet support via pandas
shap>=0.45.0       # optional: for model explainability
```

---

---

## 12. Implemented Model Structure

> Updated 2026-05-22 — multi-timeframe features, expanded training universe, signal-quality filter.

### Pipeline files

| File | Responsibility |
|---|---|
| `Code/ml/collect.py` | Fetch 30-min OHLCV from Alpaca IEX for 21 symbols → `Models/raw_bars.parquet` |
| `Code/ml/features.py` | 30-min + 4H + daily feature computation + inference adapter |
| `Code/ml/labels.py` | Forward-scan TP/SL label generation (5-day / +5% TP / -3.5% SL) |
| `Code/ml/train.py` | Signal-filtered walk-forward CV + LightGBM training → `Models/quant_model.pkl` |
| `Code/ml/predict.py` | Inference: load model, return P(TP hit) |
| `tests/test_ml_pipeline.py` | 25 unit tests (all passing, no network) |

### Feature columns (26 total — see Section 3 for full table)

See **Section 3a** for the complete feature table grouped by timeframe.

**Feature importance from 2026-05-29 training run (top 10):**

| Rank | Feature | Timeframe | LightGBM gain |
|---|---|---|---|
| 1 | `d_vol_ratio` | Daily | 1,328 |
| 2 | `d_atr_pct` | Daily | 1,286 |
| 3 | `d_return_20d` | Daily | 1,221 |
| 4 | `d_rsi` | Daily | 1,175 |
| 5 | `d_close_ema20_ratio` | Daily | 1,126 |
| 6 | `momentum_rank_20d` | **Cross-sectional** | 912 |
| 7 | `prior_5d_return` | 30-min | 465 |
| 8 | `rs_rank_5d` | **Cross-sectional** | 418 |
| 9 | `rs_vs_qqq` | 30-min | 413 |
| 10 | `day_of_week` | 30-min | 363 |

All top-5 features remain daily. **Cross-sectional ranks now occupy positions #6 and #8** — confirming the relative-strength hypothesis: knowing how a stock ranks against the universe at this moment is more informative than the absolute indicator value alone.

### Label definition (current)

```
timeout_bars = 5 trading days x 13 30-min bars/day = 65 bars

label[t] = 1  if  high[t+1 ... t+65] >= close[t] x 1.05  arrives before
               low[t+1 ... t+65]  <= close[t] x 0.965
label[t] = 0  otherwise (SL first, or neither within 65 bars)
label[t] = NaN for last 65 rows of each stock (incomplete forward window)
```

### Training data filter (signal-quality gate)

Only rows with a rule-based signal score >= `ML_SIGNAL_SCORE_THRESHOLD` (90) are included in training. This ensures the training distribution matches inference — the ML gate is only called on high-scoring setups in production.

```
Total labeled rows:   129,808
After score>=90 filter: 24,456  (18.8%)
TP rate in filtered set: 30.7%
```

### Latest walk-forward CV results (24 months, 41 stocks, score>=90 filter)

> Trained 2026-05-29 with expanded 41-stock universe + cross-sectional ranks + primary_score meta-feature.

| Metric | Value (new) | Prior (19 stocks) |
|---|---|---|
| Production precision @ 0.55 | **0.379** | 0.348 |
| Top-10% precision (gate) | **0.390** | n/a |
| Top-5% precision (high-conf only) | **0.402** | n/a |
| Avg AUC | **0.583** | 0.549 |
| Training samples (post score>=90) | 49,876 | 24,456 |
| TP rate in filtered set | 30.4% | 30.7% |
| n_splits | 4 | 4 |
| test_months | 3 | 3 |
| ML_MIN_PRECISION floor | 0.30 (top-10%) | 0.30 |

**Per-fold detail (production threshold 0.55):**

| Fold | Precision | Recall | AUC | Top-10% prec | p_max |
|---|---|---|---|---|---|
| 1 | 0.385 | 0.225 | 0.617 | 0.357 | 0.860 |
| 2 | 0.361 | 0.299 | 0.597 | 0.374 | 0.890 |
| 3 | 0.289 | 0.358 | 0.561 | 0.323 | 0.944 |
| 4 | 0.483 | 0.381 | 0.557 | 0.508 | 0.921 |

### Methodology ablation results (Lopez de Prado techniques)

Each new technique was tested in isolation. Final decisions captured here.

| Technique | Verdict | Rationale |
|---|---|---|
| **41-stock universe** | ✅ **Kept** | Broader sector coverage, top features include cross-sector ranks |
| **Cross-sectional ranks** | ✅ **Kept** | `momentum_rank_20d` ranks #6 in feature importance |
| **Primary-score meta-feature** | ✅ **Kept** | Lets ML condition on rule-based system's verdict |
| **Triple-barrier (ATR-scaled)** | ❌ **Disabled** | Tight SL (1.5% min) hit by intraday noise → labels became noisier, AUC dropped to 0.49 |
| **Isotonic calibration** | ❌ **Disabled** | Compressed probabilities to 0.20-0.27 range, no prediction exceeded 0.55 threshold |
| **Sample-uniqueness weights** | ❌ **Disabled** | 9.7x weight spread destabilized LightGBM; can re-enable once weights are capped |

The disabled techniques remain available behind config flags (`ML_TRIPLE_BARRIER_ENABLED`, `ML_CALIBRATION_METHOD`, `ML_SAMPLE_WEIGHTS_ENABLED`) for future re-evaluation when more data accumulates.

### Model (serialised as `ModelBundle`)

```python
@dataclass
class ModelBundle:
    model:        LGBMClassifier | RandomForestClassifier
    feature_cols: list[str]   # FEATURE_COLS at time of training
    trained_at:   str         # ISO timestamp
    n_samples:    int
    tp_rate:      float       # fraction of label=1 in training set
    cv_metrics:   list[dict]  # per-fold precision/recall/AUC
```

Primary: **LightGBM** (`LGBMClassifier`, 400 trees, lr=0.04, `num_leaves=31`, `min_child_samples=30`).
Fallback: `sklearn.RandomForestClassifier` (300 trees, `max_depth=10`, `min_samples_leaf=30`).
Class imbalance handled via `scale_pos_weight = (1-TP_rate) / TP_rate`.

### Walk-forward CV schedule (24-month history, 19 stocks)

```
Training data: all bars before fold boundary
Test window:   3 months immediately after boundary

Fold 1: train months 1-9   | test months 10-12
Fold 2: train months 1-12  | test months 13-15
Fold 3: train months 1-15  | test months 16-18
Fold 4: train months 1-18  | test months 19-21
```

Final model trained on **all** labeled rows that pass the signal filter after CV passes `ML_MIN_PRECISION = 0.30`.

### Inference gate

```python
feature_row = indicators_to_feature_row(indicators, bar_df, timestamp, rs_vs_qqq)
p_success   = predict_success_prob(feature_row)   # -> float [0, 1]

if p_success >= ML_CONFIDENCE_THRESHOLD:   # 0.55
    ml_multiplier = 0.8 + 0.4 * p_success  # 0.92 at p=0.30, 1.0 at p=0.50, 1.08 at p=0.70
    final_score   = base_score * regime_mult * ml_multiplier
    # Execute if final_score >= SIGNAL_BUY_THRESHOLD (100)
```

If `Models/quant_model.pkl` is absent, `predict_success_prob()` returns **0.5** (neutral) and Phase 1 falls back to rule-based scoring alone — no crash, no silent failure.

### Alert Logging System (production training-data capture, added 2026-05-29)

Every Phase 1 polling cycle, each watchlist stock that scores >= `SIGNAL_WATCH_THRESHOLD` (80) is appended to `Logs/alerts.parquet` with:

- `alert_id`, `timestamp_utc`, `timestamp_et`, `symbol`
- Score components: `base_score`, `regime_multiplier`, `final_score`
- Regime: `regime_bias`, `regime_confidence`, `vix`
- ML output: `ml_probability`, `ml_threshold_pass`
- Decision flags: `cleared`, `would_have_traded`
- Entry context: `entry_price`
- **Full feature snapshot: all 31 ML features at the moment of the alert**
- Outcome columns (filled by `backfill_outcomes()` after 5-day window): `tp_hit`, `sl_hit`, `max_favorable`, `max_adverse`, `exit_reason`, `resolved_at`

**Why this matters:** the alerts log captures live inference distribution exactly — same bars, same indicator computations, same regime context. Monthly retrain can merge these with historical Alpaca backfill to grow the training set with real-world data. Over time, this is the only way to escape the "train on the past, hope the future repeats" trap.

Available functions in `Code/ml/alert_log.py`:
- `log_alert(...)` — append a single alert row (called from `phase1_polling.py`)
- `backfill_outcomes(bars_lookup)` — walk unresolved alerts and fill `tp_hit`/`sl_hit`/etc.
- `summary()` — quick CLI status dict (total, resolved, would_have_traded, tp_rate)

### CLI commands

```bash
# Build feature dataset (fetch bars + compute indicators + labels)
python Code/main.py build-dataset --months 24

# Train model (walk-forward CV then final fit)
python Code/main.py train-model --min-precision 0.30

# Run all unit tests (no network, no credentials)
python -m pytest tests/test_ml_pipeline.py -v
```

*Last updated: 2026-05-22*
