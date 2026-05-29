import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'Code')
from config import ML_FEATURES_PATH

df = pd.read_parquet(ML_FEATURES_PATH)
df = df[df['label'].notna()].copy()
df.index = pd.to_datetime(df.index, utc=True)

early  = df[df.index < '2025-01-01']
recent = df[df.index >= '2025-01-01']

features = ['rs_vs_qqq','prior_5d_return','ema20_ema50_ratio',
            'rsi','atr_pct','atr_contraction','vol_ratio',
            'close_vwap_ratio','higher_highs','higher_lows']

print("=== Feature Distribution Shift ===")
print(f"  {'Feature':<24} {'Early mean':>11} {'Recent mean':>12} {'Drift':>8}")
for f in features:
    em = early[f].mean()
    rm = recent[f].mean()
    drift = (rm - em) / (abs(em) + 1e-9) * 100
    print(f"  {f:<24} {em:>11.4f} {rm:>12.4f} {drift:>7.1f}%")

print()
print("=== vol_ratio distribution (IEX volume health check) ===")
print(f"  Mean:   {df.vol_ratio.mean():.3f}")
print(f"  Median: {df.vol_ratio.median():.3f}")
print(f"  p90:    {df.vol_ratio.quantile(0.90):.3f}")
print(f"  p99:    {df.vol_ratio.quantile(0.99):.3f}")

print()
print("=== Pearson corr(feature, label) by era ===")
print(f"  {'Feature':<24} {'All data':>9} {'Early':>9} {'Recent':>9}")
for f in features:
    ca = df[[f,'label']].corr().iloc[0,1]
    ce = early[[f,'label']].corr().iloc[0,1]
    cr = recent[[f,'label']].corr().iloc[0,1]
    print(f"  {f:<24} {ca:>+9.4f} {ce:>+9.4f} {cr:>+9.4f}")

print()
print("=== TP rate by quarter ===")
df['ym'] = df.index.tz_localize(None).to_period('Q')
q = df.groupby('ym')['label'].agg(['mean','count'])
q.columns = ['tp_rate','n_bars']
print(q.to_string())

print()
print("=== Higher-highs / higher-lows trends (momentum regime) ===")
hh_early  = early['higher_highs'].mean()
hh_recent = recent['higher_highs'].mean()
hl_early  = early['higher_lows'].mean()
hl_recent = recent['higher_lows'].mean()
print(f"  higher_highs: early={hh_early:.2%}  recent={hh_recent:.2%}")
print(f"  higher_lows:  early={hl_early:.2%}  recent={hl_recent:.2%}")

print()
print("=== EMA alignment (bullish structure) ===")
em_early  = early['ema20_ema50_ratio'].mean()
em_recent = recent['ema20_ema50_ratio'].mean()
print(f"  ema20_ema50_ratio: early={em_early:+.4f}  recent={em_recent:+.4f}")
print(f"  (positive = EMA20 > EMA50 = uptrend)")
