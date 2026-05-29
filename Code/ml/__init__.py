"""
ML sub-package for the RATMB quant model.

Pipeline:
  collect  → raw_bars.parquet
  features → features.parquet  (indicators + forward labels)
  train    → quant_model.pkl   (LightGBM / RandomForest fallback)
  predict  → P(TP_hit)         (used by Phase 1 at inference time)
"""
