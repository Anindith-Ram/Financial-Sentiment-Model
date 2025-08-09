# Price Model Pipeline

This project forecasts next‑day stock movement using a 5‑day context window. It provides a clear, reproducible pipeline for data collection, modeling, training, and evaluation with time‑series‑aware cross‑validation.

- Data: `price model/data/`
- Models: `price model/models/`
- Source: `price model/src/`

## 1) Data pipeline

File: `src/data/data_collection.py`

- Universe: S&P 500 from `data/constituents.csv` (Wikipedia fallback).
- Source: Yahoo Finance (auto‑adjusted OHLCV) per ticker.
- Indicators & features (per day): SMA/EMA, RSI, MACD, Stochastic, ATR, Bollinger, gaps, returns, turnover, realized vol(5/10/20), regime flags.
- Sequence: 5‑day rolling window per ticker → flattened `feature_0..N-1`.
- Labels: 3‑class with two modes
  - `rolling_quantiles` (33/66% per‑ticker)
  - `absolute_bands` (recommended): neutral if |ret| ≤ `neutral_band`; up if ret ≥ `up_down_threshold`; down if ≤ −`up_down_threshold`.
- Extra columns: `TargetRet` (continuous next‑day return), `EndDate` (sequence end timestamp), `Ticker`.

Run collection:
```bash
python src/data/data_collection.py --mode full
```
Outputs: `data/professional_dataset_*.csv`, `data/latest_dataset.csv`, and a JSON quality report.

## 2) Model architecture

File: `src/models/timesnet_hybrid.py`

- CNN branch for local pattern extraction.
- PatchTST‑style TimesNet branch:
  - Per‑series normalization across time (per sample)
  - Time patch embedding (e.g., `patch_len=2/3`, `stride=1`) → Transformer blocks
- Gated fusion → MLP trunk → dual heads:
  - Classification head: logits for 3 classes
  - Regression head: next‑day return prediction (optional Huber loss)

Factory:
```python
create_timesnet_hybrid(features_per_day, num_classes=3,
  cnn_channels=512, timesnet_emb=512, timesnet_depth=5,
  seq_len=5, patch_len=2, patch_stride=1, use_series_norm=True)
```

## 3) Training

File: `src/training/research_trainer.py`

- Optimizer: AdamW with per‑component LR ratios
  - TimesNet: 0.05×, CNN: 1.0×, Head: 1.0×, weight_decay=1e‑4
- Scheduler: OneCycleLR (pct_start=0.10, peak ≈ 2× per group, cosine anneal)
- Regularization: MixUp=0.1, CutMix=0.0, label smoothing=0.02 (kept light for fast learning)
- Gradient clipping: 0.05–0.1; EMA evaluation enabled
- Losses: CrossEntropy (primary) + optional Huber on `TargetRet` (λ=0.2)
- Metrics each epoch: Accuracy, Macro‑F1, AUROC(OVR), Precision@Top‑1%; LR (max group)
- Artifacts per run (`run_id`): best checkpoint, training history JSON, curves PNG (stamped with `run_id`).

Run:
```bash
python src/training/research_trainer.py
```

### Purged time‑series CV (opt‑in)
- Chronological K‑fold sorted by `EndDate` (no shuffle), with embargo (e.g., 2%) after each validation window.
- Enable in trainer: `use_cv=True`, `n_splits=5`, `embargo_frac=0.02`.
- Each fold produces its own checkpoint/history/curves (PNG stamped with `run_id`). You can ensemble fold models by averaging probabilities.

## 4) High‑impact knobs

- Labels: `label_mode` (absolute bands), `neutral_band`, `up_down_threshold`
- Model: `patch_len`, `use_series_norm`
- Optimizer/scheduler: base LR (1e‑4 – 3e‑4), TimesNet ratio (0.05–0.1×), pct_start (0.05–0.15), peak multiplier (2–2.5×)
- Regularization: MixUp/CutMix prob, smoothing (0.02–0.05)
- Evaluation: temperature scaling (on), `select_threshold` for abstention

## 5) Production & inference

- Production and progressive trainers are dual‑head compatible (use logits for loss/metrics).
- Inference can load the best single model or ensemble fold models; temperature scaling and thresholds can be applied to probabilities.

## 6) Typical recipes

- Fast learner: CE only, MixUp 0.1, smoothing 0.02, pct_start=0.1, peak 2×, TimesNet 0.1× for first 20 epochs then 0.05×.
- CV selection: 5 folds, embargo 2%, select by Macro‑F1 or Precision@1%.
- Higher ceiling: add cross‑sectional daily z‑scores/sector one‑hot; ensemble folds; add LightGBM stacker on deep logits.

## 7) Paths

- Data: `data/latest_dataset.csv`, `data/professional_dataset_*.csv` + reports
- Models: `models/research/enhanced_cnn_best_<run_id>.pth`, `models/research/training_history.json`, `models/research/training_curves.png`
