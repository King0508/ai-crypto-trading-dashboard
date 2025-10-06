# Market Microstructure TCN (PyTorch)

Forecast short-horizon mid-price moves from limit order book (LOB) snapshots using a compact Temporal Convolutional Network (TCN) in PyTorch. The project includes:

- A clean time-series pipeline with point-in-time labeling
- Chronological train/val/test splits with walk-forward evaluation
- A small, fast TCN model
- A synthetic LOB generator to train without proprietary data

## Quickstart

```bash
# 1) Create env (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Make synthetic data (~2–3 min, small)
python scripts/simulate_lob.py --n-samples 40000 --seq-len 40 --out data/synth.parquet

# 3) Train
python src/train.py --config configs/default.yaml

# 4) Evaluate (AUC, accuracy, simple PnL backtest)
python src/eval.py --config configs/default.yaml --ckpt artifacts/model.pt
```

## Data format

Expected parquet/csv with columns:

- `timestamp` (int or ISO8601)
- **Features** like `bid_px_1..N`, `ask_px_1..N`, `bid_sz_1..N`, `ask_sz_1..N` (you can add more, e.g., imbalances)
- Label is generated **point-in-time**: `y = sign(mid[t+H] - mid[t]) ∈ {0,1}`

Use the synthetic generator as a template to adapt to your own LOB dumps.

## Result tracking

- Prints metrics to console
- Writes artifacts to `artifacts/` (model.pt, scaler.pkl, metrics.json)

## Notes

- Keep your **train/test split by time** (no shuffling across time).
- Tune `horizon`, `seq_len`, and thresholding in `configs/default.yaml`.
