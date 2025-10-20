# Market Microstructure TCN

Build and test short-horizon crypto signals with **PyTorch**.  
We take a window of recent features, run a **Temporal Convolutional Network (TCN)**, and predict the next move (a few bars ahead). Training/eval is all PyTorch (nn modules, DataLoader, AdamW, early stop). Output includes AUC/ACC and a quick returns-style PnL so you can iterate fast.

## 🆕 NEW: Real-time Trading Dashboard

**Live trading signals with WebSocket streaming!**
- 📡 Real-time data from Binance WebSocket
- 🤖 ML-powered LONG/SHORT signals
- 💰 Position sizing recommendations
- 📊 Interactive dashboard with charts and metrics

**Quick Launch:**
```powershell
# Download data and train model
python scripts/download_binance_historical.py --symbol BTCUSDT --days 90 --out data/btcusdt_1m.parquet
python src/train.py --config configs/default.yaml

# Launch dashboard
streamlit run dashboard/app.py
# or
.\launch_dashboard.ps1
```

📖 **[Full Live Trading Guide](LIVE_TRADING_GUIDE.md)**

---

## Why PyTorch here

- **Causal dilated convs (TCN):** big receptive field without heavy latency; great for 1m/1s streams.
- **Simple shape contract:** `(batch, channels, time)` → **logit**. Easy to swap models or features.
- **Fast loops:** pure torch training (BCEWithLogitsLoss, AdamW, grad clip), clean eval, easy to GPU later.
- **Extensible:** drop in other heads (Transformer/LSTM), add mixed precision, try custom losses.

---

## What you can use this for

- **Alpha research:** next-bar/next-N move classification on BTC (bars today, LOB later).
- **Real-time trading:** live predictions with WebSocket streaming and position sizing.
- **Regime flags:** rising vol/trend vs chop as a classifier target.
- **Signal stacking:** use the prob as a feature in a bigger ensemble.
- **Latency-sensitive forecasting:** TCN is lightweight and causal; good for near-real-time inference.

---

## Quickstart (Windows, CPU)

### 0) env

```powershell
# in repo root
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r alpha_tcn_starter/requirements.txt
```

### 1) Download data (NEW: from Binance API)

```powershell
cd alpha_tcn_starter

# Download last 90 days of BTCUSDT 1-minute bars
python scripts/download_binance_historical.py --symbol BTCUSDT --days 90 --out data/btcusdt_1m.parquet
```

This downloads ~130k bars (2-3 minutes). Or use existing data if you have it.

### 2) Train model

```powershell
python src/train.py --config configs/default.yaml
```

Output:
- `artifacts/model.pt` - trained model
- `artifacts/scaler.pkl` - feature scaler for inference
- `artifacts/meta.json` - metadata

### 3) Evaluate

```powershell
python src/eval.py --config configs/default.yaml --ckpt artifacts/model.pt
```

### 4) Launch live dashboard (NEW)

```powershell
streamlit run dashboard/app.py
```

Opens browser at `http://localhost:8501` with:
- Real-time price charts
- Live predictions (LONG/SHORT/NEUTRAL)
- Position sizing recommendations
- Performance metrics

See **[LIVE_TRADING_GUIDE.md](LIVE_TRADING_GUIDE.md)** for detailed instructions.
