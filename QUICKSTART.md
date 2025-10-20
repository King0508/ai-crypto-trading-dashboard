# ⚡ Quick Start Guide

Get your real-time crypto trading dashboard running in **5 minutes**!

## Step 1: Install Dependencies (1 min)

```powershell
# Make sure you're in your venv
.\.venv\Scripts\Activate.ps1

# Install new packages
pip install websocket-client streamlit plotly requests
```

## Step 2: Download Data (2-3 min)

```powershell
cd ai-crypto-trading-dashboard

# Download 90 days of BTC data from Binance
python scripts/download_binance_historical.py --symbol BTCUSDT --days 90 --out data/btcusdt_1m.parquet
```

**Expected output:**

```
Downloading BTCUSDT 1m from 2024-07-22 to 2024-10-20
Downloading: 100%|████████████| 130/130 [00:45<00:00]
Downloaded 129600 bars
✓ Saved 129600 bars to data/btcusdt_1m.parquet
```

## Step 3: Train Model (5-10 min)

```powershell
python src/train.py --config configs/default.yaml
```

**Expected output:**

```
Epoch 10/10: 100%|████████████| loss: 0.688
Val metrics: {'auc': 0.5642, 'acc': 0.5312}
Training done. Best val AUC: 0.5642
Saved best checkpoint to: artifacts/model.pt
```

**✓ You'll get 3 files:**

- `artifacts/model.pt` (trained model)
- `artifacts/scaler.pkl` (feature scaler)
- `artifacts/meta.json` (metadata)

## Step 4: Validate (30 sec)

```powershell
python scripts/validate_live_features.py
```

**Expected output:**

```
✓ VALIDATION PASSED
  Live features match training features!
  Safe to use for real-time predictions.
```

## Step 5: Launch Dashboard (instant)

```powershell
streamlit run dashboard/app.py
```

**Your browser will open to `http://localhost:8501`**

### In the Dashboard:

1. **Click "▶️ Start"** in left sidebar
2. **Wait 1-2 minutes** for initial 60 bars to collect
3. **Watch live signals** appear on the chart!

---

## 🎉 You're Live!

The dashboard will now:

- ✅ Stream real-time BTC prices from Binance
- ✅ Generate LONG/SHORT signals every minute
- ✅ Show recommended position sizes
- ✅ Track performance metrics

## 📊 What You'll See

### Main Chart

- Candlestick price chart
- 🟢 Green triangles = LONG signals
- 🔴 Red triangles = SHORT signals

### Current Signal

- **LONG**: Model expects price to rise
- **SHORT**: Model expects price to fall
- **NEUTRAL**: Model is uncertain (wait)

### Position Size

- Recommended amount to trade
- Based on your capital and signal confidence
- Example: "$260 (2.6%)" means trade $260, which is 2.6% of your capital

### Metrics

- **Win Rate**: How often predictions were correct
- **Total Signals**: Number of LONG/SHORT signals
- **Avg Confidence**: Average signal strength

---

## ⚙️ Customize Settings

In the sidebar, you can adjust:

- **Capital**: Your total trading capital (default: $10,000)
- **Long Threshold**: Higher = fewer LONG signals (default: 0.55)
- **Short Threshold**: Lower = fewer SHORT signals (default: 0.45)
- **Risk per Trade**: % to risk per signal (default: 2%)

---

## 🛑 To Stop

Press **Ctrl+C** in the terminal or click **"⏹️ Stop"** in dashboard.

---

## 📖 Learn More

- **[LIVE_TRADING_GUIDE.md](LIVE_TRADING_GUIDE.md)** - Comprehensive guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical details
- **[README.md](README.md)** - Project overview

---

## 🆘 Problems?

### Dashboard won't start

```powershell
pip install streamlit plotly websocket-client requests
```

### No model files

```powershell
python src/train.py --config configs/default.yaml
```

### WebSocket errors

- Check internet connection
- Verify Binance is accessible: https://www.binance.com
- Try restarting the dashboard

### Still stuck?

Check **[LIVE_TRADING_GUIDE.md](LIVE_TRADING_GUIDE.md)** → Troubleshooting section

---

## ⚠️ Important

This system is for **research and education**.

- Not financial advice
- Start with paper trading
- Never risk money you can't afford to lose
- Always use stop losses on real trades

---

**Happy trading! 📈🚀**
