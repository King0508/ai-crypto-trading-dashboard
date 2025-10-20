# 🎮 Simulation Mode - Quick Start Guide

Test your trading dashboard in **minutes** instead of hours!

## 🚀 Launch Simulation Dashboard

```powershell
cd ai-crypto-trading-dashboard
streamlit run dashboard/app_sim.py
```

Browser opens at `http://localhost:8501`

---

## ⚡ Quick Demo

1. **Click "▶️ START"** button
2. **Select speed**: Choose "10x" from dropdown (default)
3. **Watch it work**:
   - 60 bars collected in **6 seconds** (not 60 minutes!)
   - Charts appear immediately
   - Signals start generating
   - Metrics update in real-time

---

## 🎛️ Controls

### **Speed Options**

- **1x**: Real-time (1 bar per second)
- **5x**: 5 bars per second
- **10x**: 10 bars per second (recommended)
- **50x**: Ultra-fast testing

### **Buttons**

- **▶️ START**: Begin simulation
- **⏹️ STOP**: Stop simulation
- **🔄 RESTART**: Replay from beginning

### **Progress Bar**

Shows how much historical data has been processed

---

## 📊 What You'll See

### **Timeline**

**At 10x speed:**

```
0:00  → Click START
0:06  → 60 bars collected, charts appear!
0:12  → 120 bars, signals appearing
0:30  → 300 bars, metrics stabilizing
1:00  → 600 bars, full performance data
```

**At 1x speed:**

```
0:00  → Click START
1:00  → 60 bars collected
2:00  → 120 bars
5:00  → 300 bars
10:00 → 600 bars
```

### **Dashboard Features**

1. **Price Chart**

   - Candlestick display
   - SMA(20) and SMA(60) moving averages
   - RSI indicator below
   - Volume bars

2. **Signals**

   - 🟢 Green triangles = LONG (buy)
   - 🔴 Red triangles = SHORT (sell)
   - BUY/SELL text on chart

3. **Live Metrics**

   - Current price with % change
   - Signal (LONG/SHORT/NEUTRAL)
   - Position size recommendation
   - Win rate
   - Sharpe ratio
   - Total signals

4. **Signal History Table**
   - Last 20 active signals
   - Time, Signal, Price, Probability, Position

---

## 🎯 Use Cases

### **1. Quick Testing**

```powershell
# See if everything works in 1 minute
streamlit run dashboard/app_sim.py
# Select 50x speed → Full test in seconds!
```

### **2. Strategy Testing**

```powershell
# Test different thresholds
1. Click START at 10x speed
2. Let it run for 2 minutes (1200 bars)
3. Note win rate and Sharpe ratio
4. Click RESTART
5. Adjust Long/Short thresholds in config
6. Compare results!
```

### **3. Demo to Others**

```powershell
# Show how the system works without waiting
- Use 10x-50x speed
- Point out signals appearing on chart
- Show metrics updating
- Restart to show repeatability
```

### **4. Training Different Models**

```powershell
# Compare model performance
1. Train model A
2. Run simulation, note metrics
3. Train model B (different parameters)
4. Run simulation again
5. Compare which model performs better!
```

---

## ⚙️ Configuration

Edit `configs/live.yaml` to change:

```yaml
# Signal thresholds
long_threshold: 0.55 # Higher = fewer LONGs
short_threshold: 0.45 # Lower = fewer SHORTs

# Position sizing
capital: 10000.0 # Your capital
risk_per_trade: 0.02 # Risk 2% per trade

# Model paths (use existing model)
model_checkpoint: "artifacts/model.pt"
scaler_path: "artifacts/scaler.pkl"
meta_path: "artifacts/meta.json"
```

---

## 🆚 Simulation vs Live

| Feature          | Simulation          | Live              |
| ---------------- | ------------------- | ----------------- |
| **Speed**        | 1x-50x (adjustable) | 1x (real-time)    |
| **Data Source**  | Historical parquet  | Binance WebSocket |
| **VPN Required** | ❌ No               | ✅ Yes            |
| **Wait Time**    | 6 sec (at 10x)      | 60 min            |
| **Repeatable**   | ✅ Yes (restart)    | ❌ No             |
| **Best For**     | Testing, learning   | Real trading      |

---

## 🐛 Troubleshooting

### **"No module named 'simulation_stream'"**

```powershell
# Make sure you're in ai-crypto-trading-dashboard directory
cd ai-crypto-trading-dashboard
streamlit run dashboard/app_sim.py
```

### **"Model files not found"**

```powershell
# Check if artifacts exist
dir artifacts

# If missing, train the model first
python src/train.py --config configs/default.yaml
```

### **"Data file not found"**

```powershell
# Check if data exists
dir data\btcusdt_1m.parquet

# If missing, download it
python scripts/download_binance_historical.py --days 90
```

### **Charts not showing up**

- Wait for "Collecting initial data (60/60 bars)" message
- At 10x speed, this takes only 6 seconds
- If stuck at 0 bars, restart the dashboard

### **Simulation runs too fast/slow**

- Use dropdown to change speed
- 1x = realistic speed
- 10x = good balance (recommended)
- 50x = ultra-fast testing

---

## 💡 Pro Tips

1. **Start with 10x speed** for good balance of speed and visibility

2. **Use RESTART** to test different strategies on same data

3. **Compare metrics**:

   - Win Rate > 52% is good for crypto
   - Sharpe Ratio > 1.0 is decent
   - More signals doesn't mean better!

4. **Watch for patterns**:

   - Does model work better in volatile periods?
   - Are SHORT signals less accurate than LONG?
   - How does changing thresholds affect win rate?

5. **Test before live**:
   - Sim mode shows you what to expect
   - Once comfortable, switch to live mode
   - Retrain model with fresh data monthly

---

## 📈 Next Steps

Once you're happy with simulation mode:

### **1. Train Fresh Model** (if needed)

```powershell
python scripts/download_binance_historical.py --days 90
python src/train.py --config configs/default.yaml
```

### **2. Switch to Live Mode**

```powershell
# Connect VPN (Netherlands or non-US)
streamlit run dashboard/app_pro.py
# Click START, wait 60 minutes for real bars
```

### **3. Flask + React Version** (future)

- True Robinhood design
- Custom fonts and animations
- Mobile responsive
- Production-ready

---

## 🎉 Success Checklist

✅ Dashboard launches without errors
✅ Charts appear after 6-60 seconds (depends on speed)
✅ Signals (🟢 LONG / 🔴 SHORT) show on chart
✅ Metrics update (Win Rate, Sharpe, etc.)
✅ Signal history table populates
✅ Can RESTART and replay
✅ Can change speed and see difference

---

**Ready to test?** Run `streamlit run dashboard/app_sim.py` and see your trading system in action! 🚀
