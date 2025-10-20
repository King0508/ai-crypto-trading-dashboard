# Real-time Crypto Trading Dashboard Guide

Complete guide for setting up and running the live trading signal system.

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install new dependencies
pip install websocket-client streamlit plotly requests
```

### 2. Download Fresh Data (90 days of BTC)

```powershell
cd alpha_tcn_starter
python scripts/download_binance_historical.py --symbol BTCUSDT --days 90 --out data/btcusdt_1m.parquet
```

This will download ~130,000 1-minute bars (about 2-3 minutes).

### 3. Train the Model

```powershell
python src/train.py --config configs/default.yaml
```

This will:

- Train the TCN model on 90 days of data
- Save model to `artifacts/model.pt`
- Save scaler to `artifacts/scaler.pkl`
- Save metadata to `artifacts/meta.json`

Training takes ~5-10 minutes on CPU.

### 4. Validate Feature Alignment

**CRITICAL STEP:** Ensure live features match training features.

```powershell
python scripts/validate_live_features.py --data data/btcusdt_1m.parquet
```

You should see:

```
✓ VALIDATION PASSED
  Live features match training features!
```

### 5. Test Inference Engine

```powershell
python src/inference.py --model artifacts/model.pt --scaler artifacts/scaler.pkl --meta artifacts/meta.json --data data/btcusdt_1m.parquet
```

This will run predictions on the last 5 windows from historical data.

### 6. Launch Dashboard

```powershell
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Usage

### Controls (Left Sidebar)

1. **Configuration**

   - `Config Path`: Path to YAML config (default: `configs/live.yaml`)
   - `Symbol`: Trading pair (e.g., `btcusdt`)
   - `Capital`: Total capital for position sizing
   - `Long Threshold`: Probability threshold for LONG signals (default: 0.55)
   - `Short Threshold`: Probability threshold for SHORT signals (default: 0.45)
   - `Risk per Trade`: % of capital to risk per trade (default: 2%)

2. **Control Buttons**
   - `▶️ Start`: Connect to Binance WebSocket and start predictions
   - `⏹️ Stop`: Disconnect and stop

### Main Dashboard

#### Top Metrics

- **Current Price**: Latest BTC/USDT price with % change
- **Signal**: Current model prediction (🟢 LONG / 🔴 SHORT / ⚪ NEUTRAL)
- **Position Size**: Recommended position size in USD and %

#### Price Chart

- Candlestick chart with last 100 bars
- Volume bars below
- Green triangles (▲) mark LONG signals
- Red triangles (▼) mark SHORT signals

#### Probability Gauge

- Shows model's confidence (0-100%)
- Green zone (55-100%): LONG bias
- Gray zone (45-55%): NEUTRAL
- Red zone (0-45%): SHORT bias

#### Performance Metrics

- **Win Rate**: % of predictions that were correct
- **Total Signals**: Number of LONG/SHORT signals generated
- **Avg Confidence**: Average confidence of signals
- **Signals/Hour**: Rate of signal generation

#### Signal History

- Table of last 20 active signals
- Shows: Time, Signal, Price, Probability, Confidence, Position Size

---

## 🔧 Configuration

Edit `configs/live.yaml` to customize:

```yaml
# Trading pair
symbol: "btcusdt"
interval: "1m"

# Model paths
model_checkpoint: "artifacts/model.pt"
scaler_path: "artifacts/scaler.pkl"
meta_path: "artifacts/meta.json"

# Signal thresholds
long_threshold: 0.55 # Higher = more conservative LONG
short_threshold: 0.45 # Lower = more conservative SHORT

# Position sizing
capital: 10000.0 # Your total capital
risk_per_trade: 0.02 # Risk 2% per trade

# Dashboard settings
dashboard:
  refresh_interval: 60 # Seconds
  chart_bars: 100 # Bars to show on chart
  history_size: 50 # Signals to keep in history
```

---

## 📈 Understanding the Signals

### Signal Types

1. **LONG (🟢)**: Model predicts price will go UP
   - Triggered when: `probability >= long_threshold` (default: 0.55)
   - Action: Consider buying or holding
2. **SHORT (🔴)**: Model predicts price will go DOWN
   - Triggered when: `probability <= short_threshold` (default: 0.45)
   - Action: Consider selling or shorting
3. **NEUTRAL (⚪)**: Model is uncertain
   - Triggered when: `0.45 < probability < 0.55`
   - Action: Wait for clearer signal

### Position Sizing

The system uses a **Kelly-criterion-inspired** approach:

```
Base Size = Capital × Risk per Trade
Final Size = Base Size × (0.5 + Confidence)
```

**Example:**

- Capital: $10,000
- Risk per Trade: 2%
- Base Size: $200
- Confidence: 0.8 (80%)
- Final Size: $200 × (0.5 + 0.8) = $260

High confidence = larger position
Low confidence = smaller position

### Model Predictions

The model predicts **10 bars ahead** (10 minutes with 1m bars):

- Looks at last 60 bars (1 hour of history)
- Predicts: Will price be higher in 10 minutes?
- Outputs: Probability from 0 (definitely down) to 1 (definitely up)

---

## 🧪 Testing & Validation

### Test Live Stream (Without Model)

```powershell
python src/live_stream.py
```

This will:

1. Connect to Binance WebSocket
2. Stream 1-minute bars for 5 minutes
3. Print each bar as it closes
4. Show computed features

Use this to verify WebSocket connection works.

### Test Inference (Without Stream)

```powershell
python src/inference.py
```

This will:

1. Load trained model
2. Run predictions on historical data
3. Print signals for last 5 windows

Use this to verify model loading and predictions work.

### Validate Features

```powershell
python scripts/validate_live_features.py
```

**Always run this after training** to ensure live features match training features!

---

## 🎯 Best Practices

### 1. Model Retraining

Markets change! Retrain periodically:

```powershell
# Download fresh data
python scripts/download_binance_historical.py --symbol BTCUSDT --days 90 --out data/btcusdt_1m.parquet

# Retrain model
python src/train.py --config configs/default.yaml

# Validate
python scripts/validate_live_features.py

# Restart dashboard
streamlit run dashboard/app.py
```

Recommended schedule:

- Weekly for active trading
- Monthly for monitoring

### 2. Signal Filtering

Not all signals are equal! Consider:

- **High confidence only**: Only trade when confidence > 70%
- **Trend confirmation**: Wait for 2-3 consecutive signals in same direction
- **Volume filter**: Only trade signals during high volume periods

### 3. Risk Management

The dashboard provides **suggestions**, not trading commands!

- Never risk more than you can afford to lose
- Start with small capital for testing
- Use stop losses on actual trades
- Track actual performance separately

### 4. Market Conditions

Model trained on recent data (90 days):

- Works best in similar market conditions
- May struggle during:
  - Extreme volatility (news events, crashes)
  - Low liquidity (weekends, holidays)
  - Regime changes (bull → bear market)

---

## 🛠️ Troubleshooting

### Dashboard won't start

**Error:** `ModuleNotFoundError: No module named 'inference'`

**Fix:**

```powershell
# Make sure you're in the alpha_tcn_starter directory
cd alpha_tcn_starter

# Run from project root
streamlit run dashboard/app.py
```

### WebSocket connection fails

**Error:** `WebSocket closed` or repeated reconnection attempts

**Fix:**

1. Check internet connection
2. Verify Binance is accessible: https://www.binance.com
3. Try different symbol (e.g., `ethusdt`)
4. Check firewall settings

### Features don't match

**Error:** `VALIDATION FAILED: Live features DO NOT match training features`

**Fix:**

1. Ensure you're using same data preprocessing
2. Check for NaN handling differences
3. Verify rolling window calculations
4. Regenerate training data and retrain

### Model predictions seem random

**Possible causes:**

1. Model not trained properly (check training AUC > 0.55)
2. Market conditions changed significantly
3. Data quality issues

**Fix:**

1. Download fresh data
2. Retrain with more epochs
3. Check model architecture (try more layers/channels)

### Dashboard shows "Waiting for data..."

This is normal! The system needs:

- 60 bars minimum (1 hour with 1m bars)
- Wait patiently, bars arrive every minute
- If stuck > 5 minutes, restart dashboard

---

## 📊 Performance Expectations

### Model Performance (Typical)

From training on 90 days BTC data:

- **AUC**: 0.52 - 0.58 (>0.55 is good, >0.60 is excellent)
- **Accuracy**: 51% - 56%
- **Sharpe Ratio**: 0.5 - 2.0 (backtested)

**Note:** Crypto is noisy! Even 52% accuracy can be profitable with good risk management.

### Signal Frequency

With default thresholds (0.45/0.55):

- **Active signals**: 30-40% of bars
- **NEUTRAL**: 60-70% of bars
- **Signals per hour**: 18-25

Tighter thresholds (e.g., 0.40/0.60):

- More NEUTRAL periods
- Higher quality signals (but fewer)

---

## 🔮 Future Enhancements

Not implemented yet, but planned:

1. **Multi-crypto support**

   - Train on multiple pairs
   - Switch between BTC, ETH, SOL, etc.

2. **Auto-trading**

   - Connect to Binance API
   - Automatically execute signals
   - Track real P&L

3. **Advanced features**

   - Order book depth
   - Funding rates
   - Social sentiment

4. **Better risk metrics**

   - Real Sharpe ratio from live trades
   - Maximum drawdown tracking
   - Win/loss distribution

5. **Database persistence**
   - Store all signals
   - Historical analysis
   - Strategy backtesting

---

## 🆘 Support

For issues or questions:

1. Check this guide first
2. Review error messages in terminal
3. Test components individually (stream, inference, validation)
4. Check that model artifacts exist in `artifacts/`

---

## ⚠️ Disclaimer

This system is for **educational and research purposes** only.

- Not financial advice
- Past performance ≠ future results
- Cryptocurrency trading is risky
- Only trade with money you can afford to lose
- Always do your own research (DYOR)

**The developers are not responsible for any trading losses.**

---

## 📝 Quick Command Reference

```powershell
# Setup
pip install websocket-client streamlit plotly requests

# Download data
python scripts/download_binance_historical.py --symbol BTCUSDT --days 90 --out data/btcusdt_1m.parquet

# Train
python src/train.py --config configs/default.yaml

# Validate
python scripts/validate_live_features.py

# Test stream
python src/live_stream.py

# Test inference
python src/inference.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

Happy trading! 📈🚀
