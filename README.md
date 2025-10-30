# Automated BTC Trading Bot with TCN Model

Algorithmic trading system using Temporal Convolutional Networks (TCN) for BTC price prediction and automated trading on Alpaca.

## Features

- 🤖 **Automated Trading**: TCN deep learning model for BTC predictions
- 📊 **Live Dashboard**: Real-time monitoring with Streamlit
- 🔄 **Paper Trading**: Test strategies risk-free on Alpaca
- 📈 **Risk Management**: Position sizing, stop losses, take profits
- 💰 **Performance Tracking**: Win rate, P/L, and trade history

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add Alpaca credentials to .env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### 2. Train Model

```bash
python src/train.py --config configs/default.yaml
```

### 3. Start Trading

```bash
# Start bot
python run_bot.py

# Start dashboard (separate terminal)
python run_dashboard.py
```

## Project Structure

```
├── src/
│   ├── exchange/          # Alpaca API integration
│   ├── database/          # Trade history storage
│   ├── trading/           # Trading bot logic
│   ├── model.py           # TCN architecture
│   ├── train.py           # Model training
│   └── inference.py       # Prediction engine
├── dashboard/
│   └── app_live.py        # Streamlit dashboard
├── configs/               # Configuration files
├── artifacts/             # Trained models
└── data/                  # Historical data
```

## Configuration

Edit `run_bot.py` to adjust:
- `long_threshold`: Confidence for LONG trades
- `short_threshold`: Confidence for SHORT trades  
- `risk_per_trade`: Position sizing
- `interval`: Update frequency (seconds)

## Tech Stack

- **ML**: PyTorch, scikit-learn
- **Trading**: Alpaca API
- **Dashboard**: Streamlit, Plotly
- **Data**: SQLite, pandas

## Risk Disclaimer

⚠️ **Educational purposes only. Not financial advice.**
- Paper trading recommended for testing
- Past performance doesn't guarantee future results
- Cryptocurrency trading carries significant risk

## License

MIT License
