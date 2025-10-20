"""
Real-time Crypto Trading Dashboard
Displays live predictions, signals, and trading metrics.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import time
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

from inference import InferenceEngine, PredictionResult
from live_stream import BinanceKlineStream


# Page config
st.set_page_config(
    page_title="Crypto Trading Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_config(config_path: str = "configs/live.yaml") -> Dict:
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_price_chart(
    df: pd.DataFrame, predictions: List[PredictionResult]
) -> go.Figure:
    """
    Create price chart with predictions overlay.

    Args:
        df: DataFrame with price data
        predictions: List of recent predictions

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Price & Signals", "Volume"),
        shared_xaxes=True,
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # Add signal markers
    if predictions:
        long_signals = [p for p in predictions if p.signal == "LONG"]
        short_signals = [p for p in predictions if p.signal == "SHORT"]

        if long_signals:
            fig.add_trace(
                go.Scatter(
                    x=[p.timestamp for p in long_signals],
                    y=[p.price for p in long_signals],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=15,
                        color="green",
                        line=dict(width=2, color="darkgreen"),
                    ),
                    name="LONG",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        if short_signals:
            fig.add_trace(
                go.Scatter(
                    x=[p.timestamp for p in short_signals],
                    y=[p.price for p in short_signals],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=15,
                        color="red",
                        line=dict(width=2, color="darkred"),
                    ),
                    name="SHORT",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    # Volume bars
    colors = [
        "red" if df.iloc[i]["close"] < df.iloc[i]["open"] else "green"
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_probability_gauge(probability: float, signal: str) -> go.Figure:
    """Create a gauge chart for prediction probability."""
    # Determine color based on signal
    if signal == "LONG":
        color = "green"
    elif signal == "SHORT":
        color = "red"
    else:
        color = "gray"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Prediction Probability", "font": {"size": 20}},
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 45], "color": "rgba(255, 0, 0, 0.2)"},
                    {"range": [45, 55], "color": "rgba(128, 128, 128, 0.2)"},
                    {"range": [55, 100], "color": "rgba(0, 255, 0, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Arial"},
    )

    return fig


def calculate_metrics(
    predictions: List[PredictionResult], prices: pd.DataFrame
) -> Dict:
    """
    Calculate performance metrics from predictions.

    Args:
        predictions: List of predictions with outcomes
        prices: DataFrame with price history

    Returns:
        Dictionary of metrics
    """
    if len(predictions) < 2:
        return {
            "win_rate": 0.0,
            "total_signals": 0,
            "avg_confidence": 0.0,
            "sharpe_ratio": 0.0,
            "current_drawdown": 0.0,
        }

    # Filter out NEUTRAL signals
    active_signals = [p for p in predictions if p.signal != "NEUTRAL"]

    if not active_signals:
        return {
            "win_rate": 0.0,
            "total_signals": 0,
            "avg_confidence": 0.0,
            "sharpe_ratio": 0.0,
            "current_drawdown": 0.0,
        }

    # Simple win rate calculation (check if price moved in predicted direction)
    # This is approximate - in reality would need to track actual outcomes
    wins = 0
    for pred in active_signals[:-1]:  # Exclude latest (no outcome yet)
        # Look ahead ~10 bars (horizon) to see if prediction was correct
        future_idx = prices[prices["timestamp"] > pred.timestamp].head(10)
        if len(future_idx) > 0:
            future_price = future_idx["close"].iloc[-1]
            if pred.signal == "LONG" and future_price > pred.price:
                wins += 1
            elif pred.signal == "SHORT" and future_price < pred.price:
                wins += 1

    win_rate = (wins / max(len(active_signals) - 1, 1)) * 100

    # Average confidence
    avg_confidence = np.mean([p.confidence for p in active_signals]) * 100

    # Simple metrics
    metrics = {
        "win_rate": win_rate,
        "total_signals": len(active_signals),
        "avg_confidence": avg_confidence,
        "sharpe_ratio": 0.0,  # Would need returns history
        "current_drawdown": 0.0,  # Would need equity curve
    }

    return metrics


def main():
    """Main dashboard application."""

    st.title("📈 Real-time Crypto Trading Signals")
    st.markdown("---")

    # Initialize session state
    if "stream" not in st.session_state:
        st.session_state.stream = None
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = deque(maxlen=50)
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Load config
    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/live.yaml",
    )

    try:
        cfg = load_config(config_path)
    except Exception as e:
        st.sidebar.error(f"Error loading config: {e}")
        return

    # Editable parameters
    symbol = st.sidebar.text_input("Symbol", value=cfg.get("symbol", "btcusdt"))
    capital = st.sidebar.number_input(
        "Capital (USD)",
        min_value=100.0,
        max_value=1000000.0,
        value=float(cfg.get("capital", 10000.0)),
        step=1000.0,
    )

    long_threshold = st.sidebar.slider(
        "Long Threshold",
        min_value=0.5,
        max_value=0.9,
        value=float(cfg.get("long_threshold", 0.55)),
        step=0.01,
    )

    short_threshold = st.sidebar.slider(
        "Short Threshold",
        min_value=0.1,
        max_value=0.5,
        value=float(cfg.get("short_threshold", 0.45)),
        step=0.01,
    )

    risk_per_trade = (
        st.sidebar.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=10.0,
            value=float(cfg.get("risk_per_trade", 0.02)) * 100,
            step=0.5,
        )
        / 100
    )

    st.sidebar.markdown("---")

    # Control buttons
    col1, col2 = st.sidebar.columns(2)

    start_button = col1.button("▶️ Start", use_container_width=True)
    stop_button = col2.button("⏹️ Stop", use_container_width=True)

    # Handle start
    if start_button and not st.session_state.is_running:
        with st.spinner("Initializing..."):
            try:
                # Load inference engine
                st.session_state.engine = InferenceEngine(
                    model_path=cfg["model_checkpoint"],
                    scaler_path=cfg["scaler_path"],
                    meta_path=cfg["meta_path"],
                    device=cfg.get("device", "cpu"),
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    capital=capital,
                    risk_per_trade=risk_per_trade,
                )

                # Create callback for new bars
                def on_new_bar(df: pd.DataFrame, bar: Dict):
                    """Called when a new bar closes."""
                    try:
                        result = st.session_state.engine.predict(
                            df,
                            current_price=bar["close"],
                            timestamp=bar["timestamp"],
                        )
                        st.session_state.predictions.append(result)
                        st.session_state.last_update = datetime.now()
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

                # Start stream
                st.session_state.stream = BinanceKlineStream(
                    symbol=symbol,
                    interval=cfg.get("interval", "1m"),
                    window_size=cfg.get("stream", {}).get("window_size", 200),
                    on_bar_callback=on_new_bar,
                )

                st.session_state.stream.start()
                st.session_state.is_running = True

                st.sidebar.success("✓ Started successfully!")

            except Exception as e:
                st.sidebar.error(f"Error starting: {e}")
                st.session_state.is_running = False

    # Handle stop
    if stop_button and st.session_state.is_running:
        if st.session_state.stream:
            st.session_state.stream.stop()
        st.session_state.is_running = False
        st.sidebar.success("✓ Stopped")

    # Status indicator
    if st.session_state.is_running:
        st.sidebar.success("🟢 LIVE")
        if st.session_state.last_update:
            st.sidebar.caption(
                f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}"
            )
    else:
        st.sidebar.warning("🔴 OFFLINE")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit")

    # Main dashboard content
    if not st.session_state.is_running:
        st.info("👈 Click **Start** in the sidebar to begin streaming live data")
        st.markdown(
            """
        ### Features:
        - 📊 Real-time price charts with technical indicators
        - 🤖 ML-powered buy/sell signals
        - 💰 Dynamic position sizing recommendations
        - 📈 Performance metrics and signal history
        
        ### How it works:
        1. Configure your parameters in the sidebar
        2. Click **Start** to connect to Binance WebSocket
        3. The system will collect 1-minute bars and generate predictions
        4. Watch for LONG 🟢 and SHORT 🔴 signals on the chart
        """
        )
        return

    # Wait for data
    if not st.session_state.stream or len(st.session_state.stream.bars) < 60:
        st.info(
            f"⏳ Collecting initial data... ({len(st.session_state.stream.bars) if st.session_state.stream else 0}/60 bars)"
        )
        time.sleep(2)
        st.rerun()
        return

    # Get current data
    df = st.session_state.stream.get_dataframe()
    predictions = list(st.session_state.predictions)

    # Latest prediction
    current_pred = predictions[-1] if predictions else None

    # Layout
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader(f"{symbol.upper()} / USDT")
        if current_pred:
            st.metric(
                "Current Price",
                f"${current_pred.price:,.2f}",
                delta=f"{df['ret_1'].iloc[-1]*100:.2f}%",
            )

    with col2:
        if current_pred:
            signal_color = {
                "LONG": "🟢",
                "SHORT": "🔴",
                "NEUTRAL": "⚪",
            }
            st.metric(
                "Signal",
                f"{signal_color.get(current_pred.signal, '')} {current_pred.signal}",
            )

    with col3:
        if current_pred and current_pred.signal != "NEUTRAL":
            st.metric(
                "Position Size",
                f"${current_pred.position_size_usd:,.0f}",
                delta=f"{current_pred.position_size_pct:.1f}%",
            )

    st.markdown("---")

    # Main row
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Price chart
        chart_df = df.tail(cfg.get("dashboard", {}).get("chart_bars", 100))
        fig = create_price_chart(
            chart_df, predictions[-20:] if len(predictions) > 0 else []
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Probability gauge
        if current_pred:
            gauge_fig = create_probability_gauge(
                current_pred.probability, current_pred.signal
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Confidence
            st.metric("Confidence", f"{current_pred.confidence*100:.1f}%")

    st.markdown("---")

    # Metrics row
    st.subheader("📊 Performance Metrics")

    metrics = calculate_metrics(predictions, df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

    with col2:
        st.metric("Total Signals", metrics["total_signals"])

    with col3:
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.1f}%")

    with col4:
        st.metric(
            "Signals/Hour",
            f"{len([p for p in predictions if p.signal != 'NEUTRAL'])/(len(predictions)/60):.1f}",
        )

    st.markdown("---")

    # Signal history
    st.subheader("📝 Recent Signals")

    if predictions:
        # Create history dataframe
        history_data = []
        for pred in reversed(predictions[-20:]):
            if pred.signal != "NEUTRAL":
                history_data.append(
                    {
                        "Time": pred.timestamp.strftime("%Y-%m-%d %H:%M"),
                        "Signal": pred.signal,
                        "Price": f"${pred.price:.2f}",
                        "Probability": f"{pred.probability*100:.1f}%",
                        "Confidence": f"{pred.confidence*100:.1f}%",
                        "Position": f"${pred.position_size_usd:.0f}",
                    }
                )

        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active signals yet (all NEUTRAL)")
    else:
        st.info("Waiting for predictions...")

    # Auto-refresh
    time.sleep(10)  # Refresh every 10 seconds
    st.rerun()


if __name__ == "__main__":
    main()
