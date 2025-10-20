"""
Professional Crypto Trading Dashboard - SIMULATION MODE
Fast testing with historical data replay at adjustable speeds.
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
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

from inference import InferenceEngine, PredictionResult
from simulation_stream import SimulationStream


# Robinhood-inspired styling
st.set_page_config(
    page_title="Crypto Trading Sim",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Robinhood-style dark theme
st.markdown(
    """
<style>
    /* Robinhood-inspired theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0D0D0D;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Prevent scroll jump on refresh */
    html {
        scroll-behavior: smooth;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
        max-width: 1400px;
    }
    
    /* Robinhood-style cards */
    .metric-card {
        background: #1C1C1E;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #2C2C2E;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #3C3C3E;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 28px;
        font-weight: 700;
    }
    
    h3 {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    /* Robinhood green/red colors */
    .positive {
        color: #00C805;
        font-weight: 600;
    }
    
    .negative {
        color: #FF3B69;
        font-weight: 600;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #8E8E93;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Signal badges - Robinhood style */
    .signal-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
    }
    
    .signal-long {
        background-color: rgba(0, 200, 5, 0.15);
        color: #00C805;
        border: 2px solid rgba(0, 200, 5, 0.3);
    }
    
    .signal-short {
        background-color: rgba(255, 59, 105, 0.15);
        color: #FF3B69;
        border: 2px solid rgba(255, 59, 105, 0.3);
    }
    
    .signal-neutral {
        background-color: rgba(142, 142, 147, 0.15);
        color: #8E8E93;
        border: 2px solid rgba(142, 142, 147, 0.3);
    }
    
    /* Simulation badge */
    .sim-badge {
        background-color: rgba(157, 78, 221, 0.15);
        color: #9D4EDD;
        border: 2px solid rgba(157, 78, 221, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Buttons - Robinhood style */
    .stButton>button {
        background-color: #1C1C1E;
        color: #FFFFFF;
        border: 1px solid #2C2C2E;
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 12px 24px;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2C2C2E;
        border-color: #3C3C3E;
    }
    
    .stButton>button[kind="primary"] {
        background-color: #00C805;
        color: #000000;
        border: none;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #00E005;
    }
    
    /* Divider */
    hr {
        border-color: #2C2C2E;
        margin: 1.5rem 0;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1C1C1E;
        border: 1px solid #2C2C2E;
        border-radius: 12px;
        color: #FFFFFF;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background-color: #1C1C1E;
        border-radius: 12px;
    }
    
    /* Selectbox */
    .stSelectbox {
        background-color: #1C1C1E;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #00C805;
    }
    
    /* Caption text */
    .caption {
        color: #8E8E93;
        font-size: 13px;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_config(config_path: str = "configs/live.yaml") -> Dict:
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_main_chart(
    df: pd.DataFrame, predictions: List[PredictionResult]
) -> go.Figure:
    """Create main TradingView-style chart with price, volume, and signals."""
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=("", "", ""),
        shared_xaxes=True,
    )

    # Main price candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#00C805",
            increasing_fillcolor="#00C805",
            decreasing_line_color="#FF3B69",
            decreasing_fillcolor="#FF3B69",
        ),
        row=1,
        col=1,
    )

    # Add 20-period SMA
    sma_20 = df["close"].rolling(20).mean()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=sma_20,
            name="SMA(20)",
            line=dict(color="#FFA500", width=1.5),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Add 60-period SMA
    sma_60 = df["close"].rolling(60).mean()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=sma_60,
            name="SMA(60)",
            line=dict(color="#00BFFF", width=1.5),
            opacity=0.7,
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
                    y=[p.price * 0.995 for p in long_signals],
                    mode="markers+text",
                    marker=dict(
                        symbol="triangle-up",
                        size=20,
                        color="#00C805",
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    text=["BUY" for _ in long_signals],
                    textposition="bottom center",
                    textfont=dict(color="#00C805", size=10, family="Arial Black"),
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
                    y=[p.price * 1.005 for p in short_signals],
                    mode="markers+text",
                    marker=dict(
                        symbol="triangle-down",
                        size=20,
                        color="#FF3B69",
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    text=["SELL" for _ in short_signals],
                    textposition="top center",
                    textfont=dict(color="#FF3B69", size=10, family="Arial Black"),
                    name="SHORT",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    # Volume with color coding
    colors = [
        "#00C805" if df.iloc[i]["close"] >= df.iloc[i]["open"] else "#FF3B69"
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # RSI indicator
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=rsi,
            name="RSI(14)",
            line=dict(color="#9D4EDD", width=2),
        ),
        row=3,
        col=1,
    )

    # RSI reference lines
    fig.add_hline(
        y=70, line_dash="dash", line_color="rgba(255, 59, 105, 0.5)", row=3, col=1
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color="rgba(0, 200, 5, 0.5)", row=3, col=1
    )
    fig.add_hline(
        y=50, line_dash="dot", line_color="rgba(150, 150, 150, 0.3)", row=3, col=1
    )

    # Update layout with Robinhood-style dark theme
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#0D0D0D",
        paper_bgcolor="#0D0D0D",
        font=dict(
            color="#FFFFFF",
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            size=12,
        ),
        margin=dict(l=60, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(28, 28, 30, 0.9)",
            bordercolor="#2C2C2E",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis3=dict(
            showgrid=True,
            gridcolor="#1C1C1E",
            title="Time",
            title_font=dict(size=12, color="#8E8E93"),
        ),
        hoverlabel=dict(
            bgcolor="#1C1C1E",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="#2C2C2E",
        ),
    )

    # Update all axes with Robinhood-style colors
    for i in [1, 2, 3]:
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#1C1C1E",
            gridwidth=1,
            zeroline=False,
            row=i,
            col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#1C1C1E",
            gridwidth=1,
            zeroline=False,
            row=i,
            col=1,
        )

    fig.update_yaxes(
        title_text="Price (USD)",
        row=1,
        col=1,
        title_font=dict(size=12, color="#8E8E93"),
    )
    fig.update_yaxes(
        title_text="Volume", row=2, col=1, title_font=dict(size=12, color="#8E8E93")
    )
    fig.update_yaxes(
        title_text="RSI", row=3, col=1, title_font=dict(size=12, color="#8E8E93")
    )

    return fig


def calculate_all_time_pnl(predictions: List[PredictionResult], min_confidence: float = 0.0) -> Dict:
    """Calculate all-time PnL and trade stats across ALL predictions.
    
    Args:
        predictions: List of prediction results
        min_confidence: Minimum confidence threshold to filter trades (0.0 to 1.0)
    """
    if len(predictions) < 11:
        return {"all_time_pnl": 0.0, "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "filtered_out": 0}
    
    total_pnl = 0.0
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    filtered_out = 0
    
    # For each prediction, calculate PnL by looking 10 bars ahead
    for i in range(len(predictions) - 10):
        pred = predictions[i]
        
        # Only calculate PnL for non-NEUTRAL signals
        if pred.signal == "NEUTRAL":
            continue
        
        # Filter by minimum confidence
        if pred.confidence < min_confidence:
            filtered_out += 1
            continue
            
        total_trades += 1
        
        # Get exit price 10 bars later
        exit_pred = predictions[i + 10]
        entry_price = pred.price
        exit_price = exit_pred.price
        
        # Calculate return
        ret = (exit_price - entry_price) / entry_price
        if pred.signal == "SHORT":
            ret = -ret
            
        # Calculate PnL
        pnl = ret * pred.position_size_usd
        total_pnl += pnl
        
        if pnl > 0:
            winning_trades += 1
        else:
            losing_trades += 1
    
    return {
        "all_time_pnl": total_pnl,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "filtered_out": filtered_out,
    }


def calculate_advanced_metrics(
    predictions: List[PredictionResult], df: pd.DataFrame
) -> Dict:
    """Calculate advanced trading metrics including PnL."""
    if len(predictions) < 2:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "total_signals": 0,
            "avg_confidence": 0.0,
            "total_pnl": 0.0,
            "pnl_per_signal": {},
        }

    active_signals = [p for p in predictions if p.signal != "NEUTRAL"]

    if not active_signals:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "total_signals": 0,
            "avg_confidence": 0.0,
            "total_pnl": 0.0,
            "pnl_per_signal": {},
        }

    # Calculate returns and PnL for each signal
    returns = []
    wins = 0
    total_pnl = 0.0
    pnl_per_signal = {}

    for pred in predictions:
        if pred.signal == "NEUTRAL":
            pnl_per_signal[pred.timestamp] = 0.0
            continue

        # Look forward 10 bars to calculate PnL
        future_idx = df[df["timestamp"] > pred.timestamp].head(10)
        if len(future_idx) > 0:
            future_price = future_idx["close"].iloc[-1]
            ret = (future_price - pred.price) / pred.price

            if pred.signal == "SHORT":
                ret = -ret

            # Calculate dollar PnL
            pnl = ret * pred.position_size_usd
            pnl_per_signal[pred.timestamp] = pnl
            total_pnl += pnl

            returns.append(ret)
            if ret > 0:
                wins += 1
        else:
            # Signal too recent, no exit yet
            pnl_per_signal[pred.timestamp] = 0.0

    if not returns:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "total_signals": len(active_signals),
            "avg_confidence": np.mean([p.confidence for p in active_signals]) * 100,
            "total_pnl": 0.0,
            "pnl_per_signal": pnl_per_signal,
        }

    # Win rate
    win_rate = (wins / len(returns)) * 100 if returns else 0

    # Sharpe ratio
    returns_arr = np.array(returns)
    sharpe = (
        (np.mean(returns_arr) / (np.std(returns_arr) + 1e-9)) * np.sqrt(525600)
        if len(returns_arr) > 0
        else 0
    )

    # Average confidence
    avg_confidence = np.mean([p.confidence for p in active_signals]) * 100

    return {
        "win_rate": win_rate,
        "sharpe": sharpe,
        "total_signals": len(active_signals),
        "avg_confidence": avg_confidence,
        "total_pnl": total_pnl,
        "pnl_per_signal": pnl_per_signal,
    }


def main():
    """Main simulation dashboard."""

    # Top bar with logo and status
    col1, col2, col3 = st.columns([2, 6, 2])

    with col1:
        st.markdown("# 🎮 **Crypto Sim**")
        st.markdown('<span class="sim-badge">SIMULATION</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Create persistent containers for stable rendering (prevents scroll jump)
    control_panel_container = st.container()
    metrics_container = st.container()
    chart_container = st.container()
    signals_container = st.container()

    # Initialize session state
    if "stream" not in st.session_state:
        st.session_state.stream = None
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = deque(maxlen=10000)  # Store up to 10k signals
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "speed" not in st.session_state:
        st.session_state.speed = 10.0

    # Load config
    cfg = load_config("configs/live.yaml")

    # Settings panel in sidebar
    with st.sidebar:
        st.markdown("## ⚙️ **Trading Settings**")

        # Preset or Custom thresholds
        threshold_mode = st.radio(
            "Threshold Mode",
            ["Preset", "Custom"],
            horizontal=True,
        )

        if threshold_mode == "Preset":
            preset = st.selectbox(
                "Strategy Preset",
                [
                    "Conservative (Fewer, Higher Quality)",
                    "Balanced (Medium Signals)",
                    "Aggressive (More Signals)",
                ],
                index=1,
            )

            if "Conservative" in preset:
                long_thresh = 0.55
                short_thresh = 0.45
                st.info("🛡️ Conservative: Only high-confidence signals")
            elif "Balanced" in preset:
                long_thresh = 0.52
                short_thresh = 0.48
                st.info("⚖️ Balanced: Medium signal frequency")
            else:  # Aggressive
                long_thresh = 0.505
                short_thresh = 0.495
                st.success("⚡ Aggressive: Maximum signals (testing)")
        else:  # Custom
            col1, col2 = st.columns(2)
            with col1:
                long_thresh = st.number_input(
                    "LONG Threshold",
                    min_value=0.50,
                    max_value=0.99,
                    value=0.52,
                    step=0.01,
                    help="Probability needed to trigger LONG signal",
                )
            with col2:
                short_thresh = st.number_input(
                    "SHORT Threshold",
                    min_value=0.01,
                    max_value=0.50,
                    value=0.48,
                    step=0.01,
                    help="Probability needed to trigger SHORT signal",
                )

        st.markdown("---")

        # Position sizing settings
        st.markdown("### 💰 Position Sizing")
        capital = st.number_input(
            "Total Capital ($)",
            min_value=100.0,
            max_value=10000000.0,
            value=float(cfg.get("capital", 10000.0)),
            step=1000.0,
        )

        risk_pct = st.slider(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(cfg.get("risk_per_trade", 0.02)) * 100,
            step=0.1,
            help="Percentage of capital to risk per trade",
        )
        risk_per_trade = risk_pct / 100

        st.caption(f"Base position size: ${capital * risk_per_trade:,.2f}")
        st.caption(
            f"Range: ${capital * risk_per_trade * 0.5:,.2f} - ${capital * risk_per_trade * 1.5:,.2f}"
        )
        
        st.markdown("---")
        
        # Trade Strategy Settings
        st.markdown("### 🎯 Trade Strategy")
        
        min_confidence = st.slider(
            "Min Confidence to Trade (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=5.0,
            help="Only take trades when model confidence is above this threshold"
        )
        min_conf_decimal = min_confidence / 100
        
        st.caption("**Strategy:** Higher confidence = larger position size")
        st.caption("Position multiplier: 0.5x to 1.5x based on confidence")

    # Control panel - use container for stable positioning
    with control_panel_container:
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 4])

        with col1:
            if st.button(
                "▶️ START", use_container_width=True, type="primary", key="start_btn"
            ):
                if not st.session_state.is_running:
                    try:
                        # Load inference engine with user-selected settings
                        st.session_state.engine = InferenceEngine(
                            model_path=cfg["model_checkpoint"],
                            scaler_path=cfg["scaler_path"],
                            meta_path=cfg["meta_path"],
                            device=cfg.get("device", "cpu"),
                            long_threshold=float(long_thresh),
                            short_threshold=float(short_thresh),
                            capital=float(capital),
                            risk_per_trade=float(risk_per_trade),
                        )

                        # Create callback for new bars
                        # Capture engine and predictions in closure to avoid threading issues
                        engine = st.session_state.engine
                        predictions = st.session_state.predictions

                        def on_new_bar(df: pd.DataFrame, bar: Dict):
                            result = engine.predict(
                                df,
                                current_price=bar["close"],
                                timestamp=bar["timestamp"],
                            )
                            predictions.append(result)

                        # Start simulation stream
                        st.session_state.stream = SimulationStream(
                            data_path="data/btcusdt_1m.parquet",
                            window_size=200,
                            speed_multiplier=st.session_state.speed,
                            on_bar_callback=on_new_bar,
                        )

                        st.session_state.stream.start()
                        st.session_state.is_running = True
                        st.success(f"Started at {st.session_state.speed}x speed!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            if st.button("⏹️ STOP", use_container_width=True, key="stop_btn"):
                if st.session_state.is_running and st.session_state.stream:
                    st.session_state.stream.stop()
                    st.session_state.is_running = False
                    st.info("Stopped")

        with col3:
            if st.button("🔄 RESTART", use_container_width=True, key="restart_btn"):
                if st.session_state.stream:
                    st.session_state.predictions.clear()
                    st.session_state.stream.restart()
                    st.info("Restarted!")

        with col4:
            # Speed selector
            speed_options = {"1x": 1.0, "5x": 5.0, "10x": 10.0, "50x": 50.0}
            selected_speed = st.selectbox(
                "Speed",
                options=list(speed_options.keys()),
                index=2,  # Default to 10x
                key="speed_selector",
            )
            new_speed = speed_options[selected_speed]
            if new_speed != st.session_state.speed:
                st.session_state.speed = new_speed
                if st.session_state.stream:
                    st.session_state.stream.set_speed(new_speed)

        with col5:
            if st.session_state.is_running:
                progress = 0
                if st.session_state.stream:
                    total = len(st.session_state.stream.full_data)
                    current = st.session_state.stream.current_index
                    progress = (current / total) * 100 if total > 0 else 0
                    st.progress(
                        progress / 100, text=f"{current}/{total} bars ({progress:.1f}%)"
                    )

        with col6:
            if st.session_state.is_running:
                st.success(f"🟢 RUNNING @ {st.session_state.speed}x")
            else:
                st.warning("🔴 STOPPED")

        st.markdown("---")

    # Main content
    if not st.session_state.is_running:
        st.info("👆 Click START to begin simulation")
        st.markdown(
            """
        ### Simulation Mode - Fast Testing!
        - ⚡ **10x Speed**: See 60 bars in 6 seconds (not 60 minutes!)
        - 🔄 **Restart**: Replay same data to test strategies
        - 📊 **Full Dashboard**: Same charts and metrics as live mode
        - 🎮 **No VPN Needed**: Works completely offline
        
        **Perfect for:**
        - Testing the system quickly
        - Backtesting strategies
        - Demonstrating functionality
        - Learning how signals work
        """
        )
        return

    # Only wait for initial data if we truly have nothing yet
    if not st.session_state.stream:
        time.sleep(1)
        st.rerun()
        return
        
    # Show initial loading ONLY when we have less than 60 bars AND zero predictions
    if len(st.session_state.stream.bars) < 60 and len(st.session_state.predictions) == 0:
        bars_count = len(st.session_state.stream.bars)
        st.info(f"⏳ Warming up... ({bars_count}/60 bars)")
        time.sleep(1)
        st.rerun()
        return

    # Get data
    df = st.session_state.stream.get_dataframe()
    predictions = list(st.session_state.predictions)
    current_pred = predictions[-1] if predictions else None

    # Top metrics row - use container for stable rendering
    with metrics_container:
        metrics = calculate_advanced_metrics(predictions, df)
        
        # Calculate all-time PnL from predictions
        all_time_stats = calculate_all_time_pnl(predictions)
        all_time_pnl = all_time_stats["all_time_pnl"]
        total_trades = all_time_stats["total_trades"]
        winning_trades = all_time_stats["winning_trades"]
        losing_trades = all_time_stats["losing_trades"]
        
        # Calculate win rate
        win_rate_all_time = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 2])
        
        with col1:
            # ALL-TIME PnL - MOST IMPORTANT METRIC
            st.metric(
                "💰 ALL-TIME PROFIT/LOSS",
                f"${all_time_pnl:,.2f}",
                delta=f"{all_time_pnl:+,.2f}",
                delta_color="normal" if all_time_pnl >= 0 else "inverse",
            )
            st.caption(f"Trades: {total_trades} ({winning_trades}W/{losing_trades}L) | Win Rate: {win_rate_all_time:.1f}%")
        
        with col2:
            if current_pred:
                price_change = df["ret_1"].iloc[-1] * 100
                st.metric(
                    label="BTCUSDT",
                    value=f"${current_pred.price:,.2f}",
                    delta=f"{price_change:+.2f}%",
                )
        
        with col3:
            if current_pred:
                signal_class = f"signal-{current_pred.signal.lower()}"
                st.markdown(
                    f'<div class="signal-badge {signal_class}">{current_pred.signal}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Confidence: {current_pred.confidence*100:.1f}%")
        
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            st.caption(f"Sharpe: {metrics['sharpe']:.2f} | Signals: {metrics['total_signals']}")

        st.markdown("---")

    # Main chart - use container with key for stable rendering
    with chart_container:
        chart_df = df.tail(100)
        main_fig = create_main_chart(chart_df, predictions[-50:] if predictions else [])
        st.plotly_chart(
            main_fig, use_container_width=True, key=f"main_chart_{len(predictions)}"
        )

        st.markdown("---")

    # Signal history - use container
    with signals_container:
        st.markdown("### 📝 **Recent Signals**")

        if predictions:
            pnl_map = metrics.get("pnl_per_signal", {})
            history_data = []
            for pred in reversed(predictions[-20:]):
                if pred.signal != "NEUTRAL":
                    pnl = pnl_map.get(pred.timestamp, 0.0)
                    history_data.append(
                        {
                            "Time": pred.timestamp.strftime("%m/%d %H:%M"),
                            "Signal": pred.signal,
                            "Price": f"${pred.price:.2f}",
                            "Prob": f"{pred.probability*100:.1f}%",
                            "Conf": f"{pred.confidence*100:.1f}%",
                            "Position": f"${pred.position_size_usd:.0f}",
                            "PnL": f"${pnl:+,.2f}" if pnl != 0 else "$0.00",
                        }
                    )

            if history_data:
                st.dataframe(
                    history_data,
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )
            else:
                st.info("No active signals yet (all NEUTRAL)")

    # Auto-refresh
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
