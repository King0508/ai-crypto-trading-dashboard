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
    initial_sidebar_state="collapsed",
)

# Custom CSS for Robinhood-style dark theme
st.markdown(
    """
<style>
    .stApp {
        background-color: #000000;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .positive {
        color: #00C805;
        font-weight: 600;
    }
    .negative {
        color: #FF3B69;
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 600;
    }
    .signal-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 18px;
    }
    .signal-long {
        background-color: rgba(0, 200, 5, 0.2);
        color: #00C805;
        border: 2px solid #00C805;
    }
    .signal-short {
        background-color: rgba(255, 59, 105, 0.2);
        color: #FF3B69;
        border: 2px solid #FF3B69;
    }
    .signal-neutral {
        background-color: rgba(150, 150, 150, 0.2);
        color: #999999;
        border: 2px solid #999999;
    }
    .sim-badge {
        background-color: rgba(157, 78, 221, 0.2);
        color: #9D4EDD;
        border: 2px solid #9D4EDD;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
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
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#FFFFFF", family="Arial, sans-serif"),
        margin=dict(l=60, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor="#333333",
            borderwidth=1,
        ),
        xaxis3=dict(
            showgrid=True,
            gridcolor="#1a1a1a",
            title="Time",
            title_font=dict(size=12),
        ),
    )

    # Update all axes
    for i in [1, 2, 3]:
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#1a1a1a",
            row=i,
            col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#1a1a1a",
            row=i,
            col=1,
        )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, title_font=dict(size=12))
    fig.update_yaxes(title_text="Volume", row=2, col=1, title_font=dict(size=12))
    fig.update_yaxes(title_text="RSI", row=3, col=1, title_font=dict(size=12))

    return fig


def calculate_advanced_metrics(
    predictions: List[PredictionResult], df: pd.DataFrame
) -> Dict:
    """Calculate advanced trading metrics."""
    if len(predictions) < 2:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "total_signals": 0,
            "avg_confidence": 0.0,
        }

    active_signals = [p for p in predictions if p.signal != "NEUTRAL"]

    if not active_signals:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "total_signals": 0,
            "avg_confidence": 0.0,
        }

    # Calculate returns for each signal
    returns = []
    wins = 0

    for pred in active_signals[:-1]:
        future_idx = df[df["timestamp"] > pred.timestamp].head(10)
        if len(future_idx) > 0:
            future_price = future_idx["close"].iloc[-1]
            ret = (future_price - pred.price) / pred.price

            if pred.signal == "SHORT":
                ret = -ret

            returns.append(ret)

            if ret > 0:
                wins += 1

    if not returns:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "total_signals": len(active_signals),
            "avg_confidence": np.mean([p.confidence for p in active_signals]) * 100,
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
    }


def main():
    """Main simulation dashboard."""

    # Top bar with logo and status
    col1, col2, col3 = st.columns([2, 6, 2])

    with col1:
        st.markdown("# 🎮 **Crypto Sim**")
        st.markdown('<span class="sim-badge">SIMULATION</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Initialize session state
    if "stream" not in st.session_state:
        st.session_state.stream = None
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = deque(maxlen=100)
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "speed" not in st.session_state:
        st.session_state.speed = 10.0

    # Load config
    cfg = load_config("configs/live.yaml")

    # Control panel
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 4])

    with col1:
        if st.button("▶️ START", use_container_width=True, type="primary"):
            if not st.session_state.is_running:
                try:
                    # Load inference engine
                    st.session_state.engine = InferenceEngine(
                        model_path=cfg["model_checkpoint"],
                        scaler_path=cfg["scaler_path"],
                        meta_path=cfg["meta_path"],
                        device=cfg.get("device", "cpu"),
                        long_threshold=float(cfg.get("long_threshold", 0.55)),
                        short_threshold=float(cfg.get("short_threshold", 0.45)),
                        capital=float(cfg.get("capital", 10000.0)),
                        risk_per_trade=float(cfg.get("risk_per_trade", 0.02)),
                    )

                    # Create callback for new bars
                    def on_new_bar(df: pd.DataFrame, bar: Dict):
                        result = st.session_state.engine.predict(
                            df, current_price=bar["close"], timestamp=bar["timestamp"]
                        )
                        st.session_state.predictions.append(result)
                        st.session_state.last_update = datetime.now()

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
        if st.button("⏹️ STOP", use_container_width=True):
            if st.session_state.is_running and st.session_state.stream:
                st.session_state.stream.stop()
                st.session_state.is_running = False
                st.info("Stopped")

    with col3:
        if st.button("🔄 RESTART", use_container_width=True):
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
                st.progress(progress / 100)
                st.caption(f"{current}/{total} bars ({progress:.1f}%)")

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

    # Wait for data
    if not st.session_state.stream or len(st.session_state.stream.bars) < 60:
        bars_count = (
            len(st.session_state.stream.bars) if st.session_state.stream else 0
        )
        st.info(f"⏳ Collecting initial data... ({bars_count}/60 bars)")
        time.sleep(1)
        st.rerun()
        return

    # Get data
    df = st.session_state.stream.get_dataframe()
    predictions = list(st.session_state.predictions)
    current_pred = predictions[-1] if predictions else None

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if current_pred:
            price_change = df["ret_1"].iloc[-1] * 100
            st.metric(
                label="BTCUSDT",
                value=f"${current_pred.price:,.2f}",
                delta=f"{price_change:+.2f}%",
            )

    with col2:
        if current_pred:
            signal_class = f"signal-{current_pred.signal.lower()}"
            st.markdown(
                f'<div class="signal-badge {signal_class}">{current_pred.signal}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Confidence: {current_pred.confidence*100:.1f}%")

    with col3:
        if current_pred and current_pred.signal != "NEUTRAL":
            st.metric("Position Size", f"${current_pred.position_size_usd:,.0f}")
            st.caption(f"{current_pred.position_size_pct:.2f}% capital")

    with col4:
        metrics = calculate_advanced_metrics(predictions, df)
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        st.caption(f"Sharpe: {metrics['sharpe']:.2f}")

    with col5:
        st.metric("Total Signals", metrics["total_signals"])
        st.caption(f"Avg Conf: {metrics['avg_confidence']:.1f}%")

    st.markdown("---")

    # Main chart
    chart_df = df.tail(100)
    main_fig = create_main_chart(chart_df, predictions[-50:] if predictions else [])
    st.plotly_chart(main_fig, use_container_width=True)

    st.markdown("---")

    # Signal history
    st.markdown("### 📝 **Recent Signals**")

    if predictions:
        history_data = []
        for pred in reversed(predictions[-20:]):
            if pred.signal != "NEUTRAL":
                history_data.append(
                    {
                        "Time": pred.timestamp.strftime("%m/%d %H:%M"),
                        "Signal": pred.signal,
                        "Price": f"${pred.price:.2f}",
                        "Prob": f"{pred.probability*100:.1f}%",
                        "Conf": f"{pred.confidence*100:.1f}%",
                        "Position": f"${pred.position_size_usd:.0f}",
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

