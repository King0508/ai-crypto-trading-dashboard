"""
Professional Crypto Trading Dashboard - Robinhood/TradingView Inspired
Enhanced UI with multiple charts and quantitative analysis.
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
from live_stream import BinanceKlineStream


# Robinhood-inspired styling
st.set_page_config(
    page_title="Crypto Trading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Robinhood-style dark theme
st.markdown(
    """
<style>
    /* Robinhood dark theme */
    .stApp {
        background-color: #000000;
    }
    
    /* Main metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Green for positive */
    .positive {
        color: #00C805;
        font-weight: 600;
    }
    
    /* Red for negative */
    .negative {
        color: #FF3B69;
        font-weight: 600;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Streamlit metric styling */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 600;
    }
    
    /* Signal badge */
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
    """
    Create main TradingView-style chart with price, volume, and signals.
    """
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


def create_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Create volatility analysis chart."""
    fig = go.Figure()

    # 20-bar rolling volatility
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["roll_vol_20"] * 100,
            name="Vol(20)",
            line=dict(color="#FFA500", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.2)",
        )
    )

    # 60-bar rolling volatility
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["roll_vol_60"] * 100,
            name="Vol(60)",
            line=dict(color="#00BFFF", width=2),
        )
    )

    fig.update_layout(
        title="Volatility Analysis",
        title_font=dict(size=16, color="#FFFFFF"),
        height=250,
        plot_bgcolor="#000000",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#FFFFFF"),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(showgrid=True, gridcolor="#2d2d2d"),
        yaxis=dict(
            showgrid=True,
            gridcolor="#2d2d2d",
            title="Volatility (%)",
            title_font=dict(size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def create_returns_distribution(df: pd.DataFrame) -> go.Figure:
    """Create returns distribution histogram."""
    returns = df["ret_1"].dropna() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns",
            marker=dict(
                color=returns,
                colorscale=[[0, "#FF3B69"], [0.5, "#999999"], [1, "#00C805"]],
                line=dict(color="#FFFFFF", width=0.5),
            ),
        )
    )

    fig.update_layout(
        title="Returns Distribution",
        title_font=dict(size=16, color="#FFFFFF"),
        height=250,
        plot_bgcolor="#000000",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#FFFFFF"),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(
            showgrid=True,
            gridcolor="#2d2d2d",
            title="Return (%)",
            title_font=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#2d2d2d",
            title="Frequency",
            title_font=dict(size=12),
        ),
        showlegend=False,
    )

    return fig


def create_prediction_gauge(probability: float, signal: str) -> go.Figure:
    """Create probability gauge with Robinhood styling."""
    if signal == "LONG":
        bar_color = "#00C805"
    elif signal == "SHORT":
        bar_color = "#FF3B69"
    else:
        bar_color = "#999999"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "%", "font": {"size": 48, "color": "#FFFFFF"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 2,
                    "tickcolor": "#666666",
                    "tickfont": {"color": "#FFFFFF"},
                },
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "#1a1a1a",
                "borderwidth": 3,
                "bordercolor": "#333333",
                "steps": [
                    {"range": [0, 45], "color": "rgba(255, 59, 105, 0.15)"},
                    {"range": [45, 55], "color": "rgba(150, 150, 150, 0.15)"},
                    {"range": [55, 100], "color": "rgba(0, 200, 5, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#1a1a1a",
        font={"color": "#FFFFFF", "family": "Arial"},
    )

    return fig


def calculate_advanced_metrics(
    predictions: List[PredictionResult], df: pd.DataFrame
) -> Dict:
    """Calculate advanced trading metrics."""
    if len(predictions) < 2:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    active_signals = [p for p in predictions if p.signal != "NEUTRAL"]

    if not active_signals:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    # Calculate returns for each signal
    returns = []
    wins = []
    losses = []

    for pred in active_signals[:-1]:
        future_idx = df[df["timestamp"] > pred.timestamp].head(10)
        if len(future_idx) > 0:
            future_price = future_idx["close"].iloc[-1]
            ret = (future_price - pred.price) / pred.price

            if pred.signal == "SHORT":
                ret = -ret

            returns.append(ret)

            if ret > 0:
                wins.append(ret)
            else:
                losses.append(abs(ret))

    if not returns:
        return {
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    # Win rate
    win_rate = (len(wins) / len(returns)) * 100 if returns else 0

    # Sharpe ratio (annualized, assuming 525600 minutes per year)
    returns_arr = np.array(returns)
    sharpe = (
        (np.mean(returns_arr) / (np.std(returns_arr) + 1e-9)) * np.sqrt(525600)
        if len(returns_arr) > 0
        else 0
    )

    # Max drawdown
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0

    # Profit factor
    total_wins = sum(wins) if wins else 0
    total_losses = sum(losses) if losses else 1e-9
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Average win/loss
    avg_win = (np.mean(wins) * 100) if wins else 0
    avg_loss = (np.mean(losses) * 100) if losses else 0

    return {
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    """Main professional dashboard."""

    # Top bar with logo
    st.markdown("# 📈 **Crypto Pro**")
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

    # Load config
    cfg = load_config("configs/live.yaml")

    # Settings panel in sidebar
    with st.sidebar:
        st.markdown("## ⚙️ **Trading Settings**")

        # Symbol
        symbol = st.text_input("Symbol", value=cfg.get("symbol", "btcusdt"))

        st.markdown("---")

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
                long_th = 0.55
                short_th = 0.45
                st.info("🛡️ Conservative: Only high-confidence signals")
            elif "Balanced" in preset:
                long_th = 0.52
                short_th = 0.48
                st.info("⚖️ Balanced: Medium signal frequency")
            else:  # Aggressive
                long_th = 0.505
                short_th = 0.495
                st.success("⚡ Aggressive: Maximum signals (testing)")
        else:  # Custom
            col1, col2 = st.columns(2)
            with col1:
                long_th = st.number_input(
                    "LONG Threshold",
                    min_value=0.50,
                    max_value=0.99,
                    value=0.52,
                    step=0.01,
                    help="Probability needed to trigger LONG signal",
                )
            with col2:
                short_th = st.number_input(
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
        risk = risk_pct / 100

        st.caption(f"Base position size: ${capital * risk:,.2f}")
        st.caption(
            f"Range: ${capital * risk * 0.5:,.2f} - ${capital * risk * 1.5:,.2f}"
        )

        st.markdown("---")

        # Chart settings
        chart_bars = st.slider("Chart Bars", 50, 500, 100, 10)

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 8])

    with col1:
        if st.button("▶️ START", use_container_width=True, type="primary"):
            if not st.session_state.is_running:
                try:
                    st.session_state.engine = InferenceEngine(
                        model_path=cfg["model_checkpoint"],
                        scaler_path=cfg["scaler_path"],
                        meta_path=cfg["meta_path"],
                        device=cfg.get("device", "cpu"),
                        long_threshold=long_th,
                        short_threshold=short_th,
                        capital=capital,
                        risk_per_trade=risk,
                    )

                    # Capture engine and predictions in closure to avoid threading issues
                    engine = st.session_state.engine
                    predictions = st.session_state.predictions

                    def on_new_bar(df: pd.DataFrame, bar: Dict):
                        result = engine.predict(
                            df, current_price=bar["close"], timestamp=bar["timestamp"]
                        )
                        predictions.append(result)

                    st.session_state.stream = BinanceKlineStream(
                        symbol=symbol,
                        interval=cfg.get("interval", "1m"),
                        window_size=cfg.get("stream", {}).get("window_size", 200),
                        on_bar_callback=on_new_bar,
                    )

                    st.session_state.stream.start()
                    st.session_state.is_running = True
                    st.success("Started!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("⏹️ STOP", use_container_width=True):
            if st.session_state.is_running and st.session_state.stream:
                st.session_state.stream.stop()
                st.session_state.is_running = False
                st.info("Stopped")

    with col3:
        if st.session_state.is_running:
            st.success("🟢 LIVE")
        else:
            st.warning("🔴 OFFLINE")

    st.markdown("---")

    # Main content
    if not st.session_state.is_running:
        st.info("👆 Click START to begin streaming live data")
        st.markdown(
            """
        ### Professional Trading Dashboard
        - 📊 TradingView-style charts with RSI, SMA, Volume
        - 🎯 ML-powered signals with quantitative analysis
        - 📈 Real-time volatility and returns analysis
        - 💰 Advanced risk metrics (Sharpe, Drawdown, Profit Factor)
        """
        )
        return

    # Wait for data
    if not st.session_state.stream or len(st.session_state.stream.bars) < 60:
        st.info(
            f"⏳ Collecting data... ({len(st.session_state.stream.bars) if st.session_state.stream else 0}/60 bars)"
        )
        time.sleep(2)
        st.rerun()
        return

    # Get data
    df = st.session_state.stream.get_dataframe()
    predictions = list(st.session_state.predictions)
    current_pred = predictions[-1] if predictions else None

    # Top metrics row (Robinhood style)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if current_pred:
            price_change = df["ret_1"].iloc[-1] * 100
            st.metric(
                label=f"{symbol.upper()}/USDT",
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
            st.caption(f"{current_pred.position_size_pct:.2f}% of capital")

    with col4:
        if df is not None and len(df) > 0:
            vol_24h = df["volume"].tail(1440).sum()
            st.metric("24h Volume", f"${vol_24h:,.0f}")
            st.caption(f"Avg: ${vol_24h/1440:,.0f}/min")

    with col5:
        metrics = calculate_advanced_metrics(predictions, df)
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        st.caption(f"Sharpe: {metrics['sharpe']:.2f}")

    st.markdown("---")

    # Main chart row
    chart_df = df.tail(chart_bars)
    main_fig = create_main_chart(chart_df, predictions[-50:] if predictions else [])
    st.plotly_chart(main_fig, use_container_width=True)

    st.markdown("---")

    # Secondary charts row
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if current_pred:
            gauge = create_prediction_gauge(
                current_pred.probability, current_pred.signal
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(f"**Model Probability**")

    with col2:
        vol_fig = create_volatility_chart(chart_df)
        st.plotly_chart(vol_fig, use_container_width=True)

    with col3:
        ret_fig = create_returns_distribution(chart_df)
        st.plotly_chart(ret_fig, use_container_width=True)

    st.markdown("---")

    # Advanced metrics
    st.markdown("### 📊 **Quantitative Metrics**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    metrics = calculate_advanced_metrics(predictions, df)

    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

    with col2:
        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")

    with col4:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

    with col5:
        st.metric("Avg Win", f"{metrics['avg_win']:.2f}%")

    with col6:
        st.metric("Avg Loss", f"{metrics['avg_loss']:.2f}%")

    st.markdown("---")

    # Signal history
    st.markdown("### 📝 **Recent Signals**")

    if predictions:
        history_data = []
        for pred in reversed(predictions[-30:]):
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
            st.info("No active signals yet")

    # Auto-refresh
    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    main()
