"""
Live trading dashboard with Streamlit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from database import get_session, Signal, Trade
from exchange import AlpacaExchange
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Live Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

# Initialize exchange
@st.cache_resource
def get_exchange():
    return AlpacaExchange(paper=True)

exchange = get_exchange()

# Title
st.title("🤖 Live Trading Dashboard")

# Metrics at top
col1, col2, col3, col4, col5 = st.columns(5)

try:
    account = exchange.get_account()
    position = exchange.get_position("BTC/USD")
    price = exchange.get_latest_price("BTC/USD")
    
    with col1:
        st.metric("Account Equity", f"${float(account.equity):,.2f}")
    
    with col2:
        st.metric("Cash", f"${float(account.cash):,.2f}")
    
    with col3:
        st.metric("BTC Price", f"${price:,.2f}")
    
    with col4:
        if position:
            st.metric("Position", f"{float(position.qty):.6f} BTC")
        else:
            st.metric("Position", "None")
    
    with col5:
        if position:
            pnl = float(position.unrealized_pl)
            st.metric("Unrealized P/L", f"${pnl:,.2f}", delta=f"{(pnl/float(position.cost_basis)*100):.2f}%")
        else:
            st.metric("Unrealized P/L", "$0.00")

except Exception as e:
    st.error(f"Error connecting to Alpaca: {e}")
    st.stop()

st.divider()

# Two columns: Chart and trades
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📊 Recent Signals")
    
    # Get signals from database
    db = get_session()
    signals = db.query(Signal).order_by(Signal.timestamp.desc()).limit(100).all()
    db.close()
    
    if signals:
        df_signals = pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "signal": s.signal,
                "probability": s.probability,
                "confidence": s.confidence,
                "price": s.price,
            }
            for s in signals
        ])
        
        # Chart
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df_signals['timestamp'],
            y=df_signals['price'],
            mode='lines',
            name='BTC Price',
            line=dict(color='white', width=2)
        ))
        
        # Long signals
        longs = df_signals[df_signals['signal'] == 'LONG']
        if len(longs) > 0:
            fig.add_trace(go.Scatter(
                x=longs['timestamp'],
                y=longs['price'],
                mode='markers',
                name='LONG',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        # Short signals
        shorts = df_signals[df_signals['signal'] == 'SHORT']
        if len(shorts) > 0:
            fig.add_trace(go.Scatter(
                x=shorts['timestamp'],
                y=shorts['price'],
                mode='markers',
                name='SHORT',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            showlegend=True,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent signals table
        st.dataframe(
            df_signals[['timestamp', 'signal', 'probability', 'confidence', 'price']].head(20),
            use_container_width=True
        )
    else:
        st.info("No signals yet. Bot needs to be running.")

with right_col:
    st.subheader("💰 Trade History")
    
    # Get trades
    db = get_session()
    trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(20).all()
    db.close()
    
    if trades:
        df_trades = pd.DataFrame([
            {
                "Time": t.timestamp.strftime("%H:%M:%S"),
                "Side": t.side,
                "Qty": f"{t.quantity:.6f}",
                "Entry": f"${t.entry_price:,.2f}",
                "Exit": f"${t.exit_price:,.2f}" if t.exit_price else "-",
                "P/L": f"${t.pnl:,.2f}" if t.pnl else "-",
                "Status": t.status,
            }
            for t in trades
        ])
        
        st.dataframe(df_trades, use_container_width=True, height=400)
        
        # Performance stats
        closed_trades = [t for t in trades if t.status == "closed"]
        if closed_trades:
            winning = [t for t in closed_trades if t.pnl and t.pnl > 0]
            total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
            win_rate = len(winning) / len(closed_trades) * 100
            
            st.divider()
            st.subheader("📈 Performance")
            
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.metric("Total Trades", len(closed_trades))
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with perf_col2:
                st.metric("Winning", len(winning))
                st.metric("Total P/L", f"${total_pnl:,.2f}")
    else:
        st.info("No trades yet.")

# Auto-refresh
st.divider()
auto_refresh = st.checkbox("Auto-refresh (every 10s)", value=True)

if auto_refresh:
    time.sleep(10)
    st.rerun()

