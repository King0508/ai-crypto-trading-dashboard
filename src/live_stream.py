"""
Real-time data streaming from Binance WebSocket.
Maintains rolling window of bars and computes features on-the-fly.
"""
from __future__ import annotations

import json
import time
import threading
from collections import deque
from datetime import datetime
from typing import Callable, Optional, Dict, List
import logging

import websocket
import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BinanceKlineStream:
    """
    Streams 1-minute klines from Binance WebSocket and maintains a rolling window.
    Computes features on-the-fly to match training data format.
    """
    
    def __init__(
        self,
        symbol: str = "btcusdt",
        interval: str = "1m",
        window_size: int = 100,
        on_bar_callback: Optional[Callable] = None,
    ):
        """
        Initialize the kline stream.
        
        Args:
            symbol: Trading pair symbol (lowercase, e.g., 'btcusdt')
            interval: Kline interval ('1m', '5m', etc.)
            window_size: Number of bars to keep in rolling window
            on_bar_callback: Function to call when a new bar closes
        """
        self.symbol = symbol.lower()
        self.interval = interval
        self.window_size = window_size
        self.on_bar_callback = on_bar_callback
        
        # WebSocket connection
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"
        self.ws = None
        self.ws_thread = None
        self.running = False
        
        # Data storage (list of dicts for each bar)
        self.bars = deque(maxlen=window_size)
        self.current_bar = None
        
        # Reconnection parameters
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 30  # Max 30 seconds
        self.reconnect_attempts = 0
        
    def compute_features(self, bars: List[Dict]) -> pd.DataFrame:
        """
        Compute features from raw bars (same as training).
        
        Args:
            bars: List of bar dictionaries
        
        Returns:
            DataFrame with features
        """
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        
        # Core features from OHLCV
        df["ret_1"] = df["close"].pct_change().fillna(0.0)
        df["ret_5"] = df["close"].pct_change(5).fillna(0.0)
        df["roll_vol_20"] = df["ret_1"].rolling(20, min_periods=1).std().fillna(0.0)
        df["roll_vol_60"] = df["ret_1"].rolling(60, min_periods=1).std().fillna(0.0)
        df["dollar_vol"] = df["close"] * df["volume"]
        df["tb_ratio"] = (df["taker_quote"] / (df["quote_volume"] + 1e-9)).fillna(0.0)
        df["mid"] = df["close"].astype(float)
        
        # Keep only relevant columns
        keep = [
            "timestamp",
            "mid",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "ret_1",
            "ret_5",
            "roll_vol_20",
            "roll_vol_60",
            "dollar_vol",
            "tb_ratio",
        ]
        
        return df[keep]
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get current bars as a DataFrame with features.
        
        Returns:
            DataFrame with all bars and computed features
        """
        if not self.bars:
            return pd.DataFrame()
        
        return self.compute_features(list(self.bars))
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Binance sends kline updates in this format
            if "k" not in data:
                return
            
            kline = data["k"]
            
            # Extract kline data
            bar_data = {
                "timestamp": pd.to_datetime(kline["t"], unit="ms", utc=True),
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "quote_volume": float(kline["q"]),
                "n_trades": int(kline["n"]),
                "taker_base": float(kline["V"]),
                "taker_quote": float(kline["Q"]),
                "is_closed": kline["x"],  # True if bar is closed
            }
            
            if bar_data["is_closed"]:
                # Bar closed - add to history
                self.bars.append(bar_data)
                self.current_bar = None
                
                # Reset reconnect delay on successful data
                self.reconnect_delay = 1
                self.reconnect_attempts = 0
                
                logger.info(
                    f"[{bar_data['timestamp']}] Close: ${bar_data['close']:.2f} | "
                    f"Volume: {bar_data['volume']:.2f} | Bars in window: {len(self.bars)}"
                )
                
                # Trigger callback if provided
                if self.on_bar_callback and len(self.bars) >= 60:  # Need at least 60 bars
                    try:
                        df = self.get_dataframe()
                        self.on_bar_callback(df, bar_data)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
            else:
                # Bar still forming - update current
                self.current_bar = bar_data
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Attempt reconnection if still supposed to be running
        if self.running:
            self._reconnect()
    
    def on_open(self, ws):
        """Handle WebSocket open."""
        logger.info(f"[OK] Connected to {self.ws_url}")
        self.reconnect_delay = 1
        self.reconnect_attempts = 0
    
    def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if not self.running:
            return
        
        self.reconnect_attempts += 1
        logger.info(
            f"Reconnecting in {self.reconnect_delay}s (attempt {self.reconnect_attempts})..."
        )
        time.sleep(self.reconnect_delay)
        
        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
        if self.running:
            self.start()
    
    def start(self):
        """Start the WebSocket connection in a background thread."""
        self.running = True
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        
        # Run in background thread
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
            daemon=True,
        )
        self.ws_thread.start()
        
        logger.info(f"Starting stream for {self.symbol.upper()} {self.interval}...")
    
    def stop(self):
        """Stop the WebSocket connection."""
        logger.info("Stopping stream...")
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
        
        logger.info("[OK] Stream stopped")
    
    def wait_for_bars(self, min_bars: int = 60, timeout: float = 300):
        """
        Wait until we have at least min_bars in the window.
        
        Args:
            min_bars: Minimum number of bars required
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if requirement met, False if timeout
        """
        start_time = time.time()
        
        while len(self.bars) < min_bars:
            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for {min_bars} bars")
                return False
            
            time.sleep(1)
        
        logger.info(f"[OK] Ready with {len(self.bars)} bars")
        return True


def test_stream():
    """Test the streaming functionality."""
    
    def on_new_bar(df: pd.DataFrame, bar: Dict):
        """Callback when a new bar closes."""
        print(f"\n{'='*60}")
        print(f"New bar closed at {bar['timestamp']}")
        print(f"  Close: ${bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:.4f}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Last few features:")
        print(df[["timestamp", "close", "ret_1", "roll_vol_20"]].tail(3))
    
    # Create stream
    stream = BinanceKlineStream(
        symbol="btcusdt",
        interval="1m",
        window_size=100,
        on_bar_callback=on_new_bar,
    )
    
    # Start streaming
    stream.start()
    
    # Wait for initial bars
    logger.info("Waiting for initial bars (this will take 1-2 minutes)...")
    stream.wait_for_bars(min_bars=60)
    
    # Keep running for 5 minutes
    logger.info("Stream is live. Will run for 5 minutes...")
    try:
        time.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    # Stop
    stream.stop()


if __name__ == "__main__":
    test_stream()

