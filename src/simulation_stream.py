"""
Simulation streaming - Replays historical data at high speed.
Perfect for testing, backtesting, and demo purposes.
"""
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Callable, Optional, Dict, List
import logging

import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SimulationStream:
    """
    Simulates real-time data streaming by replaying historical data.
    Much faster than waiting for live bars - perfect for testing!
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 200,
        speed_multiplier: float = 1.0,
        on_bar_callback: Optional[Callable] = None,
    ):
        """
        Initialize simulation stream.
        
        Args:
            data_path: Path to parquet file with historical data
            window_size: Number of bars to keep in rolling window
            speed_multiplier: Speed multiplier (1.0 = real-time, 10.0 = 10x speed)
            on_bar_callback: Function to call when a new bar is processed
        """
        self.data_path = data_path
        self.window_size = window_size
        self.speed_multiplier = speed_multiplier
        self.on_bar_callback = on_bar_callback
        
        # Load historical data
        logger.info(f"Loading historical data from {data_path}...")
        self.full_data = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.full_data)} bars")
        
        # Data storage (list of dicts for each bar)
        self.bars = deque(maxlen=window_size)
        self.current_bar = None
        
        # Playback control
        self.running = False
        self.paused = False
        self.current_index = 0
        self.replay_thread = None
        
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
    
    def _replay_bars(self):
        """Replay bars from historical data."""
        logger.info(f"Starting simulation at {self.speed_multiplier}x speed")
        
        # Calculate delay between bars
        base_delay = 1.0  # 1 second for 1x speed
        delay = base_delay / self.speed_multiplier
        
        while self.running and self.current_index < len(self.full_data):
            if self.paused:
                time.sleep(0.1)
                continue
            
            # Get next bar
            row = self.full_data.iloc[self.current_index]
            
            # Convert to bar format
            bar_data = {
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "quote_volume": float(row.get("volume", 0) * row["close"]),
                "n_trades": 1000,  # Placeholder
                "taker_base": float(row.get("volume", 0) * 0.5),
                "taker_quote": float(row.get("volume", 0) * row["close"] * 0.5),
                "is_closed": True,
            }
            
            # Add to window
            self.bars.append(bar_data)
            self.current_index += 1
            
            # Log progress
            if self.current_index % 10 == 0:
                logger.info(
                    f"[Simulation] Bar {self.current_index}/{len(self.full_data)} | "
                    f"{row['timestamp']} | Close: ${row['close']:.2f}"
                )
            
            # Trigger callback if we have enough bars
            if self.on_bar_callback and len(self.bars) >= 60:
                try:
                    df = self.get_dataframe()
                    self.on_bar_callback(df, bar_data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
            
            # Wait before next bar (simulates real-time)
            time.sleep(delay)
        
        logger.info(f"Simulation complete! Processed {self.current_index} bars")
        self.running = False
    
    def start(self):
        """Start the simulation."""
        if self.running:
            logger.warning("Simulation already running")
            return
        
        self.running = True
        self.paused = False
        
        # Start replay thread
        self.replay_thread = threading.Thread(
            target=self._replay_bars,
            daemon=True,
        )
        self.replay_thread.start()
        
        logger.info(f"[OK] Simulation started at {self.speed_multiplier}x speed")
    
    def stop(self):
        """Stop the simulation."""
        logger.info("Stopping simulation...")
        self.running = False
        
        if self.replay_thread and self.replay_thread.is_alive():
            self.replay_thread.join(timeout=5)
        
        logger.info("[OK] Simulation stopped")
    
    def pause(self):
        """Pause the simulation."""
        self.paused = True
        logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation."""
        self.paused = False
        logger.info("Simulation resumed")
    
    def restart(self):
        """Restart simulation from beginning."""
        self.stop()
        time.sleep(0.5)
        self.bars.clear()
        self.current_index = 0
        self.start()
    
    def set_speed(self, multiplier: float):
        """Change playback speed."""
        self.speed_multiplier = max(0.1, min(multiplier, 100.0))
        logger.info(f"Speed changed to {self.speed_multiplier}x")
    
    def skip_to(self, index: int):
        """Skip to a specific bar index."""
        if 0 <= index < len(self.full_data):
            self.current_index = index
            logger.info(f"Skipped to bar {index}")
    
    def wait_for_bars(self, min_bars: int = 60, timeout: float = 300):
        """
        Wait until we have at least min_bars in the window.
        Much faster than live stream!
        
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
            
            if not self.running:
                logger.error("Simulation stopped before reaching minimum bars")
                return False
            
            time.sleep(0.1)
        
        logger.info(f"[OK] Ready with {len(self.bars)} bars")
        return True


def test_simulation():
    """Test the simulation stream."""
    
    def on_new_bar(df: pd.DataFrame, bar: Dict):
        """Callback when a new bar is processed."""
        print(f"\n{'='*60}")
        print(f"Bar processed: {bar['timestamp']}")
        print(f"  Close: ${bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:.4f}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Last few features:")
        print(df[["timestamp", "close", "ret_1", "roll_vol_20"]].tail(3))
    
    # Create simulation stream
    sim = SimulationStream(
        data_path="data/btcusdt_1m.parquet",
        window_size=100,
        speed_multiplier=10.0,  # 10x speed!
        on_bar_callback=on_new_bar,
    )
    
    # Start simulation
    sim.start()
    
    # Wait for initial bars
    logger.info("Waiting for initial bars...")
    sim.wait_for_bars(min_bars=60)
    
    # Let it run for 30 seconds
    logger.info("Simulation is live. Running for 30 seconds...")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    # Stop
    sim.stop()
    
    logger.info(f"Final stats: Processed {sim.current_index} bars")


if __name__ == "__main__":
    test_simulation()

