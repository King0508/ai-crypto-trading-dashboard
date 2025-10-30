"""
Start the trading bot.

Usage:
    python run_bot.py
    
Press Ctrl+C to stop.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading.bot import TradingBot
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    # Get capital from env or default to 100k
    capital = float(os.getenv("INITIAL_CAPITAL", 100000))
    
    print("\n" + "="*60)
    print("TRADING BOT STARTUP")
    print("="*60)
    print(f"Capital: ${capital:,.2f}")
    print(f"Exchange: Alpaca (Paper Trading)")
    print(f"Symbol: BTC/USD")
    print("="*60)
    print("\nStarting in 3 seconds...")
    print("Press Ctrl+C to stop anytime\n")
    
    import time
    time.sleep(3)
    
    bot = TradingBot(
        capital=capital,
        risk_per_trade=0.01,   # 1% risk per trade (smaller positions)
        long_threshold=0.501,  # VERY AGGRESSIVE: Trade at 50.1%+
        short_threshold=0.499, # VERY AGGRESSIVE: Trade at <49.9%
    )
    
    # Run every 10 seconds (VERY ACTIVE)
    bot.run(interval=10)

