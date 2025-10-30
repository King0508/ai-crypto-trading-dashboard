"""
Quick test to verify Alpaca connection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from exchange import AlpacaExchange
from database import init_db
from dotenv import load_dotenv
import os

load_dotenv()

print("\n" + "=" * 60)
print("TESTING ALPACA CONNECTION")
print("=" * 60 + "\n")

# Check credentials
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or api_key == "your_api_key_here":
    print("❌ ERROR: Alpaca API key not set!")
    print("\nPlease edit .env file and add your Alpaca credentials:")
    print("  ALPACA_API_KEY=your_actual_key")
    print("  ALPACA_SECRET_KEY=your_actual_secret")
    print("\nGet keys from: https://app.alpaca.markets/paper/dashboard/overview")
    sys.exit(1)

print(f"✓ API Key found: {api_key[:10]}...")

# Test connection
try:
    exchange = AlpacaExchange(paper=True)
    print("✓ Connected to Alpaca")

    account = exchange.get_account()
    print(f"\nAccount Info:")
    print(f"  Equity: ${float(account.equity):,.2f}")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")

    price = exchange.get_latest_price("BTC/USD")
    print(f"\nCurrent BTC Price: ${price:,.2f}")

    position = exchange.get_position("BTC/USD")
    if position:
        print(f"\nCurrent Position:")
        print(f"  Side: {position.side}")
        print(f"  Quantity: {float(position.qty):.6f} BTC")
        print(f"  Unrealized P/L: ${float(position.unrealized_pl):,.2f}")
    else:
        print(f"\nNo current position")

    print("\n" + "=" * 60)
    print("✅ CONNECTION SUCCESSFUL!")
    print("=" * 60)
    print("\nYou're ready to start trading!")
    print("\nNext steps:")
    print("  1. Make sure your model is trained:")
    print("     python src/train.py --config configs/default.yaml")
    print()
    print("  2. Start the trading bot:")
    print("     python run_bot.py")
    print()
    print("  3. Open dashboard (in separate terminal):")
    print("     python run_dashboard.py")
    print()
    print("  Or use the batch file:")
    print("     start_trading.bat")
    print()

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPlease check:")
    print("  - Your API keys are correct")
    print("  - You're using paper trading keys")
    print("  - Alpaca service is online: https://status.alpaca.markets")
    sys.exit(1)

# Test database
print("Testing database...")
init_db()
print("✓ Database initialized")

print("\n" + "=" * 60)
