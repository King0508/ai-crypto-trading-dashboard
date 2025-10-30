"""
Main trading bot that connects TCN model to Alpaca.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from exchange import AlpacaExchange
from database import init_db, get_session, Signal, Trade, Performance
from inference import InferenceEngine
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Automated trading bot using TCN model."""
    
    def __init__(
        self,
        capital: float = 100000,
        risk_per_trade: float = 0.02,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        symbol: str = "BTC/USD",
    ):
        """Initialize trading bot."""
        load_dotenv()
        
        self.capital = capital
        self.initial_capital = capital
        self.risk_per_trade = risk_per_trade
        self.symbol = symbol
        
        # Initialize components
        logger.info("Initializing trading bot...")
        
        # Exchange
        self.exchange = AlpacaExchange(paper=True)
        logger.info("✓ Connected to Alpaca")
        
        # Database
        init_db()
        logger.info("✓ Database initialized")
        
        # Model
        self.model = InferenceEngine(
            model_path="artifacts/model.pt",
            scaler_path="artifacts/scaler.pkl",
            meta_path="artifacts/meta.json",
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            capital=capital,
            risk_per_trade=risk_per_trade,
        )
        logger.info("✓ Model loaded")
        
        # State
        self.current_position = None
        self.running = False
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"Bot ready! Capital: ${capital:,.2f}")
        
        # Buy initial BTC position so we can trade both directions
        self._initialize_position()
    
    def _initialize_position(self):
        """Buy initial BTC position so we can sell on SHORT signals."""
        try:
            price = self.exchange.get_latest_price(self.symbol)
            position = self.exchange.get_position(self.symbol)
            
            if position:
                logger.info(f"Already have BTC position: {float(position.qty):.6f} BTC")
                return
            
            # Buy small initial position (0.01 BTC = ~$1,100)
            initial_qty = 0.01
            
            logger.info(f"Buying initial position: {initial_qty} BTC @ ${price:,.2f}")
            order = self.exchange.place_market_order(
                symbol=self.symbol,
                side="BUY",
                qty=initial_qty
            )
            logger.info(f"✓ Initial position bought! Can now trade both LONG and SHORT")
            
        except Exception as e:
            logger.error(f"Failed to buy initial position: {e}")
            logger.info("Will only trade LONG signals (buying new BTC)")
    
    def get_market_data(self, lookback_hours: int = 2) -> pd.DataFrame:
        """Get recent market data with features."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        df = self.exchange.get_historical_bars(
            symbol=self.symbol,
            start=start_time,
            end=end_time,
            limit=200
        )
        
        if df.empty:
            return df
        
        # Add features (matching training data)
        df = df.rename(columns={'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'})
        
        # Returns
        df["ret_1"] = df["close"].pct_change().fillna(0)
        df["ret_5"] = df["close"].pct_change(5).fillna(0)
        
        # Volatility
        df["roll_vol_20"] = df["ret_1"].rolling(20, min_periods=1).std().fillna(0)
        df["roll_vol_60"] = df["ret_1"].rolling(60, min_periods=1).std().fillna(0)
        
        # Volume features
        df["dollar_vol"] = df["close"] * df["volume"]
        df["tb_ratio"] = 0.5  # Placeholder (Alpaca doesn't provide taker data)
        df["mid"] = df["close"]
        
        return df
    
    def calculate_position_size(self, signal: str, confidence: float, price: float) -> float:
        """Calculate position size."""
        if signal == "NEUTRAL":
            return 0.0
        
        # Base size from capital and risk
        base_size_usd = self.capital * self.risk_per_trade
        
        # Scale by confidence
        size_usd = base_size_usd * (0.5 + confidence)
        
        # Convert to BTC quantity
        quantity = size_usd / price
        
        # Min/max limits
        quantity = max(0.001, min(quantity, 0.1))  # Between 0.001 and 0.1 BTC
        
        return quantity
    
    def should_enter_trade(self, signal: str) -> bool:
        """Check if we should enter a new trade."""
        # Don't trade if neutral
        if signal == "NEUTRAL":
            return False
        
        # Don't trade if we already have a position
        if self.current_position is not None:
            return False
        
        return True
    
    def should_exit_trade(self, current_price: float) -> tuple[bool, str]:
        """Check if we should exit current position."""
        if self.current_position is None:
            return False, ""
        
        entry_price = self.current_position['entry_price']
        side = self.current_position['side']
        
        # AGGRESSIVE: Tight stop loss (0.5%) and take profit (1%)
        if side == "BUY":
            # Long position
            stop_loss = entry_price * 0.995   # 0.5% stop (tight!)
            take_profit = entry_price * 1.01  # 1% target (quick profit)
            
            if current_price <= stop_loss:
                return True, "stop_loss"
            if current_price >= take_profit:
                return True, "take_profit"
        else:
            # Short position
            stop_loss = entry_price * 1.005   # 0.5% stop
            take_profit = entry_price * 0.99  # 1% target
            
            if current_price >= stop_loss:
                return True, "stop_loss"
            if current_price <= take_profit:
                return True, "take_profit"
        
        return False, ""
    
    def enter_trade(self, signal: str, price: float, confidence: float):
        """Enter a new trade."""
        side = "BUY" if signal == "LONG" else "SELL"
        quantity = self.calculate_position_size(signal, confidence, price)
        
        if quantity < 0.001:
            logger.warning("Position size too small, skipping trade")
            return
        
        try:
            # Place order
            order = self.exchange.place_market_order(
                symbol=self.symbol,
                side=side,
                qty=quantity
            )
            
            # Record position
            self.current_position = {
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'entry_time': datetime.now(),
                'order_id': order.id,
                'signal': signal,
                'confidence': confidence,
            }
            
            # Save to database
            db = get_session()
            trade = Trade(
                symbol=self.symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                entry_time=datetime.now(),
                status="open"
            )
            db.add(trade)
            db.commit()
            db.close()
            
            logger.info(
                f"✓ ENTERED {side}: {quantity:.6f} BTC @ ${price:,.2f} "
                f"(confidence: {confidence:.2%})"
            )
            
        except Exception as e:
            logger.error(f"Failed to enter trade: {e}")
    
    def exit_trade(self, price: float, reason: str):
        """Exit current trade."""
        if self.current_position is None:
            return
        
        side = "SELL" if self.current_position['side'] == "BUY" else "BUY"
        quantity = self.current_position['quantity']
        
        try:
            # Place exit order
            order = self.exchange.place_market_order(
                symbol=self.symbol,
                side=side,
                qty=quantity
            )
            
            # Calculate P/L
            entry_price = self.current_position['entry_price']
            if self.current_position['side'] == "BUY":
                pnl = (price - entry_price) * quantity
            else:
                pnl = (entry_price - price) * quantity
            
            pnl_pct = (pnl / (entry_price * quantity)) * 100
            
            # Update capital
            self.capital += pnl
            
            # Update stats
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            
            # Save to database
            db = get_session()
            trade = db.query(Trade).filter_by(status="open").first()
            if trade:
                trade.exit_price = price
                trade.exit_time = datetime.now()
                trade.pnl = pnl
                trade.pnl_pct = pnl_pct
                trade.status = "closed"
                db.commit()
            db.close()
            
            logger.info(
                f"✓ EXITED {side}: ${pnl:,.2f} ({pnl_pct:+.2f}%) | "
                f"Reason: {reason} | Capital: ${self.capital:,.2f}"
            )
            
            # Clear position
            self.current_position = None
            
        except Exception as e:
            logger.error(f"Failed to exit trade: {e}")
    
    def run_once(self):
        """Run one trading iteration."""
        try:
            # Get market data
            df = self.get_market_data()
            
            if len(df) < 60:
                logger.warning("Insufficient data, waiting...")
                return
            
            # Get current price
            current_price = self.exchange.get_latest_price(self.symbol)
            
            # Make prediction
            prediction = self.model.predict(df, current_price=current_price)
            
            # Log signal
            logger.info(
                f"Price: ${current_price:,.2f} | Signal: {prediction.signal} | "
                f"Prob: {prediction.probability:.3f} | Conf: {prediction.confidence:.2%}"
            )
            
            # Save signal to database
            db = get_session()
            signal = Signal(
                symbol=self.symbol,
                signal=prediction.signal,
                probability=prediction.probability,
                confidence=prediction.confidence,
                price=current_price,
            )
            db.add(signal)
            db.commit()
            db.close()
            
            # Check for exit
            if self.current_position:
                should_exit, exit_reason = self.should_exit_trade(current_price)
                if should_exit:
                    self.exit_trade(current_price, exit_reason)
                    return
            
            # Check for entry
            if self.should_enter_trade(prediction.signal):
                self.enter_trade(
                    signal=prediction.signal,
                    price=current_price,
                    confidence=prediction.confidence,
                )
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
    
    def run(self, interval: int = 60):
        """Run trading bot continuously."""
        logger.info("=" * 60)
        logger.info("STARTING TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Capital: ${self.capital:,.2f}")
        logger.info(f"Risk per trade: {self.risk_per_trade*100}%")
        logger.info(f"Update interval: {interval}s")
        logger.info("=" * 60)
        
        self.running = True
        
        try:
            while self.running:
                self.run_once()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\n\nStopping bot...")
            self.stop()
    
    def stop(self):
        """Stop trading bot."""
        self.running = False
        
        # Close any open positions
        if self.current_position:
            price = self.exchange.get_latest_price(self.symbol)
            self.exit_trade(price, "bot_stopped")
        
        # Print summary
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_pnl = self.capital - self.initial_capital
        pnl_pct = (total_pnl / self.initial_capital) * 100
        
        logger.info("=" * 60)
        logger.info("TRADING BOT STOPPED")
        logger.info("=" * 60)
        logger.info(f"Total trades: {self.total_trades}")
        logger.info(f"Winning trades: {self.winning_trades}")
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Total P/L: ${total_pnl:,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Final capital: ${self.capital:,.2f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    bot = TradingBot(
        capital=100000,
        risk_per_trade=0.02,
        long_threshold=0.55,
        short_threshold=0.45,
    )
    
    bot.run(interval=60)  # Run every 60 seconds

