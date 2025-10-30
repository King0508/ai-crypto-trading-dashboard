"""Database models."""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Signal(Base):
    """Trading signals."""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, default="BTCUSD")
    signal = Column(String)  # LONG, SHORT, NEUTRAL
    probability = Column(Float)
    confidence = Column(Float)
    price = Column(Float)
    executed = Column(Boolean, default=False)


class Trade(Base):
    """Executed trades."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, default="BTCUSD")
    side = Column(String)  # BUY or SELL
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    status = Column(String, default="open")  # open, closed


class Performance(Base):
    """Performance metrics."""
    __tablename__ = "performance"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    equity = Column(Float)
    cash = Column(Float)
    total_pnl = Column(Float, default=0)
    win_rate = Column(Float, default=0)
    total_trades = Column(Integer, default=0)

