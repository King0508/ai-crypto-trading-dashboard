"""Alpaca exchange client for trading."""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import CryptoDataStream

logger = logging.getLogger(__name__)


class AlpacaExchange:
    """Alpaca exchange for crypto trading."""

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.data_client = CryptoHistoricalDataClient()
        self.stream = None

    def get_account(self):
        """Get account info."""
        return self.trading_client.get_account()

    def get_balance(self) -> float:
        """Get account equity."""
        account = self.get_account()
        return float(account.equity)

    def get_position(self, symbol: str = "BTC/USD"):
        """Get current position."""
        try:
            # Alpaca crypto uses format with slash
            alpaca_symbol = symbol.replace("/", "")
            return self.trading_client.get_open_position(alpaca_symbol)
        except:
            return None

    def place_market_order(
        self, symbol: str, side: str, qty: float, client_order_id: str = None
    ):
        """Place market order."""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        # Convert BTC/USD to BTCUSD for Alpaca
        alpaca_symbol = symbol.replace("/", "")

        request = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )

        order = self.trading_client.submit_order(request)
        logger.info(f"Placed {side} order: {qty} {symbol} | Order ID: {order.id}")
        return order

    def place_limit_order(self, symbol: str, side: str, qty: float, limit_price: float):
        """Place limit order."""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            limit_price=limit_price,
        )

        return self.trading_client.submit_order(request)

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.trading_client.cancel_orders()

    def close_all_positions(self):
        """Close all positions."""
        self.trading_client.close_all_positions(cancel_orders=True)

    def get_latest_price(self, symbol: str = "BTC/USD") -> float:
        """Get latest price."""
        # Alpaca crypto data uses format with slash
        request = CryptoBarsRequest(
            symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute, limit=1
        )
        bars = self.data_client.get_crypto_bars(request)
        return float(bars[symbol][-1].close)

    def get_historical_bars(
        self, symbol: str, start: datetime, end: datetime = None, limit: int = None
    ) -> pd.DataFrame:
        """Get historical bars."""
        # Alpaca crypto data uses format with slash
        request = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end or datetime.now(),
            limit=limit,
        )

        bars = self.data_client.get_crypto_bars(request)
        df = bars.df.reset_index()
        return df
