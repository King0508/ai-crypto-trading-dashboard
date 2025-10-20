"""
Download historical kline data from Binance API and assemble into training-ready parquet.
Supports downloading last N days of 1-minute bars for any symbol.
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import List
import time

import pandas as pd
import requests
from tqdm import tqdm


def download_klines(
    symbol: str, interval: str, start_time: int, end_time: int
) -> List[List]:
    """
    Download klines from Binance API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m', '5m', '1h')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds

    Returns:
        List of klines, each kline is a list with OHLCV data
    """
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000,  # Max allowed by Binance
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


def download_range(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Download klines for the last N days in chunks.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m')
        days: Number of days to download

    Returns:
        DataFrame with all klines
    """
    # Calculate time range
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Downloading {symbol} {interval} from {start_dt} to {end_dt}")

    # Binance column names
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "n_trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]

    all_klines = []
    current_start = start_ms

    # Calculate approximate number of chunks needed
    # 1m interval: 1440 bars per day, 1000 bars per request = ~1.4 requests per day
    expected_chunks = int((days * 1440) / 1000) + 1

    with tqdm(total=expected_chunks, desc="Downloading") as pbar:
        while current_start < end_ms:
            try:
                klines = download_klines(symbol, interval, current_start, end_ms)

                if not klines:
                    break

                all_klines.extend(klines)

                # Move to next chunk (use last bar's close time + 1ms)
                current_start = klines[-1][6] + 1

                pbar.update(1)

                # Rate limiting: Binance allows ~1200 requests per minute
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"\nError downloading: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)

    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=columns)

    # Convert numeric columns
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_base",
        "taker_quote",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce")

    # Sort by time and remove duplicates
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"])
    df = df.reset_index(drop=True)

    print(f"Downloaded {len(df)} bars")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features to raw kline data (same as assemble_binance_bars.py).

    Args:
        df: Raw kline DataFrame

    Returns:
        DataFrame with features added
    """
    # Normalize timestamp -> pandas datetime
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Core features from OHLCV
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["ret_5"] = df["close"].pct_change(5).fillna(0.0)
    df["roll_vol_20"] = df["ret_1"].rolling(20).std().fillna(0.0)
    df["roll_vol_60"] = df["ret_1"].rolling(60).std().fillna(0.0)
    df["dollar_vol"] = df["close"] * df["volume"]  # proxy for activity
    df["tb_ratio"] = (df["taker_quote"] / (df["quote_volume"] + 1e-9)).fillna(0.0)

    # Use 'close' as our mid proxy for bar data
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


def make_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Create binary labels for prediction.

    Args:
        df: DataFrame with features
        horizon: Number of bars ahead to predict

    Returns:
        DataFrame with labels added
    """
    # Binary label: will close(t+h) > close(t)?
    fwd = df["mid"].shift(-horizon)
    y = (fwd > df["mid"]).astype(float)
    df["label"] = y
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download historical Binance klines and prepare for training"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--interval", type=str, default="1m", help="Kline interval (default: 1m)"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Number of days to download (default: 90)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/btcusdt_1m.parquet",
        help="Output parquet file path (default: data/btcusdt_1m.parquet)",
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="Label horizon in bars (default: 10)"
    )

    args = parser.parse_args()

    # Download data
    df = download_range(args.symbol, args.interval, args.days)

    # Add features
    print("Adding features...")
    df = add_features(df)

    # Add labels
    print(f"Creating labels with horizon={args.horizon}...")
    df = make_labels(df, horizon=args.horizon)

    # Remove rows with NaN labels (last 'horizon' rows)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    # Save to parquet
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)

    print(f"\n✓ Saved {len(df)} bars to {args.out}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(
        f"  Features: {', '.join([c for c in df.columns if c not in ['timestamp', 'label']])}"
    )


if __name__ == "__main__":
    main()
