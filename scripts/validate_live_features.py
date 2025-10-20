"""
Validate that live feature computation matches training feature computation.
This is critical to ensure model predictions are accurate in production.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import argparse
import pandas as pd
import numpy as np

from live_stream import BinanceKlineStream


def load_historical_bars(parquet_path: str, n_bars: int = 100) -> pd.DataFrame:
    """
    Load historical bars from parquet file.

    Args:
        parquet_path: Path to parquet file with historical data
        n_bars: Number of bars to load

    Returns:
        DataFrame with bars
    """
    df = pd.read_parquet(parquet_path)

    # Select last n_bars
    df = df.tail(n_bars).copy()

    return df


def convert_to_stream_format(df: pd.DataFrame) -> list:
    """
    Convert historical DataFrame to the format used by BinanceKlineStream.

    Args:
        df: Historical DataFrame with features

    Returns:
        List of bar dictionaries
    """
    bars = []

    for idx, row in df.iterrows():
        bar = {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "quote_volume": float(
                row.get("quote_volume", row["volume"] * row["close"])
            ),
            "n_trades": int(row.get("n_trades", 1000)),
            "taker_base": float(row.get("taker_base", row["volume"] * 0.5)),
            "taker_quote": float(
                row.get("taker_quote", row["volume"] * row["close"] * 0.5)
            ),
            "is_closed": True,
        }
        bars.append(bar)

    return bars


def compare_features(
    historical_df: pd.DataFrame, stream_df: pd.DataFrame, tolerance: float = 1e-5
) -> dict:
    """
    Compare features between historical and stream computation.

    Args:
        historical_df: Features from historical parquet
        stream_df: Features recomputed by stream
        tolerance: Maximum acceptable difference

    Returns:
        Dictionary with comparison results
    """
    # Feature columns to compare
    feature_cols = [
        "ret_1",
        "ret_5",
        "roll_vol_20",
        "roll_vol_60",
        "dollar_vol",
        "tb_ratio",
    ]

    results = {
        "matches": [],
        "mismatches": [],
        "errors": [],
    }

    # Align dataframes by timestamp
    merged = pd.merge(
        historical_df,
        stream_df,
        on="timestamp",
        suffixes=("_hist", "_stream"),
    )

    if len(merged) == 0:
        results["errors"].append("No matching timestamps found")
        return results

    # Compare each feature
    for col in feature_cols:
        hist_col = f"{col}_hist"
        stream_col = f"{col}_stream"

        if hist_col not in merged.columns or stream_col not in merged.columns:
            results["errors"].append(
                f"Column {col} not found in one or both DataFrames"
            )
            continue

        # Compute difference
        diff = np.abs(merged[hist_col] - merged[stream_col])
        max_diff = diff.max()
        mean_diff = diff.mean()

        match_pct = (diff <= tolerance).sum() / len(diff) * 100

        comparison = {
            "feature": col,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "match_pct": float(match_pct),
        }

        if (
            match_pct >= 99.0
        ):  # Allow 1% tolerance for edge cases (rolling windows at start)
            results["matches"].append(comparison)
        else:
            results["mismatches"].append(comparison)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate that live feature computation matches training"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/btcusdt_1m.parquet",
        help="Path to historical parquet file",
    )
    parser.add_argument(
        "--n-bars", type=int, default=100, help="Number of bars to compare"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Maximum acceptable difference"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Feature Validation: Historical vs Live Stream")
    print("=" * 60)

    # Load historical data
    print(f"\n1. Loading historical data from {args.data}...")
    historical_df = load_historical_bars(args.data, args.n_bars)
    print(f"   ✓ Loaded {len(historical_df)} bars")
    print(
        f"   Time range: {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}"
    )

    # Convert to stream format
    print(f"\n2. Converting to stream format...")
    bars = convert_to_stream_format(historical_df)
    print(f"   ✓ Converted {len(bars)} bars")

    # Compute features using stream logic
    print(f"\n3. Recomputing features using live stream logic...")
    stream = BinanceKlineStream()
    stream_df = stream.compute_features(bars)
    print(f"   ✓ Computed features for {len(stream_df)} bars")

    # Compare
    print(f"\n4. Comparing features...")
    results = compare_features(historical_df, stream_df, args.tolerance)

    # Display results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print("=" * 60)

    if results["errors"]:
        print("\n❌ ERRORS:")
        for error in results["errors"]:
            print(f"   - {error}")

    if results["matches"]:
        print(f"\n✓ MATCHING FEATURES ({len(results['matches'])}):")
        for match in results["matches"]:
            print(
                f"   {match['feature']:15} | max_diff: {match['max_diff']:.2e} | mean_diff: {match['mean_diff']:.2e} | match: {match['match_pct']:.1f}%"
            )

    if results["mismatches"]:
        print(f"\n❌ MISMATCHING FEATURES ({len(results['mismatches'])}):")
        for mismatch in results["mismatches"]:
            print(
                f"   {mismatch['feature']:15} | max_diff: {mismatch['max_diff']:.2e} | mean_diff: {mismatch['mean_diff']:.2e} | match: {mismatch['match_pct']:.1f}%"
            )

    # Overall result
    print(f"\n{'='*60}")
    if results["mismatches"] or results["errors"]:
        print("❌ VALIDATION FAILED")
        print("   Live features DO NOT match training features!")
        print("   Fix mismatches before using in production.")
        return 1
    else:
        print("✓ VALIDATION PASSED")
        print("   Live features match training features!")
        print("   Safe to use for real-time predictions.")
        return 0


if __name__ == "__main__":
    exit(main())
