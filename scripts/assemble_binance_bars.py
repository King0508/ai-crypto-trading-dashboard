# scripts/assemble_binance_bars.py
import argparse, glob, os
import pandas as pd
import numpy as np

# Binance kline columns per docs:
# [0] open time (ms/us), [1] open, [2] high, [3] low, [4] close, [5] volume,
# [6] close time, [7] quote asset volume, [8] number of trades,
# [9] taker buy base asset vol, [10] taker buy quote asset vol, [11] ignore
COLS = [
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


def load_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    parts = []
    for f in files:
        df = pd.read_csv(f, header=None, names=COLS)
        parts.append(df)
    if not parts:
        raise SystemExit(f"No CSV files found in {folder}")
    df = pd.concat(parts, ignore_index=True)
    return df


def add_features(df, freq="1m", ts_unit="ms"):
    # Normalize timestamp -> pandas datetime
    unit = "ms" if ts_unit == "ms" else "us"
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Core features from OHLCV
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["ret_5"] = df["close"].pct_change(5).fillna(0.0)
    df["roll_vol_20"] = df["ret_1"].rolling(20).std().fillna(0.0)
    df["roll_vol_60"] = df["ret_1"].rolling(60).std().fillna(0.0)
    df["dollar_vol"] = df["close"] * df["volume"]  # proxy for activity
    df["tb_ratio"] = (df["taker_quote"] / (df["quote_volume"] + 1e-9)).fillna(
        0.0
    )  # buyer-initiated %

    # Use 'close' as our mid proxy for bar data
    df["mid"] = df["close"].astype(float)

    # Drop obvious non-features
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


def make_labels(df, horizon=10):
    # Binary label: will close(t+h) > close(t)?
    fwd = df["mid"].shift(-horizon)
    y = (fwd > df["mid"]).astype(float)
    df["label"] = y
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="folder with extracted CSVs")
    ap.add_argument("--out", required=True, help="path to write parquet")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument(
        "--ts_unit",
        choices=["ms", "us"],
        default="ms",
        help="Binance spot uses microseconds from 2025-01-01",
    )
    args = ap.parse_args()

    raw = load_folder(args.folder)
    feat = add_features(raw, ts_unit=args.ts_unit)
    feat = make_labels(feat, horizon=args.horizon)

    # Remove rows whose future label is NaN (last 'horizon' rows)
    feat = feat.dropna(subset=["label"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    feat.to_parquet(args.out)
    print(f"Wrote {args.out} with shape {feat.shape}")
