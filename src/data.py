# src/data.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


@dataclass
class DatasetMeta:
    in_channels: int
    seq_len: int
    horizon: int
    n_total_sequences: int
    feature_names: List[str]
    standardize: bool


def _ensure_mid_and_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Ensure a 'mid' column exists, and if 'label' is missing, create it:
      label[t] = 1.0 if mid[t+horizon] > mid[t], else 0.0.
    Works for LOB (bid/ask present) or bar data (close).
    """
    if "mid" not in df.columns:
        if {"bid_px_1", "ask_px_1"}.issubset(df.columns):
            df["mid"] = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
        elif "close" in df.columns:
            df["mid"] = df["close"].astype(float)
        else:
            raise ValueError(
                "No 'mid' present and neither LOB ('bid_px_1','ask_px_1') "
                "nor 'close' found to construct it."
            )

    if "label" not in df.columns:
        fwd = df["mid"].shift(-horizon)
        df["label"] = (fwd > df["mid"]).astype(float)

    return df


def build_sequences(
    df: pd.DataFrame, features: List[str], label: pd.Series, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time-ordered table to overlapping sequences.
    X_seq: (N_seq, C, T), y_seq: (N_seq,)
    """
    X = df[features].values  # (N, C)
    y = label.values  # (N,)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2-D, got shape {X.shape} (ndim={X.ndim})")
    N, C = X.shape
    if N < seq_len:
        raise ValueError(f"seq_len={seq_len} > N={N}")

    N_seq = N - seq_len + 1

    X_seq = np.empty((N_seq, seq_len, C), dtype=np.float32)
    for i in range(N_seq):
        X_seq[i, :, :] = X[i : i + seq_len, :]

    y_seq = y[seq_len - 1 :]  # align labels to window ends

    # Drop windows whose label is NaN (last 'horizon' rows)
    mask = ~np.isnan(y_seq)
    X_seq = X_seq[mask]
    y_seq = y_seq[mask]

    # Conv1d expects (N, C, T)
    X_seq = np.transpose(X_seq, (0, 2, 1))
    return X_seq, y_seq.astype(np.float32)


def _standardize_train_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std on TRAIN: x is (N, C, T)."""
    mean = x.mean(axis=(0, 2))
    std = x.std(axis=(0, 2))
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardization(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply per-channel standardization to (N, C, T)."""
    x = (x - mean[:, None]) / std[:, None]
    return x.astype(np.float32)


def load_dataset(cfg: Dict) -> Tuple[Dict[str, TensorDataset], DatasetMeta, int]:
    """
    Read parquet/csv defined in cfg['data'], build (N,C,T) sequences,
    return TensorDatasets for train/val/test, plus metadata.
    """
    data_cfg = cfg["data"]
    path = data_cfg["path"]
    fmt = data_cfg.get("format", "parquet")
    ts_col: Optional[str] = data_cfg.get("timestamp_col")
    seq_len = int(data_cfg["seq_len"])
    horizon = int(data_cfg["horizon"])
    train_frac = float(data_cfg.get("train_frac", 0.7))
    val_frac_of_rest = float(data_cfg.get("val_frac_of_rest", 0.5))
    standardize = bool(data_cfg.get("standardize", True))

    # load df
    if fmt.lower() == "parquet":
        df = pd.read_parquet(path)
    elif fmt.lower() == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data format: {fmt}")

    # sort by timestamp (chronological)
    if ts_col and ts_col in df.columns:
        df = df.sort_values(ts_col).reset_index(drop=True)

    # BAR/LOB shim
    df = _ensure_mid_and_label(df, horizon=horizon)

    # infer features (numeric except timestamp/label)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(["label"])
    if ts_col and ts_col in df.columns:
        exclude.add(ts_col)

    if data_cfg.get("features") is None:
        features = [c for c in num_cols if c not in exclude]
    else:
        features = list(data_cfg["features"])

    if len(features) == 0:
        raise ValueError(
            "No numeric features found. Check data file or 'features' list."
        )

    # sequences
    X_seq, y_seq = build_sequences(df, features, df["label"], seq_len)
    N, C, T = X_seq.shape
    n_seq = N

    # splits (chronological)
    train_end = int(math.floor(N * train_frac))
    rem = N - train_end
    val_len = int(math.floor(rem * val_frac_of_rest))
    # test_len = rem - val_len

    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]
    X_val = X_seq[train_end : train_end + val_len]
    y_val = y_seq[train_end : train_end + val_len]
    X_test = X_seq[train_end + val_len :]
    y_test = y_seq[train_end + val_len :]

    # standardize using TRAIN stats
    if standardize:
        mean_c, std_c = _standardize_train_stats(X_train)
        X_train = _apply_standardization(X_train, mean_c, std_c)
        X_val = _apply_standardization(X_val, mean_c, std_c)
        X_test = _apply_standardization(X_test, mean_c, std_c)
    else:
        mean_c = np.zeros((C,), dtype=np.float32)
        std_c = np.ones((C,), dtype=np.float32)

    def to_ds(x, y):
        return TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )

    splits = {
        "train": to_ds(X_train, y_train),
        "val": to_ds(X_val, y_val),
        "test": to_ds(X_test, y_test),
    }

    meta = DatasetMeta(
        in_channels=C,
        seq_len=T,
        horizon=horizon,
        n_total_sequences=n_seq,
        feature_names=features,
        standardize=standardize,
    )
    return splits, meta, n_seq
