# src/data.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple, Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

NUMERIC_DTYPES = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]


def _infer_features(df: pd.DataFrame, timestamp_col: str) -> List[str]:
    cols = []
    for c in df.columns:
        if c == timestamp_col or c == "label":
            continue
        if str(df[c].dtype).lower() in NUMERIC_DTYPES:
            cols.append(c)
    return cols


def make_labels(
    df: pd.DataFrame, horizon: int, bid_cols: List[str], ask_cols: List[str]
) -> pd.Series:
    # Mid-price at time t from best bid/ask
    best_bid = df[bid_cols].iloc[:, 0]
    best_ask = df[ask_cols].iloc[:, 0]
    mid = (best_bid + best_ask) / 2.0
    mid_future = mid.shift(-horizon)
    y = (mid_future > mid).astype(int)
    y.iloc[-horizon:] = np.nan  # cannot label tail
    return y, mid


class SequenceDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),  # (C, T)
            torch.from_numpy(np.array(self.y[idx])),
        )


def build_sequences(
    df: pd.DataFrame, features: List[str], label: pd.Series, seq_len: int
):
    """
    Turn a time-ordered table into overlapping sequences.

    Input shapes:
      - df[features].values -> (N, C)
      - label.values        -> (N,)
    Output shapes:
      - X_seq -> (N_seq, C, T)
      - y_seq -> (N_seq,)
    """
    import numpy as np

    X = df[features].values  # (N, C)
    y = label.values  # (N,)

    if X.ndim != 2:
        raise ValueError(
            f"Expected X to be 2-D (N, C), got shape {X.shape} with ndim={X.ndim}"
        )
    N, C = X.shape
    if N < seq_len:
        raise ValueError(f"seq_len={seq_len} is larger than N={N}")

    N_seq = N - seq_len + 1

    # Build windows explicitly to avoid stride tricks issues
    X_seq = np.empty((N_seq, seq_len, C), dtype=np.float32)
    for i in range(N_seq):
        X_seq[i, :, :] = X[i : i + seq_len, :]

    # Align labels to window ends
    y_seq = y[seq_len - 1 :]

    # Drop windows where the label is NaN (end-of-series due to horizon)
    mask = ~np.isnan(y_seq)
    X_seq = X_seq[mask]
    y_seq = y_seq[mask]

    # Final model input shape: (batch, channels, time)
    if X_seq.ndim != 3:
        raise ValueError(
            f"After windowing, expected 3-D X_seq, got {X_seq.shape} with ndim={X_seq.ndim}"
        )
    X_seq = np.transpose(X_seq, (0, 2, 1))  # (N_seq, C, T)

    return X_seq, y_seq


def time_split(n: int, train_frac: float, val_frac_of_rest: float):
    n_train = int(n * train_frac)
    rest = n - n_train
    n_val = int(rest * val_frac_of_rest)
    idx = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    return idx


def load_dataset(cfg) -> Tuple[Dict, str, int]:
    path = cfg["data"]["path"]
    fmt = cfg["data"]["format"]
    ts_col = cfg["data"]["timestamp_col"]
    seq_len = int(cfg["data"]["seq_len"])
    horizon = int(cfg["data"]["horizon"])
    standardize = bool(cfg["data"]["standardize"])

    if fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Infer features if not given
    features = cfg["data"]["features"]
    if features is None:
        features = _infer_features(df, ts_col)

    # Heuristics to find bid/ask columns
    bid_cols = [c for c in df.columns if c.startswith("bid_px_")]
    ask_cols = [c for c in df.columns if c.startswith("ask_px_")]
    if len(bid_cols) == 0 or len(ask_cols) == 0:
        raise ValueError(
            "Expected bid_px_* and ask_px_* columns for mid-price labeling."
        )

    label, mid = make_labels(df, horizon, bid_cols, ask_cols)

    # Standardize features using train split only
    X = df[features].copy()
    n = len(df)
    idx = time_split(n, cfg["data"]["train_frac"], cfg["data"]["val_frac_of_rest"])
    scaler = StandardScaler()
    if standardize:
        X_train = X.iloc[idx["train"][0] : idx["train"][1]]
        scaler.fit(X_train.values)
        X.loc[:, features] = scaler.transform(X.values)
    else:
        scaler = None

    # Build sequences
    X_seq, y_seq = build_sequences(
        pd.concat([X, label.rename("label")], axis=1), features, label, seq_len
    )

    # Compute splits in sequence space
    n_seq = len(X_seq)
    idx_seq = time_split(
        n_seq, cfg["data"]["train_frac"], cfg["data"]["val_frac_of_rest"]
    )

    splits = {}
    for split, (s, e) in idx_seq.items():
        Xs = X_seq[s:e]
        ys = y_seq[s:e]
        ds = SequenceDS(Xs, ys)
        splits[split] = ds

    meta = {
        "features": features,
        "scaler": scaler,
        "seq_len": seq_len,
        "n_channels": len(features),
    }
    return splits, meta, n_seq


def save_scaler(scaler, path):
    if scaler is None:
        return
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def metrics_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    return {"auc": float(auc), "acc": float(acc)}
