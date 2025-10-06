# src/eval.py
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from data import load_dataset
from model import TCN


def _to_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_device(cfg_device: str) -> torch.device:
    d = (cfg_device or "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_pnl_from_mid(
    mid: np.ndarray,
    y_prob: np.ndarray,
    seq_len: int,
    horizon: int,
    threshold: float = 0.5,
    fee_bps: float = 2.0,
) -> Tuple[float, float]:
    """
    Return-based toy PnL:
      - Each prediction corresponds to the window ending at index 'end'.
      - next return r = (mid[end+1] - mid[end]) / mid[end]
      - Position = +1 if prob>=threshold else -1
      - Fee charged on position flips (bps).
    """
    assert len(mid) > (seq_len + horizon), "mid series too short for seq_len/horizon."
    start = seq_len - 1
    ends = np.arange(start, start + len(y_prob))  # window-end indices

    ret = (mid[ends + 1] - mid[ends]) / (mid[ends] + 1e-9)

    p = y_prob[:-1]  # align to ret length
    pos = np.where(p >= threshold, 1.0, -1.0)

    switch = np.abs(np.diff(np.r_[pos[0], pos]))  # 0 (no change) or 2 (flip)
    fees = (fee_bps / 10000.0) * (switch > 0).astype(float)

    pnl = pos * ret - fees
    mean = float(np.mean(pnl))
    std = float(np.std(pnl) + 1e-12)
    return mean, std


def main(args):
    # --- load config
    with open(args.config, "r") as f:
        cfg: Dict = yaml.safe_load(f)

    device = _get_device(cfg.get("device", "cpu"))

    # --- dataset
    splits, meta, _ = load_dataset(cfg)

    batch_size = int(cfg.get("train", {}).get("batch_size", 256))
    th = _to_float(cfg.get("eval", {}).get("threshold", 0.5), 0.5)
    fee_bps = _to_float(cfg.get("eval", {}).get("trade_fee_bps", 2.0), 2.0)

    test_loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False)

    # --- build model
    model_cfg = cfg.get("model", {})
    model = TCN(
        in_channels=(
            meta.in_channels
            if model_cfg.get("in_channels") in (None, "null")
            else int(model_cfg["in_channels"])
        ),
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        n_layers=int(model_cfg.get("n_layers", 5)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    model.eval()

    # --- load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    # --- inference on TEST
    y_true_list, y_prob_list = [], []
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            probs = sigmoid(logits).detach().cpu().numpy()
            y_prob_list.append(probs)
            y_true_list.append(yb.numpy())

    y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.array([])
    y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.array([])
    if y_prob.size == 0:
        raise RuntimeError("No test predictions produced. Check dataset/test split.")

    # --- metrics
    results = {}
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    results["test_auc"] = float(auc)

    y_pred = (y_prob >= th).astype(int)
    acc = accuracy_score(y_true, y_pred)
    results["test_acc"] = float(acc)

    # --- returns-based toy PnL
    data_cfg = cfg["data"]
    if data_cfg.get("format", "parquet").lower() == "parquet":
        df_full = pd.read_parquet(data_cfg["path"])
    else:
        df_full = pd.read_csv(data_cfg["path"])

    if "mid" in df_full.columns:
        mid = df_full["mid"].astype(float).values
    elif "close" in df_full.columns:
        mid = df_full["close"].astype(float).values
    else:
        raise ValueError("PnL needs 'mid' or 'close' in the data file.")

    mean_pnl, std_pnl = compute_pnl_from_mid(
        mid=mid,
        y_prob=y_prob,
        seq_len=int(data_cfg["seq_len"]),
        horizon=int(data_cfg["horizon"]),
        threshold=th,
        fee_bps=fee_bps,
    )
    sr = mean_pnl / (std_pnl + 1e-12)
    results["toy_mean_pnl"] = float(mean_pnl)
    results["toy_sr"] = float(sr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args)
