# src/train.py
from __future__ import annotations

import os
import json
import math
import time
import random
import argparse
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm

from data import load_dataset
from model import TCN  # make sure your model class is named TCN in src/model.py


# ------------------------- helpers ------------------------- #


def _to_int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# compute pos_weight for BCEWithLogitsLoss if labels are skewed
def compute_pos_weight(train_ds: TensorDataset, device: torch.device) -> torch.Tensor:
    # read labels efficiently
    loader = DataLoader(train_ds, batch_size=4096, shuffle=False)
    total = 0
    pos = 0
    for _, y in loader:
        y_np = y.numpy()
        total += y_np.size
        pos += int(y_np.sum())
    neg = max(total - pos, 0)
    pos = max(pos, 1)  # avoid /0
    val = float(neg / pos)
    return torch.tensor([val], device=device, dtype=torch.float32)


# ------------------------- training/eval steps ------------------------- #


def run_val(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    sigmoid = nn.Sigmoid()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            probs = sigmoid(logits).detach().cpu().numpy()
            y_prob.append(probs)
            y_true.append(yb.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    return float(auc), float(acc), y_true, y_prob


def main(args):
    # ---------------- load config ----------------
    with open(args.config, "r") as f:
        cfg: Dict = yaml.safe_load(f)

    seed = _to_int(cfg.get("seed", 1337), 1337)
    set_seed(seed)

    device = _get_device(cfg.get("device", "cpu"))

    # ---------------- dataset ----------------
    splits, meta, n_seq, scaler = load_dataset(cfg)

    batch_size = _to_int(cfg.get("train", {}).get("batch_size", 256), 256)
    epochs = _to_int(cfg.get("train", {}).get("epochs", 10), 10)
    lr = _to_float(cfg.get("train", {}).get("lr", 3e-4), 3e-4)
    weight_decay = _to_float(cfg.get("train", {}).get("weight_decay", 1e-4), 1e-4)
    grad_clip_norm = _to_float(cfg.get("train", {}).get("grad_clip_norm", 1.0), 1.0)
    patience = _to_int(cfg.get("train", {}).get("early_stop_patience", 3), 3)

    artifacts_dir = cfg.get("artifacts_dir", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    ckpt_path = os.path.join(artifacts_dir, "model.pt")

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ---------------- model ----------------
    mcfg = cfg.get("model", {})
    model = TCN(
        in_channels=(
            meta.in_channels
            if mcfg.get("in_channels") in (None, "null")
            else int(mcfg["in_channels"])
        ),
        hidden_channels=_to_int(mcfg.get("hidden_channels", 64), 64),
        n_layers=_to_int(mcfg.get("n_layers", 5), 5),
        kernel_size=_to_int(mcfg.get("kernel_size", 3), 3),
        dropout=_to_float(mcfg.get("dropout", 0.1), 0.1),
    ).to(device)

    # optional class weighting
    use_pos_weight = True
    pos_weight = compute_pos_weight(splits["train"], device) if use_pos_weight else None

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------- train loop ----------------
    best_auc = -np.inf
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        running = 0.0
        nb = 0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            opt.step()

            running += float(loss.item())
            nb += 1
            pbar.set_postfix({"loss": f"{running/nb:.3f}"})

        # ---- validation ----
        val_auc, val_acc, y_true, y_prob = run_val(model, val_loader, device)

        # tiny threshold sweep (0.30..0.70) for info
        ths = np.linspace(0.3, 0.7, 9)
        best_th, best_acc = 0.5, -1.0
        for th in ths:
            acc = accuracy_score(y_true, (y_prob >= th).astype(int))
            if acc > best_acc:
                best_acc, best_th = acc, th

        print(
            f"Val metrics: {{'auc': {val_auc}, 'acc': {val_acc}}} | [val] best_th={best_th:.2f} acc={best_acc:.4f}"
        )

        # early stopping on AUC
        improved = val_auc > best_auc or (
            math.isnan(best_auc) and not math.isnan(val_auc)
        )
        if improved:
            best_auc = val_auc
            no_improve = 0
            # save checkpoint (state_dict + scaler + metadata)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "meta": meta.__dict__,
                },
                ckpt_path,
            )
            # Save scaler separately for easy loading in inference
            scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            # Save metadata as JSON for human readability
            meta_path = os.path.join(artifacts_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "feature_names": scaler["features"],
                        "in_channels": meta.in_channels,
                        "seq_len": meta.seq_len,
                        "horizon": meta.horizon,
                        "model_config": mcfg,
                    },
                    f,
                    indent=2,
                )
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Training done. Best val AUC: {best_auc}")

    # convenience print
    if os.path.exists(ckpt_path):
        print(f"Saved best checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config."
    )
    args = parser.parse_args()
    main(args)
