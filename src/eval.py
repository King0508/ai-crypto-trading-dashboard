# src/eval.py
import os, argparse, json, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import load_dataset, metrics_binary
from model import TCN

def compute_pnl(y_true, y_prob, threshold=0.5, fee_bps=1.0):
    """
    Toy backtest:
    - Go long if prob>=threshold; short otherwise.
    - PnL proxy = (2*y-1) * (2*pred-1) - fees per trade
    This is simplistic (doesn't use actual returns); replace with your signal->return mapping.
    """
    pred = (y_prob >= threshold).astype(int)
    # Map {0,1} -> {-1,+1}
    y_signed = 2*y_true - 1
    p_signed = 2*pred - 1
    edge = y_signed * p_signed  # +1 if correct, -1 if wrong
    fees = fee_bps/10000.0  # per decision; over-penalizes—tune as needed
    pnl = edge - fees
    return float(np.mean(pnl)), float(np.std(pnl))

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cpu")
    splits, meta, _ = load_dataset(cfg)
    in_ch = meta["n_channels"] if cfg["model"]["in_channels"] is None else cfg["model"]["in_channels"]
    model = TCN(
        in_ch=in_ch,
        hidden_ch=cfg["model"]["hidden_channels"],
        n_layers=cfg["model"]["n_layers"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    loader = DataLoader(splits["test"], batch_size=512, shuffle=False, num_workers=0)

    y_true, y_prob = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            prob = torch.sigmoid(logits).cpu().numpy()
            y_true.append(y.numpy()); y_prob.append(prob)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    m = metrics_binary(y_true, y_prob, cfg["eval"]["threshold"])

    mean_pnl, std_pnl = compute_pnl(
        y_true, y_prob, threshold=cfg["eval"]["threshold"], fee_bps=cfg["eval"]["trade_fee_bps"]
    )
    sr = mean_pnl / (std_pnl + 1e-9)
    results = {"test_auc": m["auc"], "test_acc": m["acc"], "toy_mean_pnl": mean_pnl, "toy_sr": sr}

    print(json.dumps(results, indent=2))
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    with open(os.path.join(cfg["artifacts_dir"], "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()
    main(args)
