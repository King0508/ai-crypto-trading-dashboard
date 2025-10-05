# src/train.py
import os, json, argparse, yaml, math, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from data import load_dataset, save_scaler, metrics_binary
from model import TCN

def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 1337))

    device = cfg.get("device", "cpu")
    artifacts_dir = cfg.get("artifacts_dir", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    splits, meta, n_seq = load_dataset(cfg)
    in_ch = meta["n_channels"] if cfg["model"]["in_channels"] is None else cfg["model"]["in_channels"]
    model = TCN(
        in_ch=in_ch,
        hidden_ch=cfg["model"]["hidden_channels"],
        n_layers=cfg["model"]["n_layers"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    loaders = {
        split: DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=(split=="train"), num_workers=0)
        for split, ds in splits.items()
    }

    best_val = -1
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # Eval on val
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for X, y in loaders["val"]:
                X = X.to(device)
                logits = model(X)
                prob = torch.sigmoid(logits).cpu().numpy()
                y_true.append(y.numpy()); y_prob.append(prob)
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        m = metrics_binary(y_true, y_prob)
        print(f"Val metrics: {m}")

        if np.isnan(m["auc"]) or m["auc"] <= best_val:
            bad_epochs += 1
        else:
            best_val = m["auc"]
            bad_epochs = 0
            # save checkpoint
            torch.save(model.state_dict(), os.path.join(artifacts_dir, "model.pt"))
            # save meta
            save_scaler(meta["scaler"], os.path.join(artifacts_dir, "scaler.pkl"))
            with open(os.path.join(artifacts_dir, "meta.json"), "w") as f:
                json.dump({"features": meta["features"], "seq_len": meta["seq_len"]}, f, indent=2)
            with open(os.path.join(artifacts_dir, "val_metrics.json"), "w") as f:
                json.dump(m, f, indent=2)

        if bad_epochs > patience:
            print("Early stopping.")
            break

    print("Training done. Best val AUC:", best_val)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args)
