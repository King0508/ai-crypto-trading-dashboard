# Market Microstructure TCN

Build and test short-horizon crypto signals with **PyTorch**.  
We take a window of recent features, run a **Temporal Convolutional Network (TCN)**, and predict the next move (a few bars ahead). Training/eval is all PyTorch (nn modules, DataLoader, AdamW, early stop). Output includes AUC/ACC and a quick returns-style PnL so you can iterate fast.

---

## Why PyTorch here

- **Causal dilated convs (TCN):** big receptive field without heavy latency; great for 1m/1s streams.
- **Simple shape contract:** `(batch, channels, time)` → **logit**. Easy to swap models or features.
- **Fast loops:** pure torch training (BCEWithLogitsLoss, AdamW, grad clip), clean eval, easy to GPU later.
- **Extensible:** drop in other heads (Transformer/LSTM), add mixed precision, try custom losses.

---

## What you can use this for

- **Alpha research:** next-bar/next-N move classification on BTC (bars today, LOB later).
- **Regime flags:** rising vol/trend vs chop as a classifier target.
- **Signal stacking:** use the prob as a feature in a bigger ensemble.
- **Latency-sensitive forecasting:** TCN is lightweight and causal; good for near-real-time inference.

---

## Quickstart (Windows, CPU)

### 0) env

```powershell
# in repo root
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install numpy pandas pyarrow pyyaml scikit-learn tqdm
```
