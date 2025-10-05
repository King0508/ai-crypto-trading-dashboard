# scripts/simulate_lob.py
"""
Generates a synthetic limit-order-book-like dataset suitable for the starter.
It's NOT realistic microstructure—just structured noise with regimes,
so you can prove your pipeline end-to-end.
"""
import numpy as np
import pandas as pd
import argparse

def gen_series(n, regime_len=200, drift_low=-0.02, drift_high=0.02, vol=0.05, seed=42):
    rng = np.random.default_rng(seed)
    price = 100.0
    prices = []
    i = 0
    drift = 0.0
    while i < n:
        if i % regime_len == 0:
            drift = rng.uniform(drift_low, drift_high)
        # AR(1) + drift
        noise = rng.normal(0, vol)
        price = price * (1 + drift*0.001 + noise*0.001)
        prices.append(price)
        i += 1
    return np.array(prices)

def make_lob(prices, n_levels=5, spread_ticks=1, tick=0.01, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    mid = prices
    bid_px_1 = mid - spread_ticks*tick/2
    ask_px_1 = mid + spread_ticks*tick/2
    # Deeper levels with random walks around best
    data = {
        "bid_px_1": bid_px_1, "ask_px_1": ask_px_1,
    }
    for lvl in range(2, n_levels+1):
        data[f"bid_px_{lvl}"] = bid_px_1 - (lvl-1)*tick - rng.normal(0, tick*0.1, size=len(mid))
        data[f"ask_px_{lvl}"] = ask_px_1 + (lvl-1)*tick + rng.normal(0, tick*0.1, size=len(mid))
    # Sizes correlated with proximity to mid
    for lvl in range(1, n_levels+1):
        base = rng.integers(50, 200, size=len(mid))
        jitter = rng.normal(0, 10, size=len(mid))
        data[f"bid_sz_{lvl}"] = np.clip(base + jitter - (lvl-1)*5, 1, None)
        data[f"ask_sz_{lvl}"] = np.clip(base + jitter - (lvl-1)*5, 1, None)
    return pd.DataFrame(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=40000)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--out", type=str, default="data/synth.parquet")
    args = ap.parse_args()

    prices = gen_series(args.n_samples, regime_len=250, seed=123)
    df = make_lob(prices, n_levels=5)
    df.insert(0, "timestamp", np.arange(len(df)))  # simple integer timestamp

    # Optional handcrafted features (imbalances)
    for lvl in range(1, 6):
        df[f"imb_{lvl}"] = (df[f"bid_sz_{lvl}"] - df[f"ask_sz_{lvl}"]) / (df[f"bid_sz_{lvl}"] + df[f"ask_sz_{lvl}"] + 1e-6)

    # Save parquet
    out = args.out
    if out.endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    print(f"Wrote {out} with shape {df.shape}")

if __name__ == "__main__":
    main()
