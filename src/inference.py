"""
Inference engine for real-time predictions.
Loads trained model + scaler and provides predictions with position sizing.
"""
from __future__ import annotations

import os
import json
import pickle
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import TCN


@dataclass
class PredictionResult:
    """Result from model inference."""
    probability: float  # Probability of price going up
    signal: str  # 'LONG', 'SHORT', or 'NEUTRAL'
    confidence: float  # Distance from threshold (0-1 scaled)
    position_size_usd: float  # Recommended position size in USD
    position_size_pct: float  # Recommended position size as % of capital
    timestamp: pd.Timestamp
    price: float


class InferenceEngine:
    """
    Handles model loading and inference for live predictions.
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        meta_path: str,
        device: str = "cpu",
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        capital: float = 10000.0,
        risk_per_trade: float = 0.02,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            scaler_path: Path to scaler pickle file
            meta_path: Path to metadata JSON file
            device: Device to run inference on ('cpu' or 'cuda')
            long_threshold: Probability threshold for LONG signal
            short_threshold: Probability threshold for SHORT signal
            capital: Total capital for position sizing
            risk_per_trade: Risk per trade as fraction of capital (Kelly-ish)
        """
        self.device = torch.device(device)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        
        # Load metadata
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        
        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        self.feature_names = self.scaler["features"]
        self.mean = self.scaler["mean"]
        self.std = self.scaler["std"]
        
        # Load model
        model_cfg = self.meta["model_config"]
        self.model = TCN(
            in_channels=self.meta["in_channels"],
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            n_layers=int(model_cfg.get("n_layers", 5)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        ).to(self.device)
        
        # Load weights
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.sigmoid = nn.Sigmoid()
        
        print(f"[OK] Inference engine loaded")
        print(f"  Model: {model_path}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Seq length: {self.meta['seq_len']}")
        print(f"  Horizon: {self.meta['horizon']} bars")
        print(f"  Thresholds: LONG>{self.long_threshold:.2f}, SHORT<{self.short_threshold:.2f}")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from a DataFrame with the correct ordering.
        
        Args:
            df: DataFrame with feature columns
        
        Returns:
            Array of shape (1, C, T) ready for model
        """
        seq_len = self.meta["seq_len"]
        
        if len(df) < seq_len:
            raise ValueError(f"Need at least {seq_len} bars, got {len(df)}")
        
        # Take last seq_len bars
        df = df.tail(seq_len).copy()
        
        # Extract features in correct order
        X = df[self.feature_names].values  # (T, C)
        
        # Standardize using saved stats
        X = (X - self.mean) / self.std
        
        # Reshape to (1, C, T) for Conv1d
        X = X.T[np.newaxis, :, :]  # (1, C, T)
        
        return X.astype(np.float32)
    
    def predict(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> PredictionResult:
        """
        Make a prediction from recent bars.
        
        Args:
            df: DataFrame with at least seq_len bars of features
            current_price: Current price (uses last close if None)
            timestamp: Timestamp of prediction (uses now if None)
        
        Returns:
            PredictionResult with signal and position sizing
        """
        # Prepare input tensor
        X = self.prepare_features(df)
        X_tensor = torch.from_numpy(X).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(X_tensor)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            prob = self.sigmoid(logits).item()
        
        # Determine signal
        if prob >= self.long_threshold:
            signal = "LONG"
            confidence = (prob - self.long_threshold) / (1.0 - self.long_threshold)
        elif prob <= self.short_threshold:
            signal = "SHORT"
            confidence = (self.short_threshold - prob) / self.short_threshold
        else:
            signal = "NEUTRAL"
            # Distance from nearest threshold
            dist_to_long = self.long_threshold - prob
            dist_to_short = prob - self.short_threshold
            confidence = 1.0 - min(dist_to_long, dist_to_short) / (self.long_threshold - self.short_threshold)
        
        # Position sizing (simple Kelly-ish approach)
        # For LONG/SHORT: scale by confidence and risk_per_trade
        if signal in ["LONG", "SHORT"]:
            base_size = self.capital * self.risk_per_trade
            # Scale by confidence (0.5 to 1.5x base)
            position_size_usd = base_size * (0.5 + confidence)
        else:
            position_size_usd = 0.0
        
        position_size_pct = (position_size_usd / self.capital) * 100
        
        # Get price and timestamp
        if current_price is None:
            current_price = float(df["close"].iloc[-1])
        
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz="UTC")
        
        return PredictionResult(
            probability=prob,
            signal=signal,
            confidence=confidence,
            position_size_usd=position_size_usd,
            position_size_pct=position_size_pct,
            timestamp=timestamp,
            price=current_price,
        )
    
    def predict_from_window(
        self,
        window: np.ndarray,
        current_price: float,
        timestamp: pd.Timestamp,
    ) -> PredictionResult:
        """
        Make prediction from a prepared feature window.
        
        Args:
            window: Pre-standardized feature array of shape (seq_len, n_features)
            current_price: Current price
            timestamp: Current timestamp
        
        Returns:
            PredictionResult
        """
        # Reshape to (1, C, T)
        X = window.T[np.newaxis, :, :].astype(np.float32)
        X_tensor = torch.from_numpy(X).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(X_tensor)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            prob = self.sigmoid(logits).item()
        
        # Determine signal
        if prob >= self.long_threshold:
            signal = "LONG"
            confidence = (prob - self.long_threshold) / (1.0 - self.long_threshold)
        elif prob <= self.short_threshold:
            signal = "SHORT"
            confidence = (self.short_threshold - prob) / self.short_threshold
        else:
            signal = "NEUTRAL"
            dist_to_long = self.long_threshold - prob
            dist_to_short = prob - self.short_threshold
            confidence = 1.0 - min(dist_to_long, dist_to_short) / (self.long_threshold - self.short_threshold)
        
        # Position sizing
        if signal in ["LONG", "SHORT"]:
            base_size = self.capital * self.risk_per_trade
            position_size_usd = base_size * (0.5 + confidence)
        else:
            position_size_usd = 0.0
        
        position_size_pct = (position_size_usd / self.capital) * 100
        
        return PredictionResult(
            probability=prob,
            signal=signal,
            confidence=confidence,
            position_size_usd=position_size_usd,
            position_size_pct=position_size_pct,
            timestamp=timestamp,
            price=current_price,
        )


def test_inference():
    """Test inference on sample data."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/model.pt")
    parser.add_argument("--scaler", type=str, default="artifacts/scaler.pkl")
    parser.add_argument("--meta", type=str, default="artifacts/meta.json")
    parser.add_argument("--data", type=str, default="data/btcusdt_1m.parquet")
    args = parser.parse_args()
    
    # Load engine
    engine = InferenceEngine(
        model_path=args.model,
        scaler_path=args.scaler,
        meta_path=args.meta,
        capital=10000.0,
        risk_per_trade=0.02,
    )
    
    # Load some test data
    df = pd.read_parquet(args.data)
    print(f"\nLoaded {len(df)} bars from {args.data}")
    
    # Make predictions on last 5 windows
    seq_len = engine.meta["seq_len"]
    for i in range(-5, 0):
        window_df = df.iloc[i - seq_len : i]
        result = engine.predict(window_df)
        
        print(f"\n[{result.timestamp}] Price: ${result.price:.2f}")
        print(f"  Probability: {result.probability:.3f}")
        print(f"  Signal: {result.signal} (confidence: {result.confidence:.2f})")
        print(f"  Position: ${result.position_size_usd:.2f} ({result.position_size_pct:.1f}%)")


if __name__ == "__main__":
    test_inference()

