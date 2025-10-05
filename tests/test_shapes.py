# tests/test_shapes.py
import torch
from src.model import TCN

def test_forward_shapes():
    B, C, T = 8, 32, 40
    m = TCN(in_ch=C, hidden_ch=64, n_layers=3, kernel_size=3, dropout=0.1)
    x = torch.randn(B, C, T)
    y = m(x)
    assert y.shape == (B,), f"Expected (B,), got {y.shape}"
