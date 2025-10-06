# src/model.py
from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove the extra padding on the right to keep causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network producing a single logit per sequence.
    __init__(in_channels, hidden_channels, n_layers, kernel_size, dropout)
    forward(x): x is (B, C, T) -> returns (B,)
    """

    def __init__(
        self, in_channels, hidden_channels=64, n_layers=5, kernel_size=3, dropout=0.1
    ):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i in range(n_layers):
            ch_out = hidden_channels
            dilation = 2**i
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size, dilation, dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, 1)
        )
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # x: (B, C, T)
        h = self.tcn(x)  # (B, H, T)
        h = self.head(h).squeeze(-1)  # (B, H)
        logits = self.out(h).squeeze(-1)  # (B,)
        return logits
