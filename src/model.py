# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)

        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        out = self.dropout2(out)
        return out

class TCN(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, n_layers=5, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(n_layers):
            out_ch = hidden_ch
            dilation = 2 ** i
            layers.append(TemporalBlock(ch, out_ch, kernel_size, dilation, dropout))
            ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, 1)
        )

    def forward(self, x):
        # x: (B, C, T)
        z = self.tcn(x)
        logit = self.head(z)
        return logit.squeeze(1)  # (B,)
