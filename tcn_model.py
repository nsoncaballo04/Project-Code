from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.drop1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.drop2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        hidden_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_filters
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=hidden_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.network(x)  # (batch, hidden, seq_len)
        out = out[:, :, -1]  # final time step
        out = self.head(out)  # (batch, 1)
        return out
