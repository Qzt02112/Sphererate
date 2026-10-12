from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CircularConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CircularConv2d(in_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            CircularConv2d(out_channels, out_channels, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = CircularConv2d(input_channels + hidden_channels, hidden_channels * 2, 3)
        self.candidate = CircularConv2d(input_channels + hidden_channels, hidden_channels, 3)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[-2],
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        combined = torch.cat([x, h], dim=1)
        reset, update = torch.sigmoid(self.gates(combined)).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate
