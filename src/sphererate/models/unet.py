from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sphererate.models.layers import ConvBlock


class TinyUNetFeatures(nn.Module):
    def __init__(self, in_channels: int = 8, base_channels: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up1 = ConvBlock(base_channels * 2 + base_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1(torch.cat([u1, e1], dim=1))
