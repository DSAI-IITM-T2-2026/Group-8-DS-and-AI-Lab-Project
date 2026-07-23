"""U-Net on the last day of the 7-day window (ablates temporal ConvLSTM)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.convlstm_unet import DoubleConv


class UNetLastDay(nn.Module):
    """
    Spatial U-Net on X[:, -1] only.
    Input: (B, T, H, W, C) or (B, T, C, H, W) or (B, C, H, W)
    Output: (B, 1, H, W) logits
    """

    def __init__(
        self,
        in_channels: int = 30,
        hidden_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        h = hidden_channels
        self.down1 = DoubleConv(in_channels, h)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(h, h * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(h * 2, h * 4)
        self.up2 = nn.ConvTranspose2d(h * 4, h * 2, 2, stride=2)
        self.dec2 = DoubleConv(h * 4, h * 2)
        self.up1 = nn.ConvTranspose2d(h * 2, h, 2, stride=2)
        self.dec1 = DoubleConv(h * 2, h)
        self.head = nn.Conv2d(h, out_channels, kernel_size=1)

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            # (B, T, H, W, C) vs (B, T, C, H, W)
            if x.shape[-1] < x.shape[2]:
                x = x[:, -1]  # (B, H, W, C)
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                x = x[:, -1]  # (B, C, H, W)
        elif x.dim() == 4 and x.shape[-1] < x.shape[1]:
            # (B, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bchw(x)
        s1 = self.down1(x)
        x = self.pool1(s1)
        s2 = self.down2(x)
        x = self.pool2(s2)
        x = self.bottleneck(x)
        x = self.up2(x)
        if x.shape[-2:] != s2.shape[-2:]:
            x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.head(x)
