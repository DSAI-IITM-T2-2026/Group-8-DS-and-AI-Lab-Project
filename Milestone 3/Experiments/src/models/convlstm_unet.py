"""ConvLSTM encoder + U-Net-style decoder for spatiotemporal fire segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch: int, height: int, width: int, device):
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        self.num_layers = num_layers
        cells = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(cin, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        returns last hidden per layer list and final layer sequence (B, T, Hiddens, H, W)
        """
        b, t, _, h, w = x.shape
        device = x.device
        layer_input = x
        last_states = []
        for cell in self.cells:
            h_t, c_t = cell.init_hidden(b, h, w, device)
            outputs = []
            for ti in range(t):
                h_t, c_t = cell(layer_input[:, ti], h_t, c_t)
                outputs.append(h_t)
            layer_input = torch.stack(outputs, dim=1)
            last_states.append((h_t, c_t))
        return last_states, layer_input


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvLSTMUNet(nn.Module):
    """
    ConvLSTM encoder over 7-day sequence → U-Net decoder → (B, 1, H, W) logits.
    Defaults: hidden=32, 2 ConvLSTM layers (judgment call).
    """

    def __init__(
        self,
        in_channels: int = 30,
        hidden_channels: int = 32,
        num_layers: int = 2,
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder = ConvLSTM(in_channels, hidden_channels, num_layers=num_layers)
        # spatial encoder/decoder on last ConvLSTM state
        self.down1 = DoubleConv(hidden_channels, hidden_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(hidden_channels, hidden_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(hidden_channels * 2, hidden_channels * 4)
        self.up2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(hidden_channels * 4, hidden_channels * 2)
        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 2, stride=2)
        self.dec1 = DoubleConv(hidden_channels * 2, hidden_channels)
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, T, C, H, W) or (B, T, H, W, C)
        """
        if x.dim() == 5 and x.shape[-1] < x.shape[2]:
            # (B, T, H, W, C) → (B, T, C, H, W)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        last_states, _ = self.encoder(x)
        feat = last_states[-1][0]  # (B, hidden, H, W)

        s1 = self.down1(feat)
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
