"""LagFireNet: lag-consistent multimodal ConvLSTM + U-Net fire mapper."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden = hidden
        self.conv = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel_size, padding=padding)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, b, h, w, device):
        z = torch.zeros(b, self.hidden, h, w, device=device)
        return z, z.clone()


class ConvLSTM(nn.Module):
    def __init__(self, in_ch: int, hidden: int, num_layers: int = 2):
        super().__init__()
        cells = []
        for i in range(num_layers):
            cin = in_ch if i == 0 else hidden
            cells.append(ConvLSTMCell(cin, hidden))
        self.cells = nn.ModuleList(cells)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → last hidden (B, hidden, H, W) + temporal attn mix."""
        b, t, _, h, w = x.shape
        device = x.device
        layer_in = x
        last_h = None
        seq_out = None
        for cell in self.cells:
            h_t, c_t = cell.init_state(b, h, w, device)
            outs = []
            for ti in range(t):
                h_t, c_t = cell(layer_in[:, ti], h_t, c_t)
                outs.append(h_t)
            seq_out = torch.stack(outs, dim=1)
            layer_in = seq_out
            last_h = h_t
        # temporal attention over final layer sequence
        assert seq_out is not None and last_h is not None
        score = seq_out.mean(dim=(3, 4))  # B,T,Hiddens
        attn = torch.softmax(score.mean(dim=-1), dim=1)  # B,T
        mixed = (seq_out * attn[:, :, None, None, None]).sum(dim=1)
        return mixed


class SpatialEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


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


class LagFireNet(nn.Module):
    """
    LagFireNet — California 1 km wildfire alert network.

    Lag-consistent multimodal encoder (ERA5/S5P ConvLSTM + DEM/S2 spatial)
    → fuse → U-Net decoder → dense fire logits.
    """

    def __init__(
        self,
        era5_ch: int = config.ERA5_CHANNELS,
        s5p_ch: int = config.S5P_CHANNELS,
        dem_ch: int = config.DEM_CHANNELS,
        s2_ch: int = config.S2_CHANNELS,
        hidden: int = 64,
        s5p_hidden: int = 32,
        static_ch: int = 32,
        fuse_ch: int = 128,
        use_era5: bool = True,
        use_s5p: bool = True,
        use_s2: bool = True,
        use_dem: bool = True,
    ):
        super().__init__()
        self.use_era5 = use_era5
        self.use_s5p = use_s5p
        self.use_s2 = use_s2
        self.use_dem = use_dem

        self.era5_stem = nn.Conv2d(era5_ch, 32, 3, padding=1)
        self.era5_clstm = ConvLSTM(32, hidden, num_layers=2)

        self.s5p_stem = nn.Conv2d(s5p_ch, 16, 3, padding=1)
        self.s5p_clstm = ConvLSTM(16, s5p_hidden, num_layers=2)

        self.dem_enc = SpatialEncoder(dem_ch, static_ch)
        self.s2_enc = SpatialEncoder(s2_ch, static_ch)

        fuse_in = 0
        if use_era5:
            fuse_in += hidden
        if use_s5p:
            fuse_in += s5p_hidden
        if use_dem:
            fuse_in += static_ch
        if use_s2:
            fuse_in += static_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, fuse_ch, 1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True),
        )

        # U-Net decoder
        self.down1 = DoubleConv(fuse_ch, fuse_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(fuse_ch, fuse_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(fuse_ch * 2, fuse_ch * 4)
        self.up2 = nn.ConvTranspose2d(fuse_ch * 4, fuse_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(fuse_ch * 4, fuse_ch * 2)
        self.up1 = nn.ConvTranspose2d(fuse_ch * 2, fuse_ch, 2, stride=2)
        self.dec1 = DoubleConv(fuse_ch * 2, fuse_ch)
        self.head = nn.Conv2d(fuse_ch, 1, 1)

    def _encode_temporal(self, x: torch.Tensor, stem: nn.Module, clstm: ConvLSTM) -> torch.Tensor:
        # x: B,T,C,H,W
        b, t, c, h, w = x.shape
        x = stem(x.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        return clstm(x)

    def forward(
        self,
        era5: torch.Tensor,
        s5p: torch.Tensor,
        s2: torch.Tensor,
        dem: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        if self.use_era5:
            parts.append(self._encode_temporal(era5, self.era5_stem, self.era5_clstm))
        if self.use_s5p:
            parts.append(self._encode_temporal(s5p, self.s5p_stem, self.s5p_clstm))
        if self.use_dem:
            parts.append(self.dem_enc(dem))
        if self.use_s2:
            parts.append(self.s2_enc(s2))
        # align spatial size to first part
        h, w = parts[0].shape[-2:]
        aligned = []
        for p in parts:
            if p.shape[-2:] != (h, w):
                p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            aligned.append(p)
        x = self.fuse(torch.cat(aligned, dim=1))

        s1 = self.down1(x)
        x = self.pool1(s1)
        s2f = self.down2(x)
        x = self.pool2(s2f)
        x = self.bottleneck(x)
        x = self.up2(x)
        if x.shape[-2:] != s2f.shape[-2:]:
            x = F.interpolate(x, size=s2f.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, s2f], dim=1))
        x = self.up1(x)
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, s1], dim=1))
        logits = self.head(x)
        # match input spatial size
        th, tw = dem.shape[-2:]
        if logits.shape[-2:] != (th, tw):
            logits = F.interpolate(logits, size=(th, tw), mode="bilinear", align_corners=False)
        return logits


# Deprecated alias (older docs / notebooks)
LiveMultimodalDense = LagFireNet
