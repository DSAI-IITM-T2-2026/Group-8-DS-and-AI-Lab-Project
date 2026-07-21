from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 6, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualBranchCNN(nn.Module):
    """CNN(S2 patch) + MLP(ERA5/DEM) → fire logit."""

    def __init__(
        self,
        n_tabular: int,
        in_ch: int = 6,
        cnn_embed: int = 128,
        mlp_embed: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = SmallCNN(in_ch=in_ch, embed_dim=cnn_embed)
        self.mlp = TabularMLP(in_dim=n_tabular, embed_dim=mlp_embed, dropout=dropout)
        fused = cnn_embed + mlp_embed
        self.head = nn.Sequential(
            nn.Linear(fused, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.cnn(image), self.mlp(tabular)], dim=1)
        return self.head(z).squeeze(1)
