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
        return self.fc(self.net(x).flatten(1))


class Era5LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        embed_dim: int = 64,
        dropout: float = 0.2,
        num_layers: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        return self.fc(h_n[-1])


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        hidden = max(embed_dim, 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalFusion(nn.Module):
    """
    S2 CNN + S5P CNN + ERA5 LSTM + S2 numerical MLP + S5P numerical MLP.
    Each branch is optional via flags.
    """

    def __init__(
        self,
        seq_dim: int,
        s2_num_dim: int = 0,
        s5p_num_dim: int = 0,
        s2_in_ch: int = 6,
        s5p_in_ch: int = 2,
        cnn_embed: int = 128,
        s5p_cnn_embed: int = 64,
        lstm_embed: int = 64,
        lstm_hidden: int = 64,
        s2_num_embed: int = 64,
        s5p_num_embed: int = 32,
        dropout: float = 0.2,
        use_s2_patches: bool = True,
        use_s5p_patches: bool = True,
        use_s2_numerical: bool = True,
        use_s5p_numerical: bool = True,
    ):
        super().__init__()
        self.use_s2_patches = use_s2_patches
        self.use_s5p_patches = use_s5p_patches
        self.use_s2_numerical = use_s2_numerical and s2_num_dim > 0
        self.use_s5p_numerical = use_s5p_numerical and s5p_num_dim > 0

        fused = 0
        if self.use_s2_patches:
            self.s2_cnn = SmallCNN(in_ch=s2_in_ch, embed_dim=cnn_embed)
            fused += cnn_embed
        if self.use_s5p_patches:
            self.s5p_cnn = SmallCNN(in_ch=s5p_in_ch, embed_dim=s5p_cnn_embed)
            fused += s5p_cnn_embed

        self.lstm = Era5LSTM(
            input_dim=seq_dim,
            hidden=lstm_hidden,
            embed_dim=lstm_embed,
            dropout=dropout,
        )
        fused += lstm_embed

        if self.use_s2_numerical:
            self.s2_mlp = TabularMLP(s2_num_dim, s2_num_embed, dropout)
            fused += s2_num_embed
        if self.use_s5p_numerical:
            self.s5p_mlp = TabularMLP(s5p_num_dim, s5p_num_embed, dropout)
            fused += s5p_num_embed

        self.head = nn.Sequential(
            nn.Linear(fused, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        seq: torch.Tensor,
        s2_image: torch.Tensor | None = None,
        s5p_image: torch.Tensor | None = None,
        s2_num: torch.Tensor | None = None,
        s5p_num: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.lstm(seq)]
        if self.use_s2_patches:
            parts.append(self.s2_cnn(s2_image))
        if self.use_s5p_patches:
            parts.append(self.s5p_cnn(s5p_image))
        if self.use_s2_numerical:
            parts.append(self.s2_mlp(s2_num))
        if self.use_s5p_numerical:
            parts.append(self.s5p_mlp(s5p_num))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)
