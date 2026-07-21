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
        # seq: [B, T, F]
        out, (h_n, _) = self.lstm(seq)
        return self.fc(h_n[-1])


class S5PMLP(nn.Module):
    def __init__(self, embed_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x)


class CNNLSTMFusion(nn.Module):
    """CNN(S2) + LSTM(ERA5/DEM seq) + optional S5P MLP → fire logit."""

    def __init__(
        self,
        seq_dim: int,
        in_ch: int = 6,
        cnn_embed: int = 128,
        lstm_embed: int = 64,
        lstm_hidden: int = 64,
        s5p_embed: int = 32,
        dropout: float = 0.2,
        use_sentinel5p: bool = False,
    ):
        super().__init__()
        self.use_sentinel5p = use_sentinel5p
        self.cnn = SmallCNN(in_ch=in_ch, embed_dim=cnn_embed)
        self.lstm = Era5LSTM(
            input_dim=seq_dim,
            hidden=lstm_hidden,
            embed_dim=lstm_embed,
            dropout=dropout,
        )
        self.s5p_mlp = S5PMLP(embed_dim=s5p_embed, dropout=dropout) if use_sentinel5p else None
        fused = cnn_embed + lstm_embed + (s5p_embed if use_sentinel5p else 0)
        self.head = nn.Sequential(
            nn.Linear(fused, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        seq: torch.Tensor,
        s5p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.cnn(image), self.lstm(seq)]
        if self.use_sentinel5p:
            if s5p is None:
                raise ValueError("S5P branch enabled but s5p tensor is None")
            parts.append(self.s5p_mlp(s5p))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)
