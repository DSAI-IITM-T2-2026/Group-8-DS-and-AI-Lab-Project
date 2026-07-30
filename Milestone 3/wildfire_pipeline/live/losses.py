"""Focal + Tversky losses (precision-first)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = config.FOCAL_ALPHA, gamma: float = config.FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        # α on positives, (1-α) on negatives (class-balanced focal)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = config.TVERSKY_ALPHA,
        beta: float = config.TVERSKY_BETA,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Use spatial *means* (not sums). On 256² tiles, sum-FP is O(10^4–10^5) so
        Tversky≈1 and gradients vanish — loss looks stuck near 1.0.
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(1, probs.dim()))
        tp = (probs * targets).mean(dim=dims)
        fp = (probs * (1 - targets)).mean(dim=dims)
        fn = ((1 - probs) * targets).mean(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1.0 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, focal_weight: float = 1.0, tversky_weight: float = 1.0):
        super().__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, return_components: bool = False
    ):
        fl = self.focal(logits, targets)
        tv = self.tversky(logits, targets)
        total = self.focal_weight * fl + self.tversky_weight * tv
        if return_components:
            return total, fl.detach(), tv.detach()
        return total
