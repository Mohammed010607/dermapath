"""Combined Dice + BCE loss.

Binary cross-entropy supervises every pixel, while the soft Dice term directly optimises
region overlap and counteracts the class imbalance seen in the EDA (lesions cover a small,
variable fraction of each image). The two are averaged with equal weight by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)  # sum over channel + spatial, keep the batch dimension
        intersection = (probs * target).sum(dims)
        cardinality = probs.sum(dims) + target.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss
