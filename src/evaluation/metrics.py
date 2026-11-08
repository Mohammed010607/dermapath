"""Segmentation metrics.

This module currently provides the overlap metrics used both for validation monitoring
during training (Dice) and later for full evaluation. Task 11 extends it with the remaining
thesis metrics (pixel accuracy, sensitivity, specificity) and per-image aggregation.

Each function takes model logits and a binary target and returns a per-sample tensor
(length = batch size), so callers can average over a batch or collect per-image values.
"""

from __future__ import annotations

import torch

EPS = 1e-7


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Dice similarity coefficient per sample: 2|P n G| / (|P| + |G|)."""
    pred = _binarize(logits, threshold)
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dims)
    cardinality = pred.sum(dims) + target.sum(dims)
    return (2 * intersection + EPS) / (cardinality + EPS)


def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Intersection over union (Jaccard) per sample: |P n G| / |P u G|."""
    pred = _binarize(logits, threshold)
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - intersection
    return (intersection + EPS) / (union + EPS)
