"""Segmentation metrics.

Provides the five metrics reported in the thesis: Dice (DSC), IoU (Jaccard), pixel accuracy,
sensitivity, and specificity. Dice is also used for validation monitoring during training.

Each function takes model logits and a binary target and returns a per-sample tensor
(length = batch size), so callers can average over a batch or collect per-image values.
``compute_metrics`` returns all five at once.
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


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Fraction of correctly classified pixels per sample."""
    pred = _binarize(logits, threshold)
    dims = (1, 2, 3)
    correct = (pred == target).float().sum(dims)
    total = target.shape[1] * target.shape[2] * target.shape[3]
    return correct / total


def sensitivity(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Sensitivity / recall per sample: TP / (TP + FN) - of true lesion pixels, fraction found."""
    pred = _binarize(logits, threshold)
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    return (tp + EPS) / (tp + fn + EPS)


def specificity(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Specificity per sample: TN / (TN + FP) - of true background pixels, fraction kept."""
    pred = _binarize(logits, threshold)
    dims = (1, 2, 3)
    tn = ((1 - pred) * (1 - target)).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    return (tn + EPS) / (tn + fp + EPS)


# The five thesis metrics, keyed by the names used in the result files.
METRIC_FUNCTIONS = {
    "dsc": dice_coefficient,
    "iou": iou_score,
    "pixel_accuracy": pixel_accuracy,
    "sensitivity": sensitivity,
    "specificity": specificity,
}


def compute_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Return all five metrics as per-sample tensors, keyed by metric name."""
    return {name: fn(logits, target, threshold) for name, fn in METRIC_FUNCTIONS.items()}
