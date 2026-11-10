"""Checks for the five segmentation metrics on known inputs."""

import torch

from src.evaluation.metrics import compute_metrics


def _make_case():
    # A single 1x16x16 target with a central 8x8 lesion square.
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    strong = (target * 2 - 1) * 10  # +10 on lesion, -10 on background -> near-perfect logits
    return target, strong


def test_perfect_prediction():
    target, logits = _make_case()
    m = compute_metrics(logits, target)
    for key in ["dsc", "iou", "pixel_accuracy", "sensitivity", "specificity"]:
        assert m[key].item() > 0.99, f"{key} should be ~1 for a perfect prediction"


def test_inverted_prediction():
    target, logits = _make_case()
    m = compute_metrics(-logits, target)  # predict the exact opposite
    assert m["dsc"].item() < 0.01
    assert m["iou"].item() < 0.01
    assert m["sensitivity"].item() < 0.01   # no true lesion pixels found
    assert m["specificity"].item() < 0.01   # all background predicted as lesion


def test_all_background_prediction():
    target, _ = _make_case()
    logits = torch.full_like(target, -10.0)  # predict background everywhere
    m = compute_metrics(logits, target)
    assert m["sensitivity"].item() < 0.01            # found no lesion pixels
    assert m["specificity"].item() > 0.99            # kept all background
    # 64 of 256 pixels are lesion, so pixel accuracy is the 192/256 background fraction.
    assert abs(m["pixel_accuracy"].item() - 0.75) < 0.01


def test_returns_per_sample():
    target = torch.zeros(3, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    logits = (target * 2 - 1) * 10
    m = compute_metrics(logits, target)
    assert m["dsc"].shape == (3,)
