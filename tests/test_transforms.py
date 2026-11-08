"""Augmentation behaviour: train augments, val/test do not, masks stay binary."""

import torch

from src.datasets import ISICDataset, build_dataset
from src.datasets.transforms import build_eval_transforms, build_train_transforms


def test_eval_is_deterministic(fixture_root):
    """Validation/test preprocessing must be repeatable (no augmentation)."""
    ds = ISICDataset(fixture_root, "val", transform=build_eval_transforms())
    first = torch.as_tensor(ds[0]["image"])
    second = torch.as_tensor(ds[0]["image"])
    assert torch.allclose(first, second)


def test_no_aug_leak_into_val(fixture_root):
    """The val split, built through the public factory, must be deterministic."""
    ds = build_dataset(fixture_root, "val", seed=42)
    a = torch.as_tensor(ds[0]["image"])
    b = torch.as_tensor(ds[0]["image"])
    assert torch.allclose(a, b)


def test_no_aug_leak_into_test(fixture_root):
    ds = build_dataset(fixture_root, "test", seed=42)
    a = torch.as_tensor(ds[0]["image"])
    b = torch.as_tensor(ds[0]["image"])
    assert torch.allclose(a, b)


def test_train_augmentation_varies(fixture_root):
    """Training augmentation must actually perturb the image across draws."""
    ds = ISICDataset(fixture_root, "train", transform=build_train_transforms(seed=0))
    draws = [torch.as_tensor(ds[0]["image"]) for _ in range(5)]
    assert any(not torch.allclose(draws[0], d) for d in draws[1:])


def test_train_mask_stays_binary(fixture_root):
    """Geometric augmentation uses nearest-neighbour, so masks remain binary."""
    ds = ISICDataset(fixture_root, "train", transform=build_train_transforms(seed=0))
    for _ in range(10):
        values = set(torch.unique(torch.as_tensor(ds[0]["label"])).tolist())
        assert values <= {0.0, 1.0}, f"mask has non-binary values after aug: {values}"
