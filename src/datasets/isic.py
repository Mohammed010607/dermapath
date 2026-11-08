"""ISIC 2018 Task 1 dataset.

The official challenge splits are used as-is: the training, validation, and test sets are
the separate archives published by the challenge, already laid out on disk as
``data/raw/<split>/{images,masks}/`` by ``scripts/download_isic2018.py``. No re-splitting is
done, so there is no risk of leakage between splits.

Each item is a dict with:
    image : float32 tensor (3, 224, 224), ImageNet-normalised
    label : float32 tensor (1, 224, 224), binary {0, 1}
    id    : the ISIC identifier, e.g. "ISIC_0000000"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.seeding import seed_worker
from .transforms import build_eval_transforms, build_train_transforms

MASK_SUFFIX = "_segmentation"
SPLITS = ("train", "val", "test")


class ISICDataset(Dataset):
    """Image/mask pairs for one official ISIC 2018 Task 1 split."""

    def __init__(self, root, split: str, transform=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.img_dir = self.root / split / "images"
        self.msk_dir = self.root / split / "masks"

        # Keep only images that have a matching mask; store the sorted ISIC ids.
        ids = []
        for img in sorted(self.img_dir.glob("*.jpg")):
            if (self.msk_dir / f"{img.stem}{MASK_SUFFIX}.png").exists():
                ids.append(img.stem)
        if not ids:
            raise FileNotFoundError(f"no image/mask pairs found under {self.root / split}")
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def image_path(self, stem: str) -> Path:
        return self.img_dir / f"{stem}.jpg"

    def mask_path(self, stem: str) -> Path:
        return self.msk_dir / f"{stem}{MASK_SUFFIX}.png"

    def __getitem__(self, index: int):
        stem = self.ids[index]

        # Image: RGB -> channel-first float in [0, 1].
        image = np.asarray(Image.open(self.image_path(stem)).convert("RGB"), dtype=np.float32)
        image = np.transpose(image, (2, 0, 1)) / 255.0

        # Mask: grayscale -> channel-first binary {0, 1}.
        mask = np.asarray(Image.open(self.mask_path(stem)).convert("L"))
        mask = (mask > 127).astype(np.float32)[None, ...]

        sample = {"image": image, "label": mask, "id": stem}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_dataset(root, split: str, seed: int | None = None) -> ISICDataset:
    """Create a dataset with the correct transform for the split.

    Training uses the augmenting pipeline; validation and test use the deterministic one,
    which guarantees no augmentation leaks into evaluation.
    """
    if split == "train":
        transform = build_train_transforms(seed=seed)
    else:
        transform = build_eval_transforms()
    return ISICDataset(root, split, transform=transform)


def build_dataloader(dataset: ISICDataset, batch_size: int, shuffle: bool,
                     num_workers: int = 4, seed: int | None = None) -> DataLoader:
    """Wrap a dataset in a DataLoader with deterministic worker seeding."""
    generator = None
    if seed is not None:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=shuffle,  # drop the ragged last batch only while training
    )
