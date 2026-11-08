"""Dataset output contract: shapes, dtypes, binary masks, image/mask pairing."""

import torch

from src.datasets import IMG_SIZE, ISICDataset
from src.datasets.transforms import build_eval_transforms


def test_eval_shapes(fixture_root):
    ds = ISICDataset(fixture_root, "val", transform=build_eval_transforms())
    sample = ds[0]
    assert sample["image"].shape == (3, IMG_SIZE, IMG_SIZE)
    assert sample["label"].shape == (1, IMG_SIZE, IMG_SIZE)


def test_eval_dtype_is_float32(fixture_root):
    ds = ISICDataset(fixture_root, "val", transform=build_eval_transforms())
    sample = ds[0]
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32


def test_mask_is_binary(fixture_root):
    ds = ISICDataset(fixture_root, "val", transform=build_eval_transforms())
    for i in range(len(ds)):
        values = set(torch.unique(torch.as_tensor(ds[i]["label"])).tolist())
        assert values <= {0.0, 1.0}, f"mask has non-binary values: {values}"


def test_pairing_and_ids(fixture_root):
    ds = ISICDataset(fixture_root, "train")
    assert len(ds) == 3
    assert ds.ids == sorted(ds.ids)
    # Every retained id must have both an image and a mask file.
    for stem in ds.ids:
        assert ds.image_path(stem).exists()
        assert ds.mask_path(stem).exists()


def test_id_is_preserved(fixture_root):
    ds = ISICDataset(fixture_root, "val", transform=build_eval_transforms())
    assert ds[0]["id"] == ds.ids[0]
