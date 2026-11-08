"""Split integrity: the official train/val/test splits must not share any ISIC ids."""

import pytest

from src.datasets.isic import SPLITS, ISICDataset

from .conftest import REAL_ROOT


def _id_sets(root):
    return {split: set(ISICDataset(root, split).ids) for split in SPLITS}


def test_no_id_overlap_fixtures(fixture_root):
    ids = _id_sets(fixture_root)
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


@pytest.mark.skipif(
    not (REAL_ROOT / "train" / "images").exists(),
    reason="real ISIC 2018 data not present under data/raw",
)
def test_no_id_overlap_real():
    ids = _id_sets(REAL_ROOT)
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    # The official per-split counts, as stated by the challenge.
    assert len(ids["train"]) == 2594
    assert len(ids["val"]) == 100
    assert len(ids["test"]) == 1000
