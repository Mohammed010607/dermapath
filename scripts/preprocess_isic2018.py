"""Cache the dataset at the model input size for fast, GPU-bound training.

The raw ISIC images range up to ~27 megapixels, so decoding and resizing them on every
epoch makes training data-loading-bound. This script resizes each image/mask to 224x224
once and writes the result to ``data/processed/<split>/{images,masks}/`` as PNG. Training
then reads these small files instead of the originals.

This is the same resize the on-the-fly pipeline performs as its first step (bilinear for
the image, nearest-neighbour for the mask), just computed ahead of time, so the experimental
conditions are unchanged. The cache is rebuilt only for files that are missing.

Run from the repository root:
    python scripts/preprocess_isic2018.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SPLITS = ("train", "val", "test")
MASK_SUFFIX = "_segmentation"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resize_pair(img_path: Path, mask_path: Path, out_img: Path, out_mask: Path, size: int) -> None:
    if not out_img.exists():
        img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
        img.save(out_img)
    if not out_mask.exists():
        mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
        # Re-threshold so the cached mask is strictly binary {0, 255}.
        mask = mask.point(lambda v: 255 if v > 127 else 0)
        mask.save(out_mask)


def process_split(split: str, raw_dir: Path, out_dir: Path, size: int, workers: int) -> int:
    img_dir = raw_dir / split / "images"
    msk_dir = raw_dir / split / "masks"
    out_img_dir = out_dir / split / "images"
    out_msk_dir = out_dir / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    jobs = []
    for img in images:
        mask = msk_dir / f"{img.stem}{MASK_SUFFIX}.png"
        if not mask.exists():
            continue
        jobs.append((img, mask, out_img_dir / f"{img.stem}.png",
                     out_msk_dir / f"{img.stem}{MASK_SUFFIX}.png"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(resize_pair, *job, size) for job in jobs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"  {split}", leave=False):
            pass
    return len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-resize ISIC 2018 Task 1 to the model input size.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in SPLITS:
        n = process_split(split, args.raw_dir, args.out_dir, args.size, args.workers)
        print(f"{split:5s}: cached {n} pairs at {args.size}x{args.size}")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
