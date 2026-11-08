"""Download and verify the ISIC 2018 Task 1 (lesion boundary segmentation) dataset.

The six official archives are pulled from the ISIC challenge data host, unzipped into
``data/raw/<split>/{images,masks}``, and checked against the official per-split counts.

Source of the URLs and counts: the official challenge data page
(https://challenge.isic-archive.com/data). Training = 2594 images, Test = 1000 images are
stated there; the validation count (100) is confirmed from the downloaded archive itself.

Run from the repository root:
    python scripts/download_isic2018.py

Useful flags:
    --skip-download     only (re)extract and verify archives already present
    --clean-zips        delete the downloaded .zip files after successful extraction
    --workers N         parallelism for the image-integrity check (default 8)
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# --- Official source -------------------------------------------------------------------

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018"

# Logical name -> archive file name. The input images are shared by Task 1 and Task 2,
# hence the "Task1-2" prefix; the boundary masks are the "Task1" ground truth.
ARCHIVES = {
    ("train", "images"): "ISIC2018_Task1-2_Training_Input.zip",
    ("train", "masks"): "ISIC2018_Task1_Training_GroundTruth.zip",
    ("val", "images"): "ISIC2018_Task1-2_Validation_Input.zip",
    ("val", "masks"): "ISIC2018_Task1_Validation_GroundTruth.zip",
    ("test", "images"): "ISIC2018_Task1-2_Test_Input.zip",
    ("test", "masks"): "ISIC2018_Task1_Test_GroundTruth.zip",
}

# Expected image count per split, per the official challenge page.
EXPECTED_COUNTS = {"train": 2594, "val": 100, "test": 1000}

SPLITS = ("train", "val", "test")
MASK_SUFFIX = "_segmentation"  # masks are named ISIC_XXXXXXX_segmentation.png

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- Download --------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with resume support and a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    have = dest.stat().st_size if dest.exists() else 0

    if total and have == total:
        print(f"  already complete: {dest.name} ({total/1024**3:.2f} GB)")
        return

    headers, mode = {}, "wb"
    if 0 < have < total:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        # If the server ignored our Range request it replies 200 (not 206); start over
        # so we never append onto a partial file.
        if mode == "ab" and r.status_code == 200:
            mode, have = "wb", 0
        bar = tqdm(
            total=total or None, initial=have, unit="B", unit_scale=True,
            unit_divisor=1024, desc=f"  {dest.name}", leave=False,
        )
        with open(dest, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        bar.close()


# --- Extraction ------------------------------------------------------------------------

def extract(zip_path: Path, split: str, kind: str, raw_dir: Path) -> None:
    """Unzip an archive and lay its images/masks out under ``<split>/<kind>/``."""
    target = raw_dir / split / kind
    target.mkdir(parents=True, exist_ok=True)

    wanted = (".jpg", ".jpeg", ".png")
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist()
            if not m.endswith("/") and Path(m).suffix.lower() in wanted
        ]
        for m in tqdm(members, desc=f"  extract {split}/{kind}", leave=False):
            # Flatten: write every image straight into the target folder by its base name.
            name = Path(m).name
            out = target / name
            if out.exists():
                continue
            with zf.open(m) as src, open(out, "wb") as dst:
                dst.write(src.read())


# --- Verification helpers --------------------------------------------------------------

def stem_of(path: Path) -> str:
    """ISIC id for an image or mask, e.g. ISIC_0000000 (mask suffix stripped)."""
    name = path.stem
    return name[: -len(MASK_SUFFIX)] if name.endswith(MASK_SUFFIX) else name


def check_integrity(path: Path) -> str | None:
    """Return an error string if the file cannot be opened/decoded, else None."""
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncated / corrupt files without full decode
        return None
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        return f"{path.name}: {exc}"


def mask_values(path: Path) -> set[int]:
    """Unique pixel values in a mask (expected to be the binary set {0, 255})."""
    with Image.open(path) as im:
        return set(im.convert("L").getdata())


# --- Verification ----------------------------------------------------------------------

def verify(raw_dir: Path, workers: int) -> bool:
    """Run every check and print a report. Return True if the dataset is clean."""
    lines: list[str] = []
    ok = True

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 72)
    say("ISIC 2018 Task 1 - verification report")
    say("=" * 72)

    all_images: list[Path] = []
    all_masks: list[Path] = []

    for split in SPLITS:
        img_dir = raw_dir / split / "images"
        msk_dir = raw_dir / split / "masks"
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        masks = sorted(msk_dir.glob(f"*{MASK_SUFFIX}.png"))
        all_images += images
        all_masks += masks

        expected = EXPECTED_COUNTS[split]
        count_ok = len(images) == expected and len(masks) == expected

        say()
        say(f"[{split}]")
        say(f"  images: {len(images):>5}  (expected {expected})")
        say(f"  masks : {len(masks):>5}  (expected {expected})")

        # Pairing: every image has exactly one mask and vice versa.
        img_ids = {stem_of(p) for p in images}
        msk_ids = {stem_of(p) for p in masks}
        img_no_mask = sorted(img_ids - msk_ids)
        msk_no_img = sorted(msk_ids - img_ids)
        pairing_ok = not img_no_mask and not msk_no_img

        say(f"  pairing: {'OK' if pairing_ok else 'MISMATCH'}")
        if img_no_mask:
            say(f"    images without a mask ({len(img_no_mask)}): {img_no_mask[:5]} ...")
        if msk_no_img:
            say(f"    masks without an image ({len(msk_no_img)}): {msk_no_img[:5]} ...")

        ok = ok and count_ok and pairing_ok

    # --- Image resolution range (header read only, fast) -------------------------------
    say()
    say("[resolution] (across all images)")
    widths, heights = [], []
    for p in tqdm(all_images, desc="  sizing", leave=False):
        try:
            with Image.open(p) as im:
                w, h = im.size
            widths.append(w)
            heights.append(h)
        except Exception as exc:  # noqa: BLE001
            say(f"  could not read size of {p.name}: {exc}")
            ok = False
    if widths:
        say(f"  width : min {min(widths)}  max {max(widths)}")
        say(f"  height: min {min(heights)}  max {max(heights)}")

    # --- Mask value range --------------------------------------------------------------
    say()
    say("[mask values] (expected binary {0, 255})")
    seen: set[int] = set()
    non_binary: list[str] = []
    for p in tqdm(all_masks, desc="  masks", leave=False):
        vals = mask_values(p)
        seen |= vals
        if not vals <= {0, 255}:
            non_binary.append(p.name)
    say(f"  distinct values seen: {sorted(seen)}")
    if non_binary:
        say(f"  NON-BINARY masks ({len(non_binary)}): {non_binary[:5]} ...")
        ok = False

    # --- Corrupt / truncated files -----------------------------------------------------
    say()
    say("[integrity] (decoding every image and mask)")
    bad: list[str] = []
    everything = all_images + all_masks
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_integrity, p): p for p in everything}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  checking", leave=False):
            err = fut.result()
            if err:
                bad.append(err)
    if bad:
        say(f"  CORRUPT files ({len(bad)}):")
        for b in bad[:10]:
            say(f"    {b}")
        ok = False
    else:
        say(f"  all {len(everything)} files decoded cleanly")

    say()
    say("=" * 72)
    say(f"RESULT: {'PASS - dataset is complete and clean' if ok else 'FAIL - see issues above'}")
    say("=" * 72)

    # Keep a copy of the report next to the data (gitignored).
    report = raw_dir / "verification_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written to {report}")
    return ok


# --- Entry point -----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify ISIC 2018 Task 1.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--skip-download", action="store_true",
                        help="only (re)extract and verify archives already present")
    parser.add_argument("--clean-zips", action="store_true",
                        help="delete the .zip files after successful extraction")
    parser.add_argument("--workers", type=int, default=8,
                        help="threads for the image-integrity check")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    zip_dir = raw_dir / "_zips"

    for (split, kind), fname in ARCHIVES.items():
        url = f"{BASE_URL}/{fname}"
        zpath = zip_dir / fname

        if not args.skip_download:
            print(f"downloading {fname} ...")
            download(url, zpath)

        if not zpath.exists():
            print(f"ERROR: {zpath} is missing; run without --skip-download first.")
            return 2

        print(f"extracting {fname} -> {split}/{kind} ...")
        extract(zpath, split, kind, raw_dir)

    if args.clean_zips:
        for fname in ARCHIVES.values():
            (zip_dir / fname).unlink(missing_ok=True)
        print("removed downloaded .zip files")

    print()
    clean = verify(raw_dir, args.workers)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
