"""Download the ImageNet pretrained weights for TransUNet and Swin-UNet, and confirm they
load into the models.

Sources (verified against the official repositories):
  * TransUNet encoder - R50-ViT-B_16, ImageNet-21k, from the Google ViT models bucket named
    in the TransUNet README:
    https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
  * Swin-UNet encoder/decoder - Swin-Tiny, ImageNet-1k. The Swin-Unet README distributes a
    copy via Google Drive; the same file is Microsoft's official Swin Transformer release,
    which is used here as a direct, verifiable source:
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

UNet has no pretrained weights (it is the from-scratch CNN baseline).

Run from the repository root:
    python scripts/download_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # allow `import src...` when run as a script
WEIGHTS_DIR = PROJECT_ROOT / "weights"

DOWNLOADS = {
    "transunet": {
        "url": "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz",
        "dest": WEIGHTS_DIR / "imagenet21k" / "R50+ViT-B_16.npz",
    },
    "swin_unet": {
        "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        "dest": WEIGHTS_DIR / "swin_tiny_patch4_window7_224.pth",
    },
}


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with resume support and a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"  already complete: {dest.name} ({total/1024**2:.0f} MB)")
        return

    headers, mode = {}, "wb"
    if 0 < have < total:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if mode == "ab" and r.status_code == 200:
            mode, have = "wb", 0
        bar = tqdm(total=total or None, initial=have, unit="B", unit_scale=True,
                   unit_divisor=1024, desc=f"  {dest.name}", leave=False)
        with open(dest, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        bar.close()


def verify_transunet(path: Path) -> None:
    import numpy as np

    from src.models.transunet import build_transunet

    model = build_transunet(img_size=224, num_classes=1)
    model.load_from(np.load(str(path)))
    print(f"  TransUNet: loaded {path.name} OK "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f} M params)")


def verify_swin_unet(path: Path) -> None:
    from src.models.swin_unet import build_swin_unet, load_swin_pretrained

    model = build_swin_unet(img_size=224, num_classes=1)
    msg = load_swin_pretrained(model, path)
    # Report how many keys stayed uninitialised (expected: only the final seg head etc.).
    missing = len(getattr(msg, "missing_keys", [])) if hasattr(msg, "missing_keys") else "?"
    print(f"  Swin-UNet: loaded {path.name} OK "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f} M params); "
          f"load report: {msg[:120]}...")


def main() -> None:
    for name, item in DOWNLOADS.items():
        print(f"downloading {name} weights ...")
        download(item["url"], item["dest"])

    print("\nverifying the weights load into the models ...")
    verify_transunet(DOWNLOADS["transunet"]["dest"])
    verify_swin_unet(DOWNLOADS["swin_unet"]["dest"])
    print("\ndone.")


if __name__ == "__main__":
    main()
