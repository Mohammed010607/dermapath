"""Evaluate a trained segmentation model on a dataset split.

Loads a checkpoint, runs the model over the split (test by default), and reports the five
thesis metrics - Dice (DSC), IoU, pixel accuracy, sensitivity, specificity - as mean +/- std.
Per-image scores are written to results/eval/<model>_per_image.csv, the cross-model summary is
upserted into results/summary_table.csv, and qualitative mask overlays are saved to
results/figures/eval/<model>/.

Usage:
    python -m src.evaluation.evaluate --config configs/unet.yaml
    python -m src.evaluation.evaluate --config configs/unet.yaml --checkpoint checkpoints/unet/best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; write figures to files
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data.meta_obj import set_track_meta

from src.datasets import IMAGENET_MEAN, IMAGENET_STD, build_dataloader, build_dataset
from src.evaluation.metrics import METRIC_FUNCTIONS, compute_metrics
from src.models import build_model
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seeding import set_seed

set_track_meta(False)
METRIC_KEYS = list(METRIC_FUNCTIONS.keys())


def _denormalise(image: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation for display: (3,H,W) normalised -> (H,W,3) in [0,1]."""
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    return np.clip((image * std + mean).transpose(1, 2, 0), 0, 1)


def save_overlays(model, dataset, device, out_dir: Path, n: int, threshold: float = 0.5) -> None:
    """Save image / ground-truth / prediction / overlay panels for the first n samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        image = torch.as_tensor(sample["image"]).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(image)
        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        pred = prob > threshold
        gt = torch.as_tensor(sample["label"])[0].cpu().numpy() > 0.5
        disp = _denormalise(np.asarray(sample["image"]))

        overlay = disp.copy()
        overlay[pred] = 0.5 * overlay[pred] + 0.5 * np.array([1.0, 0.0, 0.0])

        fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
        for a, panel, title, cmap in [
            (ax[0], disp, "Image", None),
            (ax[1], gt, "Ground truth", "gray"),
            (ax[2], pred, "Prediction", "gray"),
            (ax[3], overlay, "Overlay", None),
        ]:
            a.imshow(panel, cmap=cmap)
            a.set_title(title)
            a.axis("off")
        fig.suptitle(sample["id"])
        fig.tight_layout()
        fig.savefig(out_dir / f"{sample['id']}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def update_summary_table(path: Path, model_name: str, summary: dict) -> None:
    """Upsert this model's row (mean and std per metric) into the shared summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model"] + [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]

    rows = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[row["model"]] = row

    row = {"model": model_name}
    for k in METRIC_KEYS:
        row[f"{k}_mean"] = f"{summary[k][0]:.4f}"
        row[f"{k}_std"] = f"{summary[k][1]:.4f}"
    rows[model_name] = row

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="defaults to checkpoints/<model>/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-overlays", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    name = config["model"]["name"]
    set_seed(config["seed"], config.get("deterministic", True))
    device = get_device()

    model = build_model(config).to(device)
    ckpt_path = args.checkpoint or f"checkpoints/{name}/best.pt"
    ckpt = load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    print(f"model={name}  device={device}  checkpoint={ckpt_path}  "
          f"(trained to epoch {ckpt.get('epoch', '?')}, best val_dsc {ckpt.get('best_metric', float('nan')):.4f})")

    dataset = build_dataset(config["data"]["root"], args.split, seed=config["seed"])
    loader = build_dataloader(dataset, config["training"]["batch_size"], shuffle=False,
                              num_workers=0, seed=config["seed"])
    print(f"evaluating on {args.split} split: {len(dataset)} images")

    results = {k: [] for k in METRIC_KEYS}
    ids = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            per_sample = compute_metrics(logits.float(), masks)
            for k in METRIC_KEYS:
                results[k] += per_sample[k].cpu().tolist()
            ids += list(batch["id"])

    # Per-image CSV.
    eval_dir = args.output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = eval_dir / f"{name}_per_image.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id"] + METRIC_KEYS)
        for i, image_id in enumerate(ids):
            writer.writerow([image_id] + [f"{results[k][i]:.6f}" for k in METRIC_KEYS])

    # Summary mean +/- std.
    summary = {k: (float(np.mean(results[k])), float(np.std(results[k]))) for k in METRIC_KEYS}
    print(f"\n{args.split} results for {name} ({len(ids)} images):")
    for k in METRIC_KEYS:
        print(f"  {k:16s}: {summary[k][0]:.4f} +/- {summary[k][1]:.4f}")
    update_summary_table(args.output_dir / "summary_table.csv", name, summary)

    # Qualitative overlays.
    save_overlays(model, dataset, device, args.output_dir / "figures" / "eval" / name, args.num_overlays)
    print(f"\nper-image -> {per_image_path}")
    print(f"summary   -> {args.output_dir / 'summary_table.csv'}")
    print(f"overlays  -> {args.output_dir / 'figures' / 'eval' / name}")


if __name__ == "__main__":
    main()
