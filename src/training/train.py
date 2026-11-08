"""Shared, config-driven training entry point for every model in the comparison.

The same harness trains UNet, TransUNet, and Swin-UNet; only the ``--config`` differs, which
is what makes the comparison fair. It implements the thesis methodology: a combined Dice+BCE
loss, the Adam optimiser at 1e-4, a cosine learning-rate schedule, AMP mixed precision,
early stopping on validation Dice, best-checkpoint saving, and per-epoch CSV logging.

Usage:
    python -m src.training.train --config configs/unet.yaml
    python -m src.training.train --config configs/unet.yaml --find-max-batch-size
    python -m src.training.train --config configs/unet.yaml --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from monai.data.meta_obj import set_track_meta
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# The transforms produce MONAI MetaTensors; metadata tracking is not needed once a batch
# reaches the model, and disabling it removes per-operation overhead during forward/backward.
set_track_meta(False)

from src.datasets import build_dataloader, build_dataset
from src.evaluation.metrics import dice_coefficient
from src.models import build_model
from src.training.losses import DiceBCELoss
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import CSVLogger
from src.utils.seeding import set_seed


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, max_batches=None):
    model.train()
    running, seen = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, max_batches=None):
    model.eval()
    loss_sum, dsc_sum, seen = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        loss_sum += loss.item() * images.size(0)
        dsc_sum += dice_coefficient(logits.float(), masks).sum().item()
        seen += images.size(0)
    return loss_sum / max(seen, 1), dsc_sum / max(seen, 1)


def find_max_batch_size(config, device, candidates):
    """Probe increasing batch sizes with a real training step until CUDA runs out of memory.

    Batches are synthetic tensors of the true training shape (B, 3, 224, 224); memory use
    depends on shape, not pixel content, so this measures the real footprint quickly.
    """
    criterion = DiceBCELoss().to(device)
    results, best = [], 0
    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(config).to(device)
            optimizer = Adam(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=True)

            images = torch.randn(bs, 3, 224, 224, device=device)
            masks = torch.randint(0, 2, (bs, 1, 224, 224), device=device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            peak = torch.cuda.max_memory_allocated() / 1024**3
            results.append((bs, peak, "ok"))
            best = bs
            del model, optimizer, images, masks, loss, scaler
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                results.append((bs, float("nan"), "OOM"))
                torch.cuda.empty_cache()
                break
            raise
    return best, results


def parse_args():
    p = argparse.ArgumentParser(description="Train a segmentation model from a YAML config.")
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override config batch size")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None, help="limit batches/epoch (smoke)")
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--find-max-batch-size", action="store_true",
                   help="probe the largest batch size that fits, then exit")
    p.add_argument("--no-wandb", action="store_true", help="disable W&B even if the config enables it")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    set_seed(config["seed"], config.get("deterministic", True))
    device = get_device()
    name = config["model"]["name"]
    print(f"model={name}  device={device}")

    if args.find_max_batch_size:
        best, results = find_max_batch_size(config, device, [4, 8, 16, 24, 32, 48, 64])
        print("\nbatch-size probe (one AMP train step each):")
        for bs, peak, status in results:
            line = f"  bs={bs:>3}  peak={peak:5.2f} GB  OK" if status == "ok" else f"  bs={bs:>3}  OOM"
            print(line)
        print(f"\nmax batch size that fits: {best}")
        return

    root = config["data"]["root"]
    seed = config["seed"]
    workers = config["data"]["num_workers"]
    batch_size = config["training"]["batch_size"]

    train_ds = build_dataset(root, "train", seed=seed)
    val_ds = build_dataset(root, "val", seed=seed)
    train_loader = build_dataloader(train_ds, batch_size, shuffle=True, num_workers=workers, seed=seed)
    val_loader = build_dataloader(val_ds, batch_size, shuffle=False, num_workers=workers, seed=seed)
    print(f"train={len(train_ds)}  val={len(val_ds)}  batch_size={batch_size}  workers={workers}")

    model = build_model(config).to(device)
    criterion = DiceBCELoss(bce_weight=0.5).to(device)

    tr = config["training"]
    epochs = tr["epochs"]
    optimizer = Adam(model.parameters(), lr=float(tr["learning_rate"]),
                     weight_decay=float(tr.get("weight_decay", 0.0)))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(tr.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_path = Path("checkpoints") / name / "best.pt"
    logger = CSVLogger(
        Path("results") / "metrics" / f"{name}.csv",
        ["epoch", "train_loss", "val_loss", "val_dsc", "lr", "epoch_time_s", "peak_vram_gb"],
    )

    use_wandb = config.get("logging", {}).get("use_wandb", False) and not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project="dermapath", name=name, config=config)

    es = tr.get("early_stopping", {})
    patience = int(es.get("patience", 15))
    best_metric, no_improve = -float("inf"), 0

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                     device, amp, args.max_train_batches)
        val_loss, val_dsc = evaluate(model, val_loader, criterion, device, amp, args.max_val_batches)
        scheduler.step()

        elapsed = time.time() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:>3}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val_dsc {val_dsc:.4f}  lr {lr:.2e}  {elapsed:5.1f}s  {peak:4.2f} GB")

        logger.log({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6), "val_dsc": round(val_dsc, 6),
                    "lr": lr, "epoch_time_s": round(elapsed, 1), "peak_vram_gb": round(peak, 2)})
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                       "val_dsc": val_dsc, "lr": lr})

        if val_dsc > best_metric:
            best_metric, no_improve = val_dsc, 0
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_metric, config)
            print(f"  new best val_dsc={best_metric:.4f}  ->  {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch} (no val_dsc gain for {patience} epochs)")
                break

    print(f"done. best val_dsc={best_metric:.4f}  checkpoint={ckpt_path}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
