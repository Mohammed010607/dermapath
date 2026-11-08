"""Save and load model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch: int, best_metric: float, config: dict) -> None:
    """Write a checkpoint containing weights, optimiser state, and the run config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict:
    """Restore weights (and optionally optimiser state) from a checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
