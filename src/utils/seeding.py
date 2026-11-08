"""Reproducibility helpers.

A single fixed seed is used across Python, NumPy, and PyTorch so that every model in the
comparison sees the same data order and the same augmentation draws. The seed value is
stored in ``configs/base.yaml`` and threaded through from there.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random source used in the project.

    Parameters
    ----------
    seed:
        The shared seed value (see ``configs/base.yaml``).
    deterministic:
        When True, ask cuDNN for deterministic kernels. This makes runs repeatable at a
        small speed cost, which is the right trade-off for a controlled comparison.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so each worker is seeded deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
