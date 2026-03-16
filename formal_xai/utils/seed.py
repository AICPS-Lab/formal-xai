"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed(sd: int = 0, cudnn: bool = False, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Args:
        sd: Seed value.
        cudnn: Whether to set CUDNN benchmark mode.
        deterministic: Whether to force deterministic CUDNN algorithms.
    """
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.benchmark = cudnn
    torch.backends.cudnn.deterministic = deterministic
