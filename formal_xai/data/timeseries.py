"""Time-series dataset loaders.

Design principle: time-series data is treated as single-channel image-like
tensors ``(1, T)`` or ``(1, H, W)`` for uniform backend compatibility.
This allows the same verification backends (NNV, Marabou) to handle
time-series data without special-casing.

Provides:
- ``TaxiNetDataset``: Concrete loader for the TaxiNet runway regression task.
- ``TimeSeriesDataset``: Extensible base for generic 1-D signal data.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Generic time-series dataset.

    Each sample is a 1-D signal of length ``T``. By default the sample is
    returned as a ``(1, T)`` tensor (single channel), making it compatible
    with the same verification pipeline used for images.

    Sub-class this for specific time-series datasets. Override
    ``_load_data`` to customize loading.

    Args:
        signals: Array of shape ``(N, T)`` — *N* samples, each of length *T*.
        labels: Array of shape ``(N,)`` or ``(N, K)`` for regression/multi-output.
        transform: Optional callable applied to each sample tensor.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        self.signals = signals.astype(np.float32)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.signals[idx])
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (1, T) — single-channel
        y = self.labels[idx]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y

    @classmethod
    def from_csv(
        cls,
        path: str,
        label_columns: list[str] | str = "label",
        transform=None,
    ) -> "TimeSeriesDataset":
        """Load from a CSV where some columns are labels and the rest are the signal.

        Args:
            path: Path to CSV.
            label_columns: Column name(s) for labels/targets.
            transform: Optional transform.
        """
        import pandas as pd

        df = pd.read_csv(path)
        if isinstance(label_columns, str):
            label_columns = [label_columns]
        labels = df[label_columns].values
        if labels.shape[1] == 1:
            labels = labels.squeeze(1)
        signals = df.drop(columns=label_columns).values
        return cls(signals, labels, transform=transform)


# ---------------------------------------------------------------------------
# TaxiNet
# ---------------------------------------------------------------------------

class TaxiNetDataset(Dataset):
    """TaxiNet runway regression dataset.

    Each sample is a grayscale image (27×54) with a continuous label
    (crosstrack position). Input is returned as ``(1, 27, 54)`` — single
    channel, compatible with both image & time-series pipelines.

    Args:
        root: Path to the TaxiNet data directory.
        split: ``"train"`` or ``"val"``.
        transform: Optional transform applied to the image tensor.
    """

    def __init__(
        self,
        root: str = "./data/TaxiNet/afternoon/",
        split: str = "train",
        transform=None,
    ):
        import pandas as pd
        import PIL.Image

        if split == "train":
            split_dir = "afternoon_train"
        elif split == "val":
            split_dir = "afternoon_val"
        else:
            split_dir = split

        self.data_dir = os.path.join(root, split_dir)
        self.transform = transform
        self._PIL = PIL.Image

        self.df = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        self.imgs = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith(".png")
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, float]:
        img_path = os.path.join(self.data_dir, self.imgs[idx])
        img = self._PIL.Image.open(img_path).convert("L")
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.unsqueeze(0)  # (1, H, W)
        label = float(self.df.iloc[idx, 1])
        if self.transform:
            img = self.transform(img)
        return img, label
