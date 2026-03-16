"""Tabular dataset loaders.

Provides a generic ``TabularDataset`` that wraps NumPy arrays or CSV files,
plus a concrete HELOC loader.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Generic tabular dataset.

    Each sample is a 1-D feature vector (optionally reshaped to ``(1, D)``
    for consistency with image pipelines where ``channel=1``).

    Args:
        features: Feature matrix ``(N, D)`` as NumPy array.
        labels: Label vector ``(N,)`` as NumPy array.
        as_image: If ``True``, reshape each sample to ``(1, D)`` so it
            behaves like a single-channel "image" for backend compatibility.
        transform: Optional callable applied to each sample tensor.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        as_image: bool = False,
        transform=None,
    ):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.as_image = as_image
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.features[idx])
        if self.as_image:
            x = x.unsqueeze(0)  # (1, D) — single-channel "image"
        if self.transform:
            x = self.transform(x)
        return x, int(self.labels[idx])

    @classmethod
    def from_csv(
        cls,
        path: str,
        label_column: str = "label",
        as_image: bool = False,
        transform=None,
    ) -> "TabularDataset":
        """Load from a CSV file where one column is the label.

        Args:
            path: Path to the CSV file.
            label_column: Name of the label column.
            as_image: If ``True``, reshape samples to ``(1, D)``.
            transform: Optional transform.
        """
        import pandas as pd

        df = pd.read_csv(path)
        labels = df[label_column].values
        features = df.drop(columns=[label_column]).values
        return cls(features, labels, as_image=as_image, transform=transform)


class HELOCDataset(TabularDataset):
    """HELOC (Home Equity Line of Credit) dataset loader.

    Expects a directory with ``heloc.csv`` or similar structure.
    Wraps ``TabularDataset`` for convenience.
    """

    def __init__(
        self,
        root: str = "./data/heloc",
        split: str = "test",
        as_image: bool = False,
        transform=None,
    ):
        csv_path = os.path.join(root, f"heloc_{split}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(root, "heloc.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"HELOC data not found at {root}. "
                "Please place heloc.csv in the data directory."
            )
        import pandas as pd

        df = pd.read_csv(csv_path)
        label_col = "RiskPerformance" if "RiskPerformance" in df.columns else df.columns[-1]
        labels = df[label_col].values
        features = df.drop(columns=[label_col]).values

        super().__init__(features, labels, as_image=as_image, transform=transform)
