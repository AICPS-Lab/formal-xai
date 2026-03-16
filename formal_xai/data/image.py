"""Image dataset loaders and sampling utilities."""

from __future__ import annotations

import gzip
import os
import random
import struct
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def process_dataset_gtrsb(trainset) -> list:
    """Filter GTSRB to the 9-class subset used in experiments.

    The selected classes are: 0, 1, 2, 3, 4, 5, 7, 8, 14.
    Labels are remapped to 0..8.
    """
    selected = [0, 1, 2, 3, 4, 5, 7, 8, 14]
    remap = {c: i for i, c in enumerate(selected)}
    filtered = [(img, remap[lab]) for img, lab in trainset if lab in selected]
    return filtered


class PermuteTransform:
    """Callable that permutes tensor dimensions (used for EMNIST)."""

    def __init__(self, order):
        self.order = order

    def __call__(self, x):
        return x.permute(*self.order)


# ---------------------------------------------------------------------------
# Custom dataset classes
# ---------------------------------------------------------------------------

class SimpleEMNIST(Dataset):
    """Minimal EMNIST loader that reads raw IDX files from disk."""

    def __init__(self, root_dir: str, split: str = "mnist", train: bool = True, transform=None):
        self.root_dir = os.path.join(root_dir, "EMNIST", "raw")
        self.split = split.lower()
        self.train = train
        self.transform = transform

        self._split_map = {
            "byclass": ("byclass", "byclass"),
            "bymerge": ("bymerge", "bymerge"),
            "balanced": ("balanced", "balanced"),
            "letters": ("letters", "letters"),
            "digits": ("digits", "digits"),
            "mnist": ("train", "t10k"),
        }
        self.images, self.labels = self._load_data()

    def _load_data(self):
        phase_map = self._split_map.get(self.split, ("train", "t10k"))
        phase = phase_map[0] if self.train else phase_map[1]
        prefix = f"emnist-{self.split}-{phase}"

        img_path = os.path.join(self.root_dir, f"{prefix}-images-idx3-ubyte.gz")
        lbl_path = os.path.join(self.root_dir, f"{prefix}-labels-idx1-ubyte.gz")

        with gzip.open(lbl_path, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

        with gzip.open(img_path, "rb") as f:
            _, n, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).copy()

        if self.split == "letters":
            labels = labels - 1  # 1-indexed → 0-indexed

        return images, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def get_sample_by_class(ds, class_label: int):
    """Return the first sample from *ds* with the given class."""
    for sample, label in ds:
        if label == class_label:
            return sample.unsqueeze(0).requires_grad_(True)
    raise ValueError(f"No sample found for class {class_label}")


def randomly_select_sample_by_class(ds: Dataset, class_label: int, seed_v: int = 0):
    """Return a random sample with the given class."""
    candidates = [(i, s, l) for i, (s, l) in enumerate(ds) if l == class_label]
    if not candidates:
        raise ValueError(f"No samples for class {class_label}")
    rng = random.Random(seed_v)
    _, s, l = rng.choice(candidates)
    return s.unsqueeze(0).requires_grad_(True), l


def randomly_select_sample(ds: Dataset, seed_v: int = 0):
    """Return a random sample from the dataset."""
    rng = random.Random(seed_v)
    idx = rng.randint(0, len(ds) - 1)
    s, l = ds[idx]
    return s.unsqueeze(0).requires_grad_(True), l


class Selector:
    """Pre-indexes a dataset for efficient per-class sampling."""

    def __init__(self, ds):
        self.ds = ds
        self.class_to_indices: dict[int, List[int]] = {}
        for i, (_, label) in enumerate(ds):
            lab = int(label)
            self.class_to_indices.setdefault(lab, []).append(i)

    def random_select_sample_by_class(self, class_label: int, seed_v: int = 0):
        """Pick a single random sample of *class_label*."""
        indices = self.class_to_indices.get(class_label, [])
        if not indices:
            raise ValueError(f"No samples for class {class_label}")
        rng = random.Random(seed_v)
        idx = rng.choice(indices)
        s, l = self.ds[idx]
        return s.unsqueeze(0).requires_grad_(True), l

    def random_select_sample_by_class_batched(
        self, class_label: int, batch_size: int, seed_v: int = 0
    ):
        """Pick *batch_size* random samples of *class_label* as a batch tensor."""
        indices = self.class_to_indices.get(class_label, [])
        if not indices:
            raise ValueError(f"No samples for class {class_label}")
        rng = random.Random(seed_v)
        chosen = rng.choices(indices, k=batch_size)
        samples = []
        labels = []
        for idx in chosen:
            s, l = self.ds[idx]
            samples.append(s)
            labels.append(l)
        return torch.stack(samples), labels
