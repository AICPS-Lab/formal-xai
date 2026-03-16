"""LIME explainer — Local Interpretable Model-agnostic Explanations.

Ported from ``lime_explainer.py`` in the original xai_verification repo.
Supports both patch-based and superpixel-based segmentation.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances


class LIMEExplainer:
    """LIME explainer for classification models.

    Generates local explanations by approximating the model's decision
    around a specific input using a weighted linear surrogate model.

    Args:
        model: PyTorch model (should be in eval mode).
        num_classes: Number of output classes.
        segmentation: Segmentation mode — ``"patch"`` or ``"superpixel"``.
        patch_size: Size of each patch when ``segmentation="patch"``.
        num_samples: Number of perturbation samples to generate.
        kernel_width: Width of the exponential kernel for locality weighting.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 10,
        segmentation: str = "patch",
        patch_size: int = 7,
        num_samples: int = 1000,
        kernel_width: float = 0.25,
    ):
        self.model = model
        self.num_classes = num_classes
        self.segmentation = segmentation
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.kernel_width = kernel_width

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _create_patch_segments(self, img_shape: Tuple[int, ...]) -> np.ndarray:
        """Create a regular grid of patches over the image."""
        if len(img_shape) == 3:
            _, h, w = img_shape
        else:
            h, w = img_shape
        segments = np.zeros((h, w), dtype=int)
        idx = 0
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                segments[i : i + self.patch_size, j : j + self.patch_size] = idx
                idx += 1
        return segments

    # ------------------------------------------------------------------
    # Core explanation
    # ------------------------------------------------------------------

    def explain(
        self,
        image: torch.Tensor,
        target: int,
        seed: int = 0,
    ) -> Dict:
        """Generate a LIME explanation.

        Args:
            image: Input tensor ``(C, H, W)`` or ``(1, C, H, W)``.
            target: Target class to explain.
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys ``"attr"`` (attribution mask), ``"time"``,
            ``"coefficients"`` (linear model weights per segment).
        """
        np.random.seed(seed)
        start = time.perf_counter()

        if image.dim() == 4:
            image = image.squeeze(0)

        img_np = image.detach().cpu().numpy()
        segments = self._create_patch_segments(img_np.shape)
        n_segments = segments.max() + 1

        # Generate perturbation samples
        data = np.random.binomial(1, 0.5, size=(self.num_samples, n_segments))
        data[0] = np.ones(n_segments)  # original image

        # Predict on all perturbations
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for row in data:
                perturbed = img_np.copy()
                for seg_id in range(n_segments):
                    if row[seg_id] == 0:
                        mask = segments == seg_id
                        for c in range(perturbed.shape[0]):
                            perturbed[c][mask] = 0.0
                inp = torch.from_numpy(perturbed).unsqueeze(0).float()
                out = self.model(inp)
                predictions.append(out.cpu().numpy().flatten())

        predictions = np.array(predictions)

        # Compute distances and kernel weights
        distances = pairwise_distances(data, data[0:1], metric="cosine").ravel()
        kernel = np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))

        # Fit weighted linear model
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(data, predictions[:, target], sample_weight=kernel)

        # Build attribution mask
        coefficients = ridge.coef_
        attr = np.zeros_like(segments, dtype=np.float32)
        for seg_id in range(n_segments):
            attr[segments == seg_id] = coefficients[seg_id]

        elapsed = time.perf_counter() - start
        return {
            "attr": attr,
            "time": elapsed,
            "coefficients": coefficients,
        }
