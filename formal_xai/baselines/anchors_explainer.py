"""Anchors explainer — rule-based local explanations.

Ported from ``anchors_explainer.py`` in the original xai_verification repo.
Finds a minimal set of input features (patches) that anchor the prediction.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class AnchorsExplainer:
    """Anchors explainer for classification models.

    Identifies a minimal set of patches that, when present, are sufficient
    to maintain the model's prediction with high confidence.

    Args:
        model: PyTorch model (eval mode).
        num_classes: Number of output classes.
        patch_size: Size of each square patch.
        threshold: Precision threshold for an anchor (default 0.95).
        num_samples: Number of samples per anchor candidate evaluation.
        beam_width: Beam search width.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 10,
        patch_size: int = 7,
        threshold: float = 0.95,
        num_samples: int = 1000,
        beam_width: int = 2,
    ):
        self.model = model
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.threshold = threshold
        self.num_samples = num_samples
        self.beam_width = beam_width

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _create_patch_segments(self, img_shape: Tuple[int, ...]) -> np.ndarray:
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

    def _precision(
        self,
        image: np.ndarray,
        anchor: List[int],
        segments: np.ndarray,
        pred_class: int,
    ) -> float:
        """Estimate the precision of a candidate anchor set."""
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.num_samples):
                perturbed = np.random.uniform(0, 1, size=image.shape).astype(np.float32)
                for seg_id in anchor:
                    mask = segments == seg_id
                    for c in range(image.shape[0]):
                        perturbed[c][mask] = image[c][mask]
                inp = torch.from_numpy(perturbed).unsqueeze(0).float()
                out = self.model(inp)
                if out.argmax(1).item() == pred_class:
                    correct += 1
        return correct / self.num_samples

    def explain(
        self,
        image: torch.Tensor,
        seed: int = 0,
    ) -> Dict:
        """Generate an Anchors explanation.

        Args:
            image: Input tensor ``(C, H, W)`` or ``(1, C, H, W)``.
            seed: Random seed.

        Returns:
            Dict with keys ``"anchor"`` (list of anchor segment ids),
            ``"precision"``, ``"segments"``, ``"attr"`` (binary mask), ``"time"``.
        """
        np.random.seed(seed)
        start = time.perf_counter()

        if image.dim() == 4:
            image = image.squeeze(0)

        img_np = image.detach().cpu().numpy()
        segments = self._create_patch_segments(img_np.shape)
        n_segments = segments.max() + 1

        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            pred = self.model(image.unsqueeze(0)).argmax(1).item()

        # Beam search for anchor
        candidates: List[Tuple[List[int], float]] = []
        for seg_id in range(n_segments):
            prec = self._precision(img_np, [seg_id], segments, pred)
            candidates.append(([seg_id], prec))

        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[: self.beam_width]

        best_anchor, best_prec = candidates[0]
        while best_prec < self.threshold and len(best_anchor) < n_segments:
            new_candidates = []
            for anchor, _ in candidates:
                for seg_id in range(n_segments):
                    if seg_id in anchor:
                        continue
                    new_anchor = anchor + [seg_id]
                    prec = self._precision(img_np, new_anchor, segments, pred)
                    new_candidates.append((new_anchor, prec))
            new_candidates.sort(key=lambda x: -x[1])
            candidates = new_candidates[: self.beam_width]
            if candidates:
                best_anchor, best_prec = candidates[0]
            else:
                break

        # Build binary attribution mask
        attr = np.zeros(segments.shape, dtype=np.float32)
        for seg_id in best_anchor:
            attr[segments == seg_id] = 1.0

        elapsed = time.perf_counter() - start
        return {
            "anchor": best_anchor,
            "precision": best_prec,
            "segments": segments,
            "attr": attr,
            "time": elapsed,
        }
