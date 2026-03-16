"""Prototype-based explainer — example-driven explanations.

Ported from ``prototype_explainer.py`` in the original xai_verification repo.
Finds prototypes close to the decision boundary and uses them for
semi-factual explanations.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PrototypeExplainer:
    """Prototype-based explainer.

    Finds the nearest samples close to the decision boundary and produces
    semi-factual explanations ("what if this had been different?").

    Args:
        model: PyTorch model (eval mode).
        num_classes: Number of output classes.
        k: Number of prototypes to retrieve.
        layer_name: Optional intermediate layer for embedding comparison.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 10,
        k: int = 5,
        layer_name: Optional[str] = None,
    ):
        self.model = model
        self.num_classes = num_classes
        self.k = k
        self.layer_name = layer_name

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from the model.

        If ``layer_name`` is set, hooks into that layer; otherwise uses
        the model's penultimate layer output via a forward hook.
        """
        if self.layer_name is None:
            # Default: use the second-to-last layer
            embeddings = []

            def hook_fn(module, input, output):
                embeddings.append(input[0].detach())

            # Find the last linear layer
            last_linear = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    last_linear = module

            if last_linear is None:
                return x.flatten(1)

            handle = last_linear.register_forward_hook(hook_fn)
            with torch.no_grad():
                self.model(x)
            handle.remove()
            return embeddings[0] if embeddings else x.flatten(1)
        else:
            embeddings = []

            def hook_fn(module, input, output):
                embeddings.append(output.detach())

            target_layer = dict(self.model.named_modules())[self.layer_name]
            handle = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                self.model(x)
            handle.remove()
            return embeddings[0].flatten(1)

    # ------------------------------------------------------------------
    # Core explanation
    # ------------------------------------------------------------------

    def explain(
        self,
        image: torch.Tensor,
        dataset: Dataset,
        target_class: Optional[int] = None,
    ) -> Dict:
        """Generate a prototype-based explanation.

        Args:
            image: Input tensor ``(C, H, W)`` or ``(1, C, H, W)``.
            dataset: Reference dataset for prototype search.
            target_class: If given, prototypes are restricted to this class.

        Returns:
            Dict with keys ``"prototypes"`` (list of tensors),
            ``"distances"`` (list of floats), ``"labels"`` (list of ints),
            ``"time"``.
        """
        start = time.perf_counter()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        self.model.eval()
        query_emb = self._get_embedding(image)

        # Compute distances to all samples in the reference set
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_distances = []
        all_labels = []
        all_images = []

        with torch.no_grad():
            for batch_images, batch_labels in loader:
                emb = self._get_embedding(batch_images)
                dists = torch.cdist(query_emb, emb).squeeze(0)
                for i in range(len(batch_labels)):
                    label = int(batch_labels[i])
                    if target_class is not None and label != target_class:
                        continue
                    all_distances.append(dists[i].item())
                    all_labels.append(label)
                    all_images.append(batch_images[i])

        if not all_distances:
            elapsed = time.perf_counter() - start
            return {"prototypes": [], "distances": [], "labels": [], "time": elapsed}

        # Sort by distance and pick top-k
        indices = np.argsort(all_distances)[: self.k]
        prototypes = [all_images[i] for i in indices]
        distances = [all_distances[i] for i in indices]
        labels = [all_labels[i] for i in indices]

        elapsed = time.perf_counter() - start
        return {
            "prototypes": prototypes,
            "distances": distances,
            "labels": labels,
            "time": elapsed,
        }
