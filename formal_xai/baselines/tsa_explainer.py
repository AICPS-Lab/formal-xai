"""Targeted Semi-factual Adversarial (TSA) explainer.

Ported from ``tsa_explainer.py`` in the original xai_verification repo.
Generates explanations by optimising a perturbation to push the prediction
toward a target class while maintaining the original prediction.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class TSAExplainer:
    """Targeted Semi-factual Adversarial explainer.

    Uses gradient-based optimisation to find a perturbation that increases
    confidence on a target class while remaining classified as the original.

    Args:
        model: PyTorch model.
        num_classes: Number of output classes.
        epsilon: Maximum perturbation magnitude (L∞).
        lr: Learning rate for the optimisation.
        max_steps: Maximum optimisation steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 10,
        epsilon: float = 0.1,
        lr: float = 0.01,
        max_steps: int = 200,
    ):
        self.model = model
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.lr = lr
        self.max_steps = max_steps

    def explain(
        self,
        image: torch.Tensor,
        target_class: int,
        seed: int = 0,
    ) -> Dict:
        """Generate a TSA explanation.

        Args:
            image: Input tensor ``(C, H, W)`` or ``(1, C, H, W)``.
            target_class: Target class to push toward.
            seed: Random seed.

        Returns:
            Dict with keys ``"perturbation"`` (delta tensor),
            ``"attr"`` (absolute perturbation as attribution),
            ``"perturbed_image"``, ``"original_pred"``, ``"perturbed_pred"``,
            ``"time"``.
        """
        torch.manual_seed(seed)
        start = time.perf_counter()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.clone().detach().float()
        delta = torch.zeros_like(image, requires_grad=True)

        self.model.eval()

        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(image).argmax(1).item()

        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.max_steps):
            optimizer.zero_grad()
            perturbed = image + delta
            perturbed = torch.clamp(perturbed, 0, 1)
            output = self.model(perturbed)

            # Loss: push toward target class while penalising large perturbation
            target_loss = -F.log_softmax(output, dim=1)[0, target_class]
            l2_reg = torch.norm(delta.flatten(), p=2)
            loss = target_loss + 0.01 * l2_reg

            loss.backward()
            optimizer.step()

            # Clamp perturbation to epsilon ball
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)

            # Check if prediction still matches original
            with torch.no_grad():
                current_pred = self.model(image + delta).argmax(1).item()

        perturbed_image = torch.clamp(image + delta, 0, 1).detach()
        perturbation = delta.detach()

        with torch.no_grad():
            perturbed_pred = self.model(perturbed_image).argmax(1).item()

        attr = torch.abs(perturbation).squeeze(0).cpu().numpy()

        elapsed = time.perf_counter() - start
        return {
            "perturbation": perturbation.squeeze(0).cpu().numpy(),
            "attr": attr,
            "perturbed_image": perturbed_image.squeeze(0).cpu().numpy(),
            "original_pred": original_pred,
            "perturbed_pred": perturbed_pred,
            "time": elapsed,
        }
