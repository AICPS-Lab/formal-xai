"""Heuristic feature ranker for VitaX.

Ranks input features (pixels/dimensions) by importance using gradient-based
heuristics. This ranking drives the binary search in the VitaX explainer.

Ported from ``heuristic_integrated_gradients`` in the original
``verified_xai.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from formal_xai.utils.seed import seed as set_seed
from formal_xai.utils.math import approximation_parameters


class HeuristicRanker:
    """Rank features by importance using gradient-based heuristics.

    Supported methods:
    - ``"sa"``: Saliency (gradient magnitude)
    - ``"ig"``: Integrated Gradients (via Captum)
    - ``"dl"``: DeepLift (via Captum)
    - ``"shap"``: GradientSHAP (via Captum)
    - ``"random"``: Random importance (control baseline)

    Args:
        model: PyTorch model.
        epsilon: Perturbation budget.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 5 / 255.0,
        seed: int = 42,
    ):
        self.model = model
        self.epsilon = epsilon
        set_seed(seed)

    # ------------------------------------------------------------------
    # Attribution methods
    # ------------------------------------------------------------------

    def attribute_saliency(
        self, input: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plain gradient saliency."""
        outputs = self.model(input)
        if target <= -1:
            grads = torch.autograd.grad(torch.unbind(outputs), input)
        else:
            grads = torch.autograd.grad(torch.unbind(outputs[:, target]), input)
        grads = grads[0]
        return grads, torch.abs(grads)

    def attribute_ig(
        self, input: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrated Gradients (via Captum)."""
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(self.model)
        baselines = torch.randn_like(input)
        attr = ig.attribute(input, target=target, baselines=baselines)
        attr = attr[0] if isinstance(attr, tuple) else attr
        return attr, torch.abs(attr)

    def attribute_deep_lift(
        self, input: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DeepLift (via Captum)."""
        from captum.attr import DeepLift

        dl = DeepLift(self.model)
        baselines = torch.randn_like(input)
        attr = dl.attribute(input, target=target, baselines=baselines)
        attr = attr[0] if isinstance(attr, tuple) else attr
        return attr, torch.abs(attr)

    def attribute_shap(
        self, input: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GradientSHAP (via Captum)."""
        from captum.attr import GradientShap

        gs = GradientShap(self.model)
        baselines = torch.randn_like(input)
        attr = gs.attribute(input, target=target, baselines=baselines)
        attr = attr[0] if isinstance(attr, tuple) else attr
        return attr, torch.abs(attr)

    def attribute_random(
        self, input: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random attribution (control baseline)."""
        attr = torch.randn_like(input)
        return attr, torch.abs(attr)

    def attribute_custom_ig(
        self, input: torch.Tensor, target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom integrated-gradients between L∞ bounds."""
        input_lb = torch.max(torch.zeros_like(input), input - self.epsilon)
        input_ub = torch.min(torch.ones_like(input), input + self.epsilon)
        n_steps = 50

        scaled, step_sizes, alphas = self._generate_scaled_features(
            input_lb, input_ub, n_steps
        )
        outputs = self.model(scaled)
        outputs = outputs[:, target]
        grads = torch.autograd.grad(torch.unbind(outputs), scaled)

        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]
        total = tuple(
            self._reshape_and_sum(sg, n_steps, g.shape[0] // n_steps, g.shape[1:])
            for sg, g in zip(scaled_grads, grads)
        )
        total_grads = total[0]
        return total_grads, torch.abs(total_grads)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    _METHOD_MAP = {
        "sa": "attribute_saliency",
        "ig": "attribute_ig",
        "dl": "attribute_deep_lift",
        "shap": "attribute_shap",
        "random": "attribute_random",
    }

    def rank_features(
        self,
        input: torch.Tensor,
        target: int,
        method: str = "sa",
        normalized: str = "directional",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank all features from most to least important.

        Args:
            input: Input tensor (batched or un-batched).
            target: Target class for attribution.
            method: Heuristic method name (``"sa"``, ``"ig"``, ``"dl"``…).
            normalized: ``"directional"`` maps gradients to [-1, 1];
                ``"min_max"`` uses raw absolute gradients.

        Returns:
            ``(sorted_indices, sorted_img, importance)``
        """
        fn_name = self._METHOD_MAP.get(method)
        if fn_name is None:
            raise ValueError(f"Invalid heuristic method: {method}")
        attr_fn = getattr(self, fn_name)
        grad, importance_ranking = attr_fn(input, target)

        sorted_indices = torch.argsort(importance_ranking.flatten(), descending=True)
        indices = torch.unravel_index(sorted_indices, input.shape)
        sorted_img = input[indices].reshape(input.shape)

        if normalized == "directional":
            importance = 2 * ((grad - grad.min()) / (grad.max() - grad.min())) - 1.0
        elif normalized == "min_max":
            importance = importance_ranking
        else:
            raise ValueError(f"Invalid normalization: {normalized}")

        return sorted_indices, sorted_img, importance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_scaled_features(input_lb, input_ub, n_steps, method="trapezoid"):
        step_fn, alpha_fn = approximation_parameters(method)
        step_sizes = step_fn(n_steps)
        alphas = alpha_fn(n_steps)
        features = []
        for a in alphas:
            features.append((input_lb + a * (input_ub - input_lb)).requires_grad_())
        return torch.concat(features), step_sizes, alphas

    @staticmethod
    def _reshape_and_sum(tensor, n_steps, n_examples, layer_size):
        return torch.sum(
            tensor.reshape((n_steps, n_examples) + layer_size), dim=0
        )
