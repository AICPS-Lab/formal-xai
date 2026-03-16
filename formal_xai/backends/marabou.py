"""Marabou / VeriX verification backend.

Requires:
- ``maraboupy`` (https://github.com/NeuralNetworkVerification/Marabou)
- VeriX library (https://github.com/Abigail-djm/VeriX)

This backend wraps VeriX's pixel-level SMT-based verification to
provide the same ``get_ranges`` interface as the NNV backend.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
import torch

from formal_xai.backends.base import VerificationBackend

logger = logging.getLogger(__name__)


def _check_marabou():
    """Raise a helpful error if Marabou is not available."""
    try:
        import maraboupy  # noqa: F401
    except ImportError:
        raise ImportError(
            "Marabou backend requires maraboupy.\n"
            "See: https://github.com/NeuralNetworkVerification/Marabou"
        )


class MarabouBackend(VerificationBackend):
    """VeriX / Marabou verification backend.

    This backend uses VeriX's traversal-based pixel explanation approach
    which internally calls Marabou for SMT-based satisfiability checks.

    Args:
        model_path: Path to the ONNX model file.
        dataset: Dataset name for VeriX configuration (e.g. ``"MNIST"``).
        epsilon: Perturbation budget (L∞).
        traverse: Traversal order — ``"heuristic"`` or ``"sequential"``.
    """

    def __init__(
        self,
        model_path: str,
        dataset: str = "MNIST",
        epsilon: float = 0.05,
        traverse: str = "heuristic",
    ):
        _check_marabou()

        self.model_path = model_path
        self.dataset = dataset
        self.epsilon = epsilon
        self.traverse = traverse
        self._verix = None

    def _init_verix(self, image: np.ndarray):
        """Lazily initialise VeriX with a concrete image."""
        try:
            from VeriX.VeriX import VeriX
        except ImportError:
            raise ImportError(
                "VeriX library not found. Install it or add it to your "
                "Python path. See: https://github.com/Abigail-djm/VeriX"
            )
        self._verix = VeriX(
            dataset=self.dataset,
            image=image,
            model_path=self.model_path,
        )
        self._verix.traversal_order(traverse=self.traverse)

    def get_ranges(
        self,
        img: torch.Tensor,
        ub: torch.Tensor,
        lb: torch.Tensor,
        target: Union[int, Tuple[int, int]],
    ) -> Tuple[Tuple, object]:
        """Run VeriX pixel-level verification.

        Returns:
            ``((None, None), (sat_set, unsat_set, timeout_set))``
            since Marabou returns SAT/UNSAT sets rather than numeric ranges.
        """
        image_np = img.detach().cpu().numpy()
        if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
            image_np = image_np.transpose(1, 2, 0)

        self._init_verix(image_np)
        sat, unsat, timeout = self._verix.get_explanation(
            epsilon=self.epsilon,
            plot_counterfactual=False,
            plot_timeout=False,
        )
        return (None, None), {"sat": sat, "unsat": unsat, "timeout": timeout}

    def close(self) -> None:
        self._verix = None
