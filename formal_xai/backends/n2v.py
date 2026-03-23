"""Pure-Python NNV backend via the n2v library.

Requires:
- n2v (https://github.com/sammsaski/n2v) — installed or available as a
  submodule at ``third_party/n2v``.

This backend wraps a PyTorch model directly (no ONNX export needed) and
performs set-based reachability analysis using Star sets.

Supported reachability methods (mapped from the MATLAB naming convention):
- ``approx-star``: Over-approximate reachability (``method='approx'``).
- ``exact-star``: Exact star-set reachability (``method='exact'``).
- ``relax-star-area_<factor>``: Approx with a custom relaxation factor.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import List, Tuple, Union

import numpy as np
import torch

from formal_xai.backends.base import VerificationBackend
from formal_xai.utils.math import is_float

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the n2v submodule is importable
# ---------------------------------------------------------------------------
_N2V_SUBMODULE = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "third_party", "n2v"
)
_N2V_SUBMODULE = os.path.normpath(_N2V_SUBMODULE)
if os.path.isdir(_N2V_SUBMODULE) and _N2V_SUBMODULE not in sys.path:
    sys.path.insert(0, _N2V_SUBMODULE)

try:
    from n2v import NeuralNetwork
    from n2v.sets import Star
except ImportError:
    raise ImportError(
        "n2v backend requires the n2v library.\n"
        "Either install it (`pip install -e third_party/n2v`) or make sure\n"
        "the git submodule is initialised:\n"
        "    git submodule update --init --recursive"
    )


class N2VPyBackend(VerificationBackend):
    """Pure-Python NNV verification backend.

    Unlike the MATLAB ``NNVBackend``, this backend works **directly with
    PyTorch models** — no ONNX export is required.

    Args:
        model: PyTorch ``nn.Module`` in eval mode.
        output_size: Number of model outputs (classes).
        reach_method: Reachability method name (``"approx-star"``,
            ``"exact-star"``, or ``"relax-star-area_<factor>"``).
        epsilon: Perturbation budget (L∞).  Not used directly by the
            backend (bounds are passed in ``get_ranges``), but stored
            for consistency with the MATLAB backend.
        model_path: Unused — kept for API compatibility.
    """

    VALID_METHODS = {"approx-star", "exact-star"}

    def __init__(
        self,
        model: torch.nn.Module,
        output_size: int = 10,
        reach_method: str = "approx-star",
        epsilon: float = 35 / 255.0,
        model_path: str | None = None,
    ):
        # Validate reach method
        valid = (
            reach_method in self.VALID_METHODS
            or reach_method.startswith("relax-star-area")
        )
        if not valid:
            raise ValueError(f"Invalid reach_method: {reach_method}")
        if reach_method.startswith("relax-star-area"):
            factor = reach_method.split("_")[-1]
            if not is_float(factor):
                raise ValueError(
                    f"relax-star-area requires a numeric relaxFactor, got: {factor}"
                )

        self.reach_method = reach_method
        self.output_size = output_size
        self.epsilon = epsilon

        # n2v's _reach_sequential uses list(model.children()) to extract
        # layers.  Models that create nn.ReLU() / nn.Flatten() inline in
        # forward() (rather than registering them as attributes) will be
        # missing those layers and produce WRONG results.
        # Fix: convert to nn.Sequential where every op is a registered child.
        seq_model = self._to_sequential(model)
        seq_model.eval()
        self._net = NeuralNetwork(seq_model)
        logger.info("n2v NeuralNetwork wrapper ready (%d layers)", len(self._net.layers))

    # ------------------------------------------------------------------
    # Model conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_sequential(model: torch.nn.Module) -> torch.nn.Module:
        """Convert a model to ``nn.Sequential`` if it uses inline activations.

        Many models (e.g. our MLP) define ``nn.ReLU()`` inside ``forward()``
        instead of registering it as a sub-module.  n2v's layer-by-layer
        reachability never calls ``forward()``, so those activations are
        invisible.  This method traces the registered children and inserts
        ``nn.Flatten`` / ``nn.ReLU`` layers between them so that the
        ``nn.Sequential`` is equivalent.

        If the model is *already* an ``nn.Sequential`` with all layers
        registered, it is returned unchanged.
        """
        import torch.nn as nn

        # If already Sequential with non-Linear children present, trust it
        if isinstance(model, nn.Sequential):
            return model

        children = list(model.children())
        if not children:
            return model  # single-layer or opaque model – pass through

        # Check if all children are Linear (i.e. activations are inline)
        all_linear = all(isinstance(c, nn.Linear) for c in children)
        if not all_linear:
            # Model already has registered activations → trust it
            return model

        # Reconstruct: Flatten ➜ (Linear ➜ ReLU) × (n-1) ➜ Linear
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        for i, child in enumerate(children):
            layers.append(child)
            if i < len(children) - 1:  # add ReLU after every layer except last
                layers.append(nn.ReLU())

        seq = nn.Sequential(*layers)
        seq.eval()
        return seq

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_reach_args(self) -> Tuple[str, dict]:
        """Map MATLAB-style method names to n2v ``reach()`` arguments."""
        rm = self.reach_method
        if rm == "exact-star":
            return "exact", {}
        if rm.startswith("relax-star-area"):
            factor = float(rm.split("_")[-1])
            return "approx", {"relax_factor": factor}
        # default: approx-star
        return "approx", {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_ranges(
        self,
        img: torch.Tensor,
        ub: torch.Tensor,
        lb: torch.Tensor,
        target: Union[int, Tuple[int, int]],
    ) -> Tuple[Tuple, object]:
        """Compute output reachable-set ranges via n2v.

        The method:
        1. Flattens ``lb`` / ``ub`` to 1-D vectors.
        2. Constructs a ``Star`` from those bounds.
        3. Calls ``net.reach()`` to propagate the star through the network.
        4. Unions ``get_ranges()`` over the (possibly many) output stars.
        5. Returns ``((lb_out, ub_out), robustness_code)`` where
           ``robustness_code`` is ``1`` (robust) or ``0`` (not robust).
        """
        # Flatten to 1-D numpy
        lb_np = lb.detach().cpu().numpy().flatten().astype(np.float64)
        ub_np = ub.detach().cpu().numpy().flatten().astype(np.float64)

        # Build input Star from bounds
        input_star = Star.from_bounds(lb_np, ub_np)

        # Reach
        method, kwargs = self._map_reach_args()
        output_stars: List = self._net.reach(input_star, method=method, **kwargs)

        # Aggregate ranges over all output stars
        lb_out = np.full(self.output_size, np.inf)
        ub_out = np.full(self.output_size, -np.inf)

        for star in output_stars:
            s_lb, s_ub = star.get_ranges()
            s_lb = np.asarray(s_lb).flatten()
            s_ub = np.asarray(s_ub).flatten()
            lb_out = np.minimum(lb_out, s_lb[: self.output_size])
            ub_out = np.maximum(ub_out, s_ub[: self.output_size])

        # Determine robustness code (same logic as MATLAB NNV)
        if isinstance(target, int):
            # Classification: target-class lower bound must exceed all
            # other classes' upper bounds.
            other_ub_max = max(
                ub_out[i] for i in range(self.output_size) if i != target
            )
            robust = int(lb_out[target] > other_ub_max)
        elif isinstance(target, tuple):
            # Regression: output must stay within [lo, hi]
            robust = int(lb_out[0] > target[0] and ub_out[0] < target[1])
        else:
            robust = 0

        return (lb_out, ub_out), robust
