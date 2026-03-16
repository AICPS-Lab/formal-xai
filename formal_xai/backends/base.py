"""Abstract base class for verification backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch


class VerificationBackend(ABC):
    """Interface that every verification backend must implement.

    A backend accepts an input image with upper/lower bounds and a target
    specification, then returns the output reachable set ranges and a
    robustness result code.
    """

    @abstractmethod
    def get_ranges(
        self,
        img: torch.Tensor,
        ub: torch.Tensor,
        lb: torch.Tensor,
        target: Union[int, Tuple[int, int]],
    ) -> Tuple[Tuple, object]:
        """Compute output reachable-set ranges.

        Args:
            img: Original input tensor.
            ub: Element-wise upper bound.
            lb: Element-wise lower bound.
            target: Target class index (classification) or
                ``(lo, hi)`` range (regression).

        Returns:
            ``((lb_out, ub_out), robustness_result)`` where
            ``lb_out`` / ``ub_out`` are per-class output bounds and
            ``robustness_result`` is a backend-specific code.
        """
        ...

    def close(self) -> None:
        """Release any resources held by the backend (e.g. MATLAB engine)."""
        pass
