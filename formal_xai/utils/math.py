"""Mathematical helpers for integration approximation."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def is_float(element) -> bool:
    """Check whether *element* can be converted to ``float``."""
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Integration approximation builders
# ---------------------------------------------------------------------------

def approximation_parameters(
    method: str,
) -> Tuple[Callable, Callable]:
    """Return ``(step_sizes_fn, alphas_fn)`` for the given integration method.

    Args:
        method: One of ``"riemann_trapezoid"``, ``"riemann_left"``,
            ``"riemann_right"``, ``"riemann_middle"``, ``"gauss_legendre"``,
            or the short-hand ``"trapezoid"`` (alias for ``"riemann_trapezoid"``).

    Returns:
        A pair of callables that each accept an integer *n* and return an array.
    """
    if method in ("trapezoid", "riemann_trapezoid"):
        return riemann_builders("trapezoid")
    if method.startswith("riemann"):
        variant = method.split("_", 1)[1] if "_" in method else "trapezoid"
        return riemann_builders(variant)
    if method == "gauss_legendre":
        return gauss_legendre_builders()
    raise ValueError(f"Unknown integration method: {method}")


def riemann_builders(
    method: str = "trapezoid",
) -> Tuple[Callable, Callable]:
    """Return Riemann sum helper functions.

    Args:
        method: ``"trapezoid"``, ``"left"``, ``"right"``, or ``"middle"``.

    Returns:
        ``(step_sizes, alphas)`` — each is a callable ``int -> np.ndarray``.
    """

    def step_sizes(n: int) -> np.ndarray:
        if method == "trapezoid":
            return np.array([1 / (2 * n)] + [1 / n] * (n - 1) + [1 / (2 * n)])
        return np.array([1 / n] * n)

    def alphas(n: int) -> np.ndarray:
        if method == "trapezoid":
            return np.linspace(0, 1, n + 1)
        if method == "left":
            return np.linspace(0, 1 - 1 / n, n)
        if method == "right":
            return np.linspace(1 / n, 1, n)
        if method == "middle":
            return np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n)
        raise ValueError(f"Unknown Riemann method: {method}")

    return step_sizes, alphas


def gauss_legendre_builders() -> Tuple[Callable, Callable]:
    """Return Gauss–Legendre quadrature helper functions.

    Returns:
        ``(step_sizes, alphas)`` — each is a callable ``int -> np.ndarray``.
    """

    def step_sizes(n: int) -> np.ndarray:
        points, weights = np.polynomial.legendre.leggauss(n)
        return weights / 2.0  # map from [-1, 1] to [0, 1]

    def alphas(n: int) -> np.ndarray:
        points, _ = np.polynomial.legendre.leggauss(n)
        return (points + 1) / 2.0  # map from [-1, 1] to [0, 1]

    return step_sizes, alphas
