#!/usr/bin/env python3
"""Unit tests for the pure-Python n2v verification backend.

These tests use tiny synthetic models (no model files needed) to verify
that the ``N2VPyBackend`` correctly:
1. Wraps a PyTorch model and performs reachability analysis.
2. Returns bounds with correct shape and monotonicity (lb <= ub).
3. Produces a valid robustness code (0 or 1).
4. Handles both ``approx-star`` and ``exact-star`` methods.

Usage::

    python tests/test_n2v_backend.py
"""

import sys

import numpy as np
import torch
import torch.nn as nn


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_tiny_model(input_dim=4, hidden_dim=8, output_dim=3):
    """Build a tiny deterministic MLP."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    model.eval()
    return model


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_backend_approx_star():
    """Approx-star: bounds shape, monotonicity, and robustness code."""
    from formal_xai.backends.n2v import N2VPyBackend

    model = _make_tiny_model()
    backend = N2VPyBackend(
        model=model,
        output_size=3,
        reach_method="approx-star",
        epsilon=0.1,
    )

    img = torch.tensor([0.5, 0.5, 0.5, 0.5])
    eps = 0.1
    ub = img + eps
    lb = img - eps

    target = 0  # classification target is class 0
    (lb_out, ub_out), robust = backend.get_ranges(img, ub, lb, target)

    assert lb_out.shape == (3,), f"lb_out shape: {lb_out.shape}"
    assert ub_out.shape == (3,), f"ub_out shape: {ub_out.shape}"
    assert np.all(lb_out <= ub_out), "lb_out must be <= ub_out"
    assert robust in (0, 1), f"robustness code: {robust}"

    print(f"  ✓ approx-star: lb={lb_out}, ub={ub_out}, robust={robust}")
    return True


def test_backend_exact_star():
    """Exact-star: bounds shape, monotonicity, and robustness code."""
    from formal_xai.backends.n2v import N2VPyBackend

    model = _make_tiny_model()
    backend = N2VPyBackend(
        model=model,
        output_size=3,
        reach_method="exact-star",
        epsilon=0.1,
    )

    img = torch.tensor([0.5, 0.5, 0.5, 0.5])
    eps = 0.1
    ub = img + eps
    lb = img - eps

    target = 0
    (lb_out, ub_out), robust = backend.get_ranges(img, ub, lb, target)

    assert lb_out.shape == (3,), f"lb_out shape: {lb_out.shape}"
    assert ub_out.shape == (3,), f"ub_out shape: {ub_out.shape}"
    assert np.all(lb_out <= ub_out), "lb_out must be <= ub_out"
    assert robust in (0, 1), f"robustness code: {robust}"

    print(f"  ✓ exact-star:  lb={lb_out}, ub={ub_out}, robust={robust}")
    return True


def test_exact_tighter_or_equal_to_approx():
    """Exact bounds should be at least as tight as approx bounds."""
    from formal_xai.backends.n2v import N2VPyBackend

    model = _make_tiny_model()

    img = torch.tensor([0.5, 0.5, 0.5, 0.5])
    eps = 0.1
    ub = img + eps
    lb = img - eps
    target = 0

    approx = N2VPyBackend(model=model, output_size=3,
                          reach_method="approx-star", epsilon=eps)
    exact  = N2VPyBackend(model=model, output_size=3,
                          reach_method="exact-star", epsilon=eps)

    (a_lb, a_ub), _ = approx.get_ranges(img, ub, lb, target)
    (e_lb, e_ub), _ = exact.get_ranges(img, ub, lb, target)

    # Exact lb >= approx lb  and  exact ub <= approx ub  (tighter)
    tol = 1e-6
    assert np.all(e_lb >= a_lb - tol), f"exact lb should >= approx lb"
    assert np.all(e_ub <= a_ub + tol), f"exact ub should <= approx ub"

    print(f"  ✓ exact bounds are tighter/equal to approx bounds")
    return True


def test_invalid_method():
    """Invalid method should raise ValueError."""
    from formal_xai.backends.n2v import N2VPyBackend

    model = _make_tiny_model()
    try:
        N2VPyBackend(model=model, output_size=3,
                     reach_method="bad-method", epsilon=0.1)
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError:
        print("  ✓ ValueError raised for bad method name")
        return True


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("n2v Backend Unit Tests")
    print("=" * 50)

    results = []
    for test_fn in [
        test_backend_approx_star,
        test_backend_exact_star,
        test_exact_tighter_or_equal_to_approx,
        test_invalid_method,
    ]:
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if all(results) else 1)
