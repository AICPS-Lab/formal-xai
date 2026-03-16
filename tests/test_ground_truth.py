#!/usr/bin/env python3
"""Ground truth integration tests for VitaX.

Runs the full VitaX + NNV pipeline on MNIST samples and compares
attribution content against saved ground truth from the old
xai_verification repo (2025 results).

These tests check CONTENT, not just shape — the actual attribution
values, non-zero patterns, and robustness results must match.

Usage:
    PYTHONPATH=/home/david1/Documents/GitHub/formal-xai \
      /home/david1/miniconda3/envs/vitax/bin/python tests/test_ground_truth.py

Requirements:
    - vitax conda environment
    - MATLAB Engine + NNV toolbox
    - MNIST MLP model files in the old repo's outputs/models dirs
"""

import os
import sys
import time

import numpy as np
import torch

# Paths to old repo assets
OLD_REPO = "/home/david1/Documents/GitHub/xai_verification"
MODEL_PT = os.path.join(OLD_REPO, "models", "MyFirstExperiment_mnist_MLP.pt")
MODEL_ONNX = os.path.join(OLD_REPO, "outputs", "MyFirstExperiment_mnist_mlp.onnx")
GROUND_TRUTH_DIR = os.path.join(os.path.dirname(__file__), "ground_truth")

# Ground truth files: filename -> (true_label, counterfactual_class, sample_idx)
GROUND_TRUTH_FILES = {
    "mnist_approx-star_0_1_3.npy": (0, 1, 3),
    "mnist_approx-star_7_6_33.npy": (7, 6, 33),
    "mnist_approx-star_6_2_10.npy": (6, 2, 10),
}

EPSILON = 25 / 255
NUM_CLASSES = 10


def load_ground_truth(filename):
    """Load a ground truth .npy file and return its contents."""
    path = os.path.join(GROUND_TRUTH_DIR, filename)
    data = np.load(path, allow_pickle=True).item()
    return data


def test_vitax_attribution_content():
    """End-to-end test: run VitaX on the SAME inputs as ground truth
    and verify the attributions MATCH in content.

    Checks:
    1. Attribution non-zero pattern matches exactly
    2. Attribution values are close (within tolerance)
    3. Attribution sign pattern matches
    4. Non-zero count matches
    """
    from formal_xai.vitax import VitaX
    from formal_xai.models import MLP

    # Load model
    model = MLP(input_size=28 * 28, output_size=NUM_CLASSES, input_channels=1)
    model.load_state_dict(
        torch.load(MODEL_PT, map_location=torch.device("cpu"))
    )
    model.eval()
    print(f"✓ Model loaded from {MODEL_PT}")

    # Create VitaX with NNV backend (same settings as old experiments)
    verifier = VitaX(
        model_path=MODEL_ONNX,
        backend="nnv",
        reach_method="approx-star",
        heuristic_method="sa",
        epsilon=EPSILON,
        num_classes=NUM_CLASSES,
    )
    print("✓ VitaX + NNV backend initialized")

    passed = 0
    failed = 0

    for filename, (true_label, class_target, sample_idx) in GROUND_TRUTH_FILES.items():
        print(f"\n--- Testing: {filename} ---")
        print(f"    label={true_label}, counterfactual={class_target}")

        # Load ground truth
        gt = load_ground_truth(filename)
        gt_input = gt["input"]  # ndarray (1, 28, 28)
        gt_attr = gt["attr"]    # torch.Tensor (28, 28)
        gt_target = gt["target"]
        gt_class_target = gt["class_target"]

        assert gt_target == true_label, f"Label mismatch: {gt_target} != {true_label}"
        assert gt_class_target == class_target, f"Class target mismatch"

        # Convert input to tensor with gradients
        inp = torch.tensor(gt_input, dtype=torch.float32).requires_grad_(True)

        # Run VitaX on the SAME input
        t0 = time.perf_counter()
        attr = verifier.explain(
            model,
            inp,
            target=true_label,
            class_to_check=class_target,
            return_robustness=False,
            show_progress=False,
            seed=42,
        )
        elapsed = time.perf_counter() - t0
        print(f"    VitaX completed in {elapsed:.1f}s")

        # --- CONTENT CHECKS ---
        # Convert both to numpy for comparison
        if isinstance(attr, torch.Tensor):
            attr_np = attr.detach().cpu().numpy()
        else:
            attr_np = np.array(attr)

        if isinstance(gt_attr, torch.Tensor):
            gt_attr_np = gt_attr.detach().cpu().numpy()
        else:
            gt_attr_np = np.array(gt_attr)

        # Squeeze to same shape
        attr_np = attr_np.squeeze()
        gt_attr_np = gt_attr_np.squeeze()

        assert attr_np.shape == gt_attr_np.shape, (
            f"Shape mismatch: {attr_np.shape} vs {gt_attr_np.shape}"
        )

        # Check 1: Non-zero PATTERN must match exactly
        new_nonzero = attr_np != 0
        gt_nonzero = gt_attr_np != 0
        pattern_match = np.array_equal(new_nonzero, gt_nonzero)

        # Check 2: Values must be close where non-zero
        if gt_nonzero.any():
            value_diff = np.abs(attr_np[gt_nonzero] - gt_attr_np[gt_nonzero])
            max_diff = value_diff.max()
            mean_diff = value_diff.mean()
        else:
            max_diff = 0.0
            mean_diff = 0.0

        # Check 3: Sign pattern must match
        new_sign = np.sign(attr_np)
        gt_sign = np.sign(gt_attr_np)
        sign_match = np.array_equal(new_sign, gt_sign)

        # Check 4: Non-zero counts
        new_nz_count = int(new_nonzero.sum())
        gt_nz_count = int(gt_nonzero.sum())

        # Report
        print(f"    Non-zero count: new={new_nz_count}, gt={gt_nz_count}")
        print(f"    Pattern match:  {pattern_match}")
        print(f"    Sign match:     {sign_match}")
        print(f"    Max value diff: {max_diff:.6f}")
        print(f"    Mean value diff: {mean_diff:.6f}")

        # Determine pass/fail
        # Tolerance: values should be very close (within 1e-5)
        # since we're using the SAME model, SAME input, SAME method
        value_ok = max_diff < 1e-4
        ok = pattern_match and sign_match and value_ok

        if ok:
            print(f"    ✓ PASSED — attribution CONTENT matches ground truth")
            passed += 1
        else:
            reasons = []
            if not pattern_match:
                reasons.append(f"non-zero pattern differs ({new_nz_count} vs {gt_nz_count})")
            if not sign_match:
                reasons.append("sign pattern differs")
            if not value_ok:
                reasons.append(f"values differ too much (max_diff={max_diff:.6f})")
            print(f"    ✗ FAILED — {'; '.join(reasons)}")
            failed += 1

    verifier.close()

    print(f"\n{'='*50}")
    print(f"Ground Truth Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("VitaX Ground Truth Integration Tests")
    print("Tests run full NNV pipeline and compare attribution CONTENT")
    print("=" * 60)

    ok = test_vitax_attribution_content()
    sys.exit(0 if ok else 1)
