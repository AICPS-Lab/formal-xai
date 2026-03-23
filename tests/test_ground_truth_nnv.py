#!/usr/bin/env python3
"""Ground truth test: NNV (MATLAB) backend vs NNV oracle.

Uses the committed test model (tests/models/) and NNV oracle ground truth
(tests/ground_truth/nnv/) to verify that the NNV backend produces
consistent attributions.

Requirements:
    - MATLAB Engine + NNV toolbox
    - Recommend using the ``vitax`` conda environment

Usage::

    python tests/test_ground_truth_nnv.py

To generate the oracle first (requires MATLAB)::

    python tests/scripts/generate_oracle.py --backend nnv
"""

import os
import sys
import time

import numpy as np
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_DIR)

MODEL_PT = os.path.join(TESTS_DIR, "models", "test_mnist_mlp.pt")
MODEL_ONNX = os.path.join(TESTS_DIR, "models", "test_mnist_mlp.onnx")
GROUND_TRUTH_DIR = os.path.join(TESTS_DIR, "ground_truth", "nnv")

EPSILON = 25 / 255
NUM_CLASSES = 10


def test_nnv_ground_truth():
    """Run VitaX + NNV on the test model and compare to saved oracle."""
    from formal_xai.vitax import VitaX
    from formal_xai.models import MLP

    # Load model
    model = MLP(input_size=784, output_size=NUM_CLASSES, input_channels=1)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    model.eval()
    print(f"✓ Model: {MODEL_PT}")

    # Create VitaX with NNV backend
    verifier = VitaX(
        model_path=MODEL_ONNX,
        backend="nnv",
        reach_method="approx-star",
        heuristic_method="sa",
        epsilon=EPSILON,
        num_classes=NUM_CLASSES,
    )
    print("✓ VitaX + NNV initialized\n")

    passed = 0
    failed = 0

    for f in sorted(os.listdir(GROUND_TRUTH_DIR)):
        if not f.endswith(".npy"):
            continue

        gt = np.load(os.path.join(GROUND_TRUTH_DIR, f), allow_pickle=True).item()
        target = gt["target"]
        class_target = gt["class_target"]
        gt_attr = gt["attr"]
        gt_attr_np = gt_attr.detach().cpu().numpy().squeeze() if hasattr(gt_attr, "detach") else np.array(gt_attr).squeeze()

        print(f"--- {f} (target={target}, cf={class_target}) ---")

        inp = torch.tensor(gt["input"], dtype=torch.float32).unsqueeze(0).requires_grad_(True)

        t0 = time.perf_counter()
        attr = verifier.explain(
            model, inp, target=target, class_to_check=class_target,
            return_robustness=False, show_progress=False, seed=42,
        )
        elapsed = time.perf_counter() - t0

        attr_np = attr.detach().cpu().numpy().squeeze() if isinstance(attr, torch.Tensor) else np.array(attr).squeeze()
        print(f"    Time: {elapsed:.1f}s")

        # Strict checks — same backend, same model, same seed
        pattern_ok = np.array_equal(attr_np != 0, gt_attr_np != 0)
        sign_ok = np.array_equal(np.sign(attr_np), np.sign(gt_attr_np))
        max_diff = np.abs(attr_np - gt_attr_np).max()
        value_ok = max_diff < 1e-4  # slightly looser for MATLAB float rounding

        nz_new = int((attr_np != 0).sum())
        nz_gt = int((gt_attr_np != 0).sum())
        print(f"    Non-zero: new={nz_new}, gt={nz_gt}")
        print(f"    Pattern match: {pattern_ok}")
        print(f"    Sign match:    {sign_ok}")
        print(f"    Max |diff|:    {max_diff:.8f}")

        if pattern_ok and sign_ok and value_ok:
            print(f"    ✓ PASSED")
            passed += 1
        else:
            reasons = []
            if not pattern_ok: reasons.append("pattern mismatch")
            if not sign_ok: reasons.append("sign mismatch")
            if not value_ok: reasons.append(f"max_diff={max_diff:.8f}")
            print(f"    ✗ FAILED — {'; '.join(reasons)}")
            failed += 1
        print()

    verifier.close()
    print(f"{'='*50}")
    print(f"NNV Ground Truth: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("VitaX Ground Truth Test — NNV (MATLAB) backend")
    print("=" * 60)

    if not os.path.exists(MODEL_PT):
        print(f"✗ Model not found: {MODEL_PT}")
        print("Run: python tests/scripts/train_test_model.py")
        sys.exit(1)
    if not os.path.isdir(GROUND_TRUTH_DIR):
        print(f"✗ Ground truth not found: {GROUND_TRUTH_DIR}")
        print("Run: python tests/scripts/generate_oracle.py --backend nnv")
        sys.exit(1)

    ok = test_nnv_ground_truth()
    sys.exit(0 if ok else 1)
