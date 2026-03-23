#!/usr/bin/env python3
"""Generate ground truth oracle explanations for VitaX tests.

Runs VitaX with the specified backend on the test model and saves
attribution .npy files to tests/ground_truth/.

Usage::

    # Generate oracle with n2v (pure Python — no MATLAB needed)
    python tests/scripts/generate_oracle.py --backend n2v

    # Generate oracle with NNV (requires MATLAB + NNV toolbox)
    python tests/scripts/generate_oracle.py --backend nnv

    # Run multiple times to verify determinism
    python tests/scripts/generate_oracle.py --backend n2v --verify-determinism

Each run saves to tests/ground_truth/<backend>/<filename>.npy
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_DIR)

MODEL_DIR = os.path.join(TESTS_DIR, "models")
MODEL_PT = os.path.join(MODEL_DIR, "test_mnist_mlp.pt")
MODEL_ONNX = os.path.join(MODEL_DIR, "test_mnist_mlp.onnx")
SAMPLES_PT = os.path.join(MODEL_DIR, "test_samples.pt")

EPSILON = 25 / 255
NUM_CLASSES = 10

# Pairs of (target_class, counterfactual_class) to test
TEST_PAIRS = [
    (0, 1),
    (7, 3),
    (3, 8),
]


def generate(backend: str, output_dir: str, reach_method: str = "approx-star"):
    """Generate oracle explanations."""
    from formal_xai.vitax import VitaX
    from formal_xai.models import MLP

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = MLP(input_size=784, output_size=NUM_CLASSES, input_channels=1)
    model.load_state_dict(
        torch.load(MODEL_PT, map_location=torch.device("cpu"))
    )
    model.eval()
    print(f"✓ Model loaded from {MODEL_PT}")

    # Load test samples
    samples = torch.load(SAMPLES_PT, map_location=torch.device("cpu"))
    print(f"✓ Loaded {len(samples)} test samples")

    # Create VitaX
    backend_kwargs = {}
    if backend == "n2v":
        backend_kwargs["model"] = model
    verifier = VitaX(
        model_path=MODEL_ONNX,
        backend=backend,
        reach_method=reach_method,
        heuristic_method="sa",
        epsilon=EPSILON,
        num_classes=NUM_CLASSES,
        **backend_kwargs,
    )
    print(f"✓ VitaX + {backend} backend initialized\n")

    for target, counterfactual in TEST_PAIRS:
        print(f"--- target={target}, counterfactual={counterfactual} ---")

        # Get sample image for target class
        if target not in samples:
            print(f"  ⚠ No sample for class {target}, skipping")
            continue

        inp = samples[target].unsqueeze(0).requires_grad_(True)

        t0 = time.perf_counter()
        attr = verifier.explain(
            model, inp,
            target=target,
            class_to_check=counterfactual,
            return_robustness=False,
            show_progress=True,
            seed=42,
        )
        elapsed = time.perf_counter() - t0

        # Save
        filename = f"test_mnist_{target}_vs_{counterfactual}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, {
            "input": inp.detach().cpu().numpy().squeeze(0),  # (1, 28, 28)
            "attr": attr.detach().cpu() if isinstance(attr, torch.Tensor) else attr,
            "target": target,
            "class_target": counterfactual,
            "time": elapsed,
            "backend": backend,
            "reach_method": reach_method,
            "epsilon": EPSILON,
        })
        print(f"  Saved: {filepath}")

        attr_np = attr.detach().cpu().numpy() if isinstance(attr, torch.Tensor) else np.array(attr)
        nz = int((attr_np != 0).sum())
        print(f"  Time: {elapsed:.1f}s, non-zero pixels: {nz}\n")

    verifier.close()
    print("✓ Done")


def verify_determinism(backend: str, output_dir: str, runs: int = 3):
    """Run generation multiple times and check results are deterministic."""
    import tempfile

    print(f"=== Determinism check ({runs} runs) ===\n")

    results = {}
    for run in range(runs):
        print(f"--- Run {run+1}/{runs} ---")
        tmp_dir = tempfile.mkdtemp()
        generate(backend, tmp_dir, reach_method="approx-star")

        for f in sorted(os.listdir(tmp_dir)):
            if not f.endswith(".npy"):
                continue
            data = np.load(os.path.join(tmp_dir, f), allow_pickle=True).item()
            attr = data["attr"]
            if hasattr(attr, "detach"):
                attr = attr.detach().cpu().numpy()
            attr = np.array(attr).squeeze()

            if f not in results:
                results[f] = []
            results[f].append(attr)
        print()

    print("=== Determinism results ===")
    all_ok = True
    for f, attrs in results.items():
        ref = attrs[0]
        match = all(np.array_equal(ref, a) for a in attrs[1:])
        status = "✓" if match else "✗"
        print(f"  {status} {f}: {'deterministic' if match else 'NOT deterministic'}")
        if not match:
            all_ok = False
            for i, a in enumerate(attrs[1:], 2):
                diff = np.abs(ref - a).max()
                print(f"      Run {i} max diff: {diff:.8f}")

    if all_ok:
        print(f"\n✓ All {len(results)} samples deterministic across {runs} runs")
        # Copy last run to output dir
        print(f"  Saving to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        last_tmp = tempfile.mkdtemp()
        generate(backend, output_dir)
    else:
        print(f"\n✗ Some samples NOT deterministic!")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VitaX ground truth")
    parser.add_argument("--backend", default="n2v", choices=["n2v", "nnv"],
                        help="Verification backend (default: n2v)")
    parser.add_argument("--reach-method", default="approx-star",
                        choices=["approx-star", "exact-star"],
                        help="Reachability method (default: approx-star)")
    parser.add_argument("--verify-determinism", action="store_true",
                        help="Run multiple times to check determinism")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of determinism runs (default: 3)")
    args = parser.parse_args()

    # Output dir is just the backend name (approx-star and exact-star
    # produce identical results for both NNV and n2v on this model).
    output_dir = os.path.join(TESTS_DIR, "ground_truth", args.backend)

    if args.verify_determinism:
        verify_determinism(args.backend, output_dir, args.runs)
    else:
        generate(args.backend, output_dir)
