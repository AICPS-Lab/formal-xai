#!/usr/bin/env python3
"""Run VitaX on MNIST.

Usage::

    python experiments/run_vitax.py
    python experiments/run_vitax.py --backend marabou --heuristic ig
    python experiments/run_vitax.py --epsilon 0.05 --target 3 --counterfactual 7
"""

import argparse
import sys

import torch
from torchvision import datasets, transforms


def parse_args():
    p = argparse.ArgumentParser(description="Run VitaX verified attribution")

    # Model
    p.add_argument("--weights", default="models/MyFirstExperiment_mnist_MLP.pt",
                    help="Path to PyTorch model weights (.pt)")
    p.add_argument("--onnx", default="outputs/MyFirstExperiment_mnist_mlp.onnx",
                    help="Path to ONNX model for verification backend")

    # Data
    p.add_argument("--dataset", default="mnist", choices=["mnist"],
                    help="Dataset to use (default: mnist)")
    p.add_argument("--data-dir", default="./data",
                    help="Root directory for dataset download")
    p.add_argument("--num-classes", type=int, default=10,
                    help="Number of output classes")
    p.add_argument("--input-size", type=int, default=784,
                    help="Flattened input size (28*28 for MNIST)")

    # VitaX
    p.add_argument("--backend", default="n2v", choices=["n2v", "nnv", "marabou"],
                    help="Verification backend (default: n2v)")
    p.add_argument("--reach-method", default="approx-star",
                    help="Reachability method (default: approx-star)")
    p.add_argument("--heuristic", default="sa",
                    choices=["sa", "ig", "dl", "svs", "random"],
                    help="Feature ranking heuristic (default: sa)")
    p.add_argument("--epsilon", type=float, default=25/255,
                    help="L-inf perturbation radius (default: 25/255)")

    # Sample selection
    p.add_argument("--target", type=int, default=7,
                    help="Target class label of the sample (default: 7)")
    p.add_argument("--counterfactual", type=int, default=3,
                    help="Counterfactual class to check against (default: 3)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


def main():
    args = parse_args()

    from formal_xai.vitax import VitaX
    from formal_xai.models import MLP
    from formal_xai.data import get_sample_by_class
    from formal_xai.utils import get_device, seed

    seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # --- Load model ---
    model = MLP(input_size=args.input_size, output_size=args.num_classes,
                input_channels=1)
    model.load_state_dict(
        torch.load(args.weights, map_location=torch.device("cpu"))
    )
    model.eval()
    print("✓ Model loaded")

    # --- Load data ---
    test_ds = datasets.MNIST(
        root=args.data_dir, train=False, download=True,
        transform=transforms.ToTensor(),
    )

    # --- Create VitaX ---
    try:
        backend_kwargs = {}
        if args.backend == "n2v":
            backend_kwargs["model"] = model
        verifier = VitaX(
            model_path=args.onnx,
            backend=args.backend,
            reach_method=args.reach_method,
            heuristic_method=args.heuristic,
            epsilon=args.epsilon,
            num_classes=args.num_classes,
            **backend_kwargs,
        )
    except ImportError as e:
        print(f"Backend not available: {e}")
        print(f"This example requires the {args.backend} backend.")
        sys.exit(1)

    print("✓ VitaX initialized")

    # --- Run on a sample ---
    sample = get_sample_by_class(test_ds, class_label=args.target)
    print(f"Sample shape: {sample.shape}")

    attr, is_robust = verifier.explain(
        model,
        sample,
        target=args.target,
        class_to_check=args.counterfactual,
        return_robustness=True,
        show_progress=True,
    )

    print(f"✓ Result: robust={is_robust}")
    print(f"  Attribution shape: {attr.shape}")
    print(f"  Non-zero features: {(attr != 0).flatten().sum().item()}")

    verifier.close()
    print("✓ Done")


if __name__ == "__main__":
    main()
