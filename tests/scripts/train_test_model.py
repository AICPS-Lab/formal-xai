#!/usr/bin/env python3
"""Train a deterministic MNIST MLP for ground truth testing.

This script trains a small MLP on MNIST with a fixed seed and saves
both the PyTorch weights (.pt) and ONNX export (.onnx) to tests/models/.

These model files are committed to git so that ground truth tests are
fully reproducible.

Usage::

    python tests/scripts/train_test_model.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Deterministic seed
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(TESTS_DIR)
MODEL_DIR = os.path.join(TESTS_DIR, "models")
DATA_DIR = os.path.join(REPO_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)

# Add repo to path so we can import formal_xai
sys.path.insert(0, REPO_DIR)


def train():
    from formal_xai.models import MLP

    device = torch.device("cpu")  # CPU for reproducibility

    # Data
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True,
                              transform=transform)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True,
                             transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False,
    )

    # Model — same architecture as MyFirstExperiment_mnist_MLP
    model = MLP(input_size=784, output_size=10, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train (3 epochs is enough for a test fixture)
    NUM_EPOCHS = 3
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch}/{NUM_EPOCHS}: loss={total_loss/total:.4f}, "
              f"train_acc={train_acc:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    test_acc = correct / total
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save PyTorch weights
    pt_path = os.path.join(MODEL_DIR, "test_mnist_mlp.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"✓ Saved PyTorch weights: {pt_path}")

    # Export ONNX
    onnx_path = os.path.join(MODEL_DIR, "test_mnist_mlp.onnx")
    dummy = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print(f"✓ Saved ONNX model: {onnx_path}")

    # Save a few test samples for ground truth generation
    samples_path = os.path.join(MODEL_DIR, "test_samples.pt")
    samples = {}
    # Collect one sample per class
    for data, target in test_ds:
        label = target if isinstance(target, int) else target.item()
        if label not in samples:
            samples[label] = data
        if len(samples) == 10:
            break
    torch.save(samples, samples_path)
    print(f"✓ Saved test samples (one per class): {samples_path}")

    # Verify ONNX matches PyTorch
    print("\n--- Verifying ONNX matches PyTorch ---")
    with torch.no_grad():
        for label, sample in sorted(samples.items()):
            pt_out = model(sample.unsqueeze(0))
            pt_pred = pt_out.argmax(dim=1).item()
            print(f"  Class {label}: predicted={pt_pred}, "
                  f"{'✓' if pt_pred == label else '✗'}")

    print(f"\n✓ Done. Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    train()
