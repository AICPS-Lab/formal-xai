#!/usr/bin/env python3
"""Import tests for formal_xai.

Verifies that all modules import cleanly and key classes are accessible.
"""

import importlib
import sys


def test_imports():
    """Test that all formal_xai modules can be imported."""
    modules = [
        "formal_xai",
        "formal_xai.utils",
        "formal_xai.utils.device",
        "formal_xai.utils.seed",
        "formal_xai.utils.visualization",
        "formal_xai.utils.math",
        "formal_xai.models",
        "formal_xai.models.mlp",
        "formal_xai.models.cnn",
        "formal_xai.data",
        "formal_xai.data.image",
        "formal_xai.data.tabular",
        "formal_xai.data.timeseries",
        "formal_xai.baselines",
        "formal_xai.baselines.lime_explainer",
        "formal_xai.baselines.anchors_explainer",
        "formal_xai.baselines.prototype_explainer",
        "formal_xai.baselines.tsa_explainer",
        "formal_xai.vitax",
        "formal_xai.vitax.heuristic",
        "formal_xai.vitax.explainer",
        "formal_xai.backends",
        "formal_xai.backends.base",
        # NNV and Marabou backends are optional — tested separately
    ]

    passed = 0
    failed = 0
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            print(f"  ✓ {mod_name}")
            passed += 1
        except ImportError as e:
            print(f"  ✗ {mod_name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(modules)} modules")
    return failed == 0


def test_classes():
    """Test that key classes can be instantiated or accessed."""
    from formal_xai.vitax import VitaX, HeuristicRanker
    from formal_xai.models import MLP, CNN, SmallMLP, MLP_DENSE, MLP_DENSE_LARGE, CNN_DENSE, CNN_taxi
    from formal_xai.baselines import LIMEExplainer, AnchorsExplainer, PrototypeExplainer, TSAExplainer
    from formal_xai.utils import get_device, seed
    from formal_xai.backends.base import VerificationBackend

    classes = [
        VitaX, HeuristicRanker,
        MLP, CNN, SmallMLP, MLP_DENSE, MLP_DENSE_LARGE, CNN_DENSE, CNN_taxi,
        LIMEExplainer, AnchorsExplainer, PrototypeExplainer, TSAExplainer,
        VerificationBackend,
    ]
    for cls in classes:
        print(f"  ✓ {cls.__module__}.{cls.__name__}")

    funcs = [get_device, seed]
    for fn in funcs:
        print(f"  ✓ {fn.__module__}.{fn.__name__}")

    print(f"\nAll {len(classes)} classes and {len(funcs)} functions accessible")
    return True


def test_models_forward():
    """Smoke test: create model instances and run a forward pass."""
    import torch
    from formal_xai.models import MLP, CNN, SmallMLP, MLP_DENSE, CNN_DENSE

    tests = [
        ("MLP", MLP(784, 10, 1), torch.randn(1, 1, 28, 28)),
        ("SmallMLP", SmallMLP(784, 10, 1), torch.randn(1, 1, 28, 28)),
        ("MLP_DENSE", MLP_DENSE(784, 10, 1), torch.randn(1, 1, 28, 28)),
        ("CNN", CNN(1, 10), torch.randn(1, 1, 28, 28)),
        ("CNN_DENSE", CNN_DENSE(1, 10, (28, 28)), torch.randn(1, 1, 28, 28)),
    ]

    for name, model, x in tests:
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"  ✓ {name}: input={list(x.shape)} → output={list(out.shape)}")

    print(f"\n{len(tests)} model forward passes OK")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("formal_xai — Import Tests")
    print("=" * 50)

    print("\n1. Module imports:")
    ok1 = test_imports()

    print("\n2. Class accessibility:")
    ok2 = test_classes()

    print("\n3. Model forward passes:")
    ok3 = test_models_forward()

    print("\n" + "=" * 50)
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
