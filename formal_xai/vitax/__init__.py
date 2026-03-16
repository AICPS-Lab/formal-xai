"""VitaX — Formally Verified Attribution Explainer.

Usage::

    from formal_xai.vitax import VitaX, HeuristicRanker

    verifier = VitaX(
        model_path="model.onnx",
        backend="nnv",
        reach_method="approx-star",
        heuristic_method="sa",
        epsilon=25/255,
        num_classes=10,
    )
    attr = verifier.explain(model, image, target=7, class_to_check=3)
"""

from formal_xai.vitax.explainer import VitaX
from formal_xai.vitax.heuristic import HeuristicRanker

__all__ = ["VitaX", "HeuristicRanker"]
