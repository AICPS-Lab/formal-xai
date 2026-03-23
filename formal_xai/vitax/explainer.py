"""VitaX — Formally Verified Attribution Explainer.

The core algorithm:
1. **Rank** features by importance using a heuristic (SA, IG, etc.)
2. **Binary search** over ranked features to find the minimal set that
   causes a robustness violation
3. **Verify** robustness at each step via a formal verification backend
   (NNV or Marabou)

Ported and renamed from ``verified_xai`` in the original repository.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from formal_xai.vitax.heuristic import HeuristicRanker


class VitaX:
    """Formally verified attribution explainer.

    Args:
        model_path: Path to the ONNX model for the verification backend.
        backend: Verification backend name (``"nnv"`` or ``"marabou"``).
        reach_method: Reachability method (backend-specific).
        heuristic_method: Feature ranking heuristic (``"sa"``, ``"ig"``…).
        epsilon: L∞ perturbation budget.
        num_classes: Number of output classes (``1`` for regression).
        save_solver_time: If ``True``, record per-step solver times.
        backend_kwargs: Extra keyword arguments forwarded to the backend.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "n2v",
        reach_method: str = "approx-star",
        heuristic_method: str = "sa",
        epsilon: float = 35 / 255.0,
        num_classes: int = 10,
        save_solver_time: bool = False,
        **backend_kwargs,
    ):
        self.model_path = model_path
        self.reach_method = reach_method
        self.heuristic_method = heuristic_method
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.save_solver_time = save_solver_time
        self.solver_time: List[float] = []

        # Instantiate the verification backend
        self._backend = self._create_backend(
            backend, model_path, reach_method, epsilon, num_classes, **backend_kwargs
        )

    # ------------------------------------------------------------------
    # Backend factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_backend(
        name: str,
        model_path: str,
        reach_method: str,
        epsilon: float,
        num_classes: int,
        **kwargs,
    ):
        if name == "n2v":
            from formal_xai.backends.n2v import N2VPyBackend

            model = kwargs.pop("model", None)
            if model is None:
                raise ValueError(
                    "The n2v backend requires a PyTorch model.  Pass "
                    "model=<nn.Module> when creating VitaX with backend='n2v'."
                )
            return N2VPyBackend(
                model=model,
                output_size=num_classes,
                reach_method=reach_method,
                epsilon=epsilon,
                model_path=model_path,
            )
        if name == "nnv":
            from formal_xai.backends.nnv import NNVBackend

            return NNVBackend(
                model_path=model_path,
                output_size=num_classes,
                reach_method=reach_method,
                epsilon=epsilon,
                **kwargs,
            )
        if name == "marabou":
            from formal_xai.backends.marabou import MarabouBackend

            return MarabouBackend(
                model_path=model_path,
                epsilon=epsilon,
                **kwargs,
            )
        raise ValueError(f"Unknown backend: {name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        target: Union[int, Tuple[int, int]],
        class_to_check: int,
        visual: bool = False,
        save_fig: bool = False,
        show_progress: bool = False,
        return_robustness: bool = False,
        raw: bool = False,
        seed: int = 42,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, bool]]:
        """Run VitaX to produce a verified attribution map.

        Args:
            model: PyTorch model (eval mode, same model exported to ONNX).
            img: Input tensor (batched ``(1, C, H, W)`` or un-batched).
            target: True class (int) or regression target range ``(lo, hi)``.
            class_to_check: Counterfactual class to verify against.
                Use ``-1`` for regression tasks.
            visual: Show matplotlib visualisation.
            save_fig: Save figures to ``./outputs/``.
            show_progress: Display tqdm progress bar.
            return_robustness: If ``True``, also return a bool indicating
                whether the model is robust.
            raw: Return raw upper/lower masks.
            seed: Random seed.

        Returns:
            Attribution tensor, or ``(attribution, is_robust)`` if
            ``return_robustness=True``.
        """
        # Step 1: rank features by heuristic importance
        ranker = HeuristicRanker(model, epsilon=self.epsilon, seed=seed)
        indices, sorted_img, importance = ranker.rank_features(
            img,
            target=class_to_check,
            method=self.heuristic_method,
            normalized="directional",
        )

        # Step 2: binary search for the minimal non-robust region
        attr, unrobust_targets = self._binary_search_adaptive(
            img,
            importance,
            indices,
            target,
            class_to_check,
            visual=visual,
            save_fig=save_fig,
            show_progress=show_progress,
            raw=raw,
        )

        if return_robustness:
            return attr, len(unrobust_targets) == 0
        return attr

    # Alias: keep backward compat with the old `forward()` name
    forward = explain

    def iterate_all_counterfactual(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        target: int,
        visual: bool = False,
        save_fig: bool = False,
    ) -> list:
        """Run VitaX against every other class."""
        results = []
        for c in range(self.num_classes):
            if c == target:
                continue
            results.append(
                self.explain(
                    model, img, target, c, visual=visual, save_fig=save_fig
                )
            )
        return results

    # ------------------------------------------------------------------
    # Robustness checks
    # ------------------------------------------------------------------

    @staticmethod
    def robustness_given_range(
        lb_out: float, ub_out: float, target: Tuple[int, int]
    ) -> bool:
        """Regression robustness: output stays within ``[target[0], target[1]]``."""
        return lb_out > target[0] and ub_out < target[1]

    @staticmethod
    def robustness_given_class(
        lb_out: np.ndarray,
        ub_out: np.ndarray,
        class_target: int,
        class_to_check: int,
        num_classes: int = 10,
    ) -> Tuple[bool, List[int]]:
        """Classification robustness check.

        Returns ``(is_robust, list_of_threat_classes)``.
        """
        threats: List[int] = []
        for i in range(num_classes):
            if i == class_target:
                continue
            if ub_out[class_to_check] <= ub_out[i] and i != class_to_check:
                threats.append(i)

        if class_target != class_to_check:
            if lb_out[class_target] <= ub_out[class_to_check]:
                threats.append(class_to_check)
            return lb_out[class_target] > ub_out[class_to_check], threats
        else:
            max_ub = max(ub_out[i] for i in range(num_classes) if i != class_target)
            return lb_out[class_target] > max_ub, threats

    # ------------------------------------------------------------------
    # Important features extraction
    # ------------------------------------------------------------------

    @staticmethod
    def important_features(
        data: torch.Tensor,
        grad: torch.Tensor,
        ub_clip: torch.Tensor,
        lb_clip: torch.Tensor,
        raw: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract directional attribution from verified bounds.

        If a pixel is at the boundary (0 or 1) and the gradient says to
        push it further in that direction, the pixel is not actually
        important — it cannot change.

        Returns:
            ``(attribution, res_ub_mask, res_lb_mask)``
        """
        ub_clip = ub_clip.squeeze()
        lb_clip = lb_clip.squeeze()
        img = data.squeeze()

        # Upper bound contributions
        diff_ub = (ub_clip - img).squeeze()
        res_ub = torch.where(
            torch.logical_and(diff_ub > 0, grad > 0),
            torch.ones_like(img),
            torch.zeros_like(img),
        )

        # Lower bound contributions
        diff_lb = (img - lb_clip).squeeze()
        res_lb = torch.where(
            torch.logical_and(diff_lb > 0, grad < 0),
            torch.ones_like(img),
            torch.zeros_like(img),
        )

        if raw:
            return ub_clip - lb_clip, res_ub, res_lb

        attribution = torch.zeros_like(img)
        attribution = torch.where(res_ub == 1, ub_clip - img, attribution)
        attribution = torch.where(res_lb == 1, lb_clip - img, attribution)
        return attribution, res_ub, res_lb

    # ------------------------------------------------------------------
    # Condition check (single verification step)
    # ------------------------------------------------------------------

    def _condition_met(
        self,
        indices: torch.Tensor,
        img: torch.Tensor,
        importance: torch.Tensor,
        target: Union[int, Tuple[int, int]],
        class_to_check: int,
        raw: bool = False,
    ) -> Tuple[bool, dict]:
        """Check robustness when perturbing features at *indices*."""
        t0 = time.perf_counter()

        half_indices = torch.unravel_index(indices, img.squeeze().shape)
        ub = torch.clone(img)
        lb = torch.clone(img)
        ub[half_indices] = ub[half_indices] + self.epsilon
        lb[half_indices] = lb[half_indices] - self.epsilon
        ub = torch.minimum(ub, torch.ones_like(ub))
        lb = torch.maximum(lb, torch.zeros_like(lb))

        (lb_out, ub_out), res = self._backend.get_ranges(img, ub, lb, target)

        # Determine robustness
        if self.reach_method == "cp-star":
            is_robust = int(res) == 1
            threats = [] if is_robust else [class_to_check]
        elif isinstance(target, tuple):
            is_robust = self.robustness_given_range(lb_out, ub_out, target)
            threats = []
        elif isinstance(target, int):
            is_robust, threats = self.robustness_given_class(
                lb_out, ub_out, target, class_to_check, self.num_classes
            )
        else:
            raise ValueError(f"Invalid target: {target}")

        attribution, res_ub, res_lb = self.important_features(
            img, importance, ub, lb, raw=raw
        )

        elapsed = time.perf_counter() - t0
        return is_robust, {
            "threats": threats,
            "res_ub": res_ub,
            "res_lb": res_lb,
            "attribution": attribution.squeeze(),
            "time": elapsed,
        }

    # ------------------------------------------------------------------
    # Binary search
    # ------------------------------------------------------------------

    def _binary_search_adaptive(
        self,
        img: torch.Tensor,
        importance: torch.Tensor,
        indices: torch.Tensor,
        target: Union[int, Tuple[int, int]],
        class_to_check: int,
        visual: bool = False,
        save_fig: bool = False,
        show_progress: bool = False,
        raw: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Binary search for the minimal set of important features.

        Splits the sorted indices and checks robustness at each midpoint.
        Converges to the smallest prefix of ranked features that still
        causes a robustness violation.
        """
        start = 0
        end = len(indices) - 1
        mid = end
        max_iters = math.ceil(math.log2(len(indices)))

        pbar = tqdm(total=max_iters) if show_progress else None

        while start <= end:
            mid = (start + end) // 2
            is_robust, info = self._condition_met(
                indices[: mid + 1],
                img.squeeze(),
                importance,
                target,
                class_to_check,
                raw=raw,
            )
            if self.save_solver_time:
                self.solver_time.append(info["time"])

            if is_robust:
                start = mid + 1
            else:
                end = mid - 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Features checked: {mid + 1}")

        if pbar is not None:
            pbar.close()

        # Final verification at converged midpoint
        final_robust, final_info = self._condition_met(
            indices[: mid + 1],
            img.squeeze(),
            importance,
            target,
            class_to_check,
            raw=raw,
        )
        if not final_robust:
            mid = max(mid - 1, 0)
            _, final_info = self._condition_met(
                indices[: mid + 1],
                img.squeeze(),
                importance,
                target,
                class_to_check,
                raw=raw,
            )

        return final_info["attribution"], final_info["threats"]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release backend resources."""
        self._backend.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
