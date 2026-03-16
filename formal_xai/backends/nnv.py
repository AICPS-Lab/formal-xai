"""NNV (Neural Network Verification) backend via MATLAB Engine.

Requires:
- MATLAB Engine for Python (``matlabengine``)
- NNV toolbox installed in MATLAB (https://github.com/verivital/nnv)

The backend supports the following reachability methods:
- ``approx-star``: Over-approximate star-set reachability.
- ``exact-star``: Exact star-set reachability (may be slow).
- ``relax-star-area_<factor>``: Relaxed star with a given relaxation factor.
- ``cp-star``: Crown-propagation star method.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
import torch

from formal_xai.backends.base import VerificationBackend
from formal_xai.utils.math import is_float

try:
    import matlab.engine
except ImportError:
    raise ImportError(
        "NNV backend requires MATLAB Engine for Python.\n"
        "Install it with: pip install matlabengine\n"
        "See: https://www.mathworks.com/help/matlab/matlab_external/"
        "install-the-matlab-engine-for-python.html"
    )


logger = logging.getLogger(__name__)


class NNVBackend(VerificationBackend):
    """NNV verification backend using MATLAB Engine.

    Args:
        model_path: Path to the ONNX model file.
        output_size: Number of model outputs (classes).
        reach_method: Reachability method name.
        epsilon: Perturbation budget (L∞).
        engine: Optional pre-started MATLAB engine.
        num_workers: Number of MATLAB parallel workers.
    """

    VALID_METHODS = {"approx-star", "exact-star", "cp-star"}

    def __init__(
        self,
        model_path: str,
        output_size: int = 10,
        reach_method: str = "approx-star",
        epsilon: float = 35 / 255.0,
        engine=None,
        num_workers: int = 8,
    ):
        # Validate reach method
        valid = (
            reach_method in self.VALID_METHODS
            or reach_method.startswith("relax-star-area")
        )
        if not valid:
            raise ValueError(f"Invalid reach_method: {reach_method}")
        if reach_method.startswith("relax-star-area"):
            factor = reach_method.split("_")[-1]
            if not is_float(factor):
                raise ValueError(
                    f"relax-star-area requires a numeric relaxFactor, got: {factor}"
                )

        self.reach_method = reach_method
        self.model_path = model_path
        self.output_size = output_size
        self.epsilon = epsilon

        # Start or reuse MATLAB engine
        if engine is None:
            self.engine = matlab.engine.start_matlab()
            self._owns_engine = True
        else:
            self.engine = engine
            self._owns_engine = False
        logger.info("MATLAB engine ready")

        # Load network
        matlab_net = self.engine.importNetworkFromONNX(
            self.model_path,
            "InputDataFormats", "BCSS",
            "OutputDataFormats", "BC",
            nargout=1,
        )

        if reach_method == "cp-star":
            self.engine.workspace["net"] = matlab_net
        else:
            net = self.engine.matlab2nnv(matlab_net, nargout=1)
            self.engine.workspace["net"] = net

        # Configure parallel pool
        self._setup_parallel_pool(num_workers)

    # ------------------------------------------------------------------
    # Parallel pool
    # ------------------------------------------------------------------

    def _setup_parallel_pool(self, num_workers: int) -> None:
        try:
            self.engine.eval("poolobj = gcp('nocreate');", nargout=0)
            exists = bool(self.engine.eval("~isempty(poolobj)", nargout=1))
            if not exists:
                logger.info("Starting parallel pool with %d workers", num_workers)
                self.engine.eval(f"parpool('Processes', {num_workers});", nargout=0)
            else:
                n = self.engine.eval("poolobj.NumWorkers", nargout=1)
                logger.info("Reusing parallel pool with %s workers", n)
        except Exception as e:
            logger.warning("Parallel pool setup failed: %s", e)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_ranges(
        self,
        img: torch.Tensor,
        ub: torch.Tensor,
        lb: torch.Tensor,
        target: Union[int, Tuple[int, int]],
    ) -> Tuple[Tuple, object]:
        """Compute output reachable-set ranges via NNV."""
        reach = self.reach_method
        relax_factor = None
        if reach.startswith("relax-star-area"):
            relax_factor = float(reach.split("_")[-1])
            reach = "relax-star-area"

        if reach != "cp-star":
            self.engine.eval(
                f"net.OutputSize = {self.output_size};", nargout=0
            )

        # Permute CHW → HWC for MATLAB
        if img.dim() == 3:
            img = img.contiguous().permute(1, 2, 0).contiguous()
            lb = lb.contiguous().permute(1, 2, 0).contiguous()
            ub = ub.contiguous().permute(1, 2, 0).contiguous()

        # Transfer to MATLAB workspace
        self.engine.workspace["img"] = self.engine.single(img.detach().numpy())
        self.engine.workspace["lb_clip"] = self.engine.single(lb.detach().numpy())
        self.engine.workspace["ub_clip"] = self.engine.single(ub.detach().numpy())

        # Target specification
        if isinstance(target, tuple):
            self.engine.workspace["G1"] = 3.0
            self.engine.workspace["g1"] = float(target[1])
            self.engine.workspace["G2"] = float(target[0])
            self.engine.workspace["g2"] = -3.0
            self.engine.eval("U1 = HalfSpace(G1,g1);", nargout=0)
            self.engine.eval("U2 = HalfSpace(G2,g2);", nargout=0)
            self.engine.eval("target = [U1,U2];", nargout=0)
        elif isinstance(target, int):
            self.engine.workspace["target"] = self.engine.single(target + 1)
        else:
            raise ValueError(f"Invalid target: {target}")

        self.engine.workspace["epsilon"] = self.epsilon
        self.engine.eval("IS = ImageStar(lb_clip, ub_clip);", nargout=0)
        self.engine.eval(f"reachOptions.reachMethod = '{reach}';", nargout=0)

        if relax_factor is not None:
            self.engine.eval(
                f"reachOptions.relaxFactor = {relax_factor};", nargout=0
            )

        # Verification
        if reach == "cp-star":
            self.engine.eval("reachOptions.train_device = 'cpu';", nargout=0)
            res = self.engine.eval(
                f"verify_robustness_cp(net, IS, reachOptions, target, {self.output_size});",
                nargout=1,
            )
            return (None, None), res

        res = self.engine.eval(
            "net.verify_robustness(IS, reachOptions, target);", nargout=1
        )
        self.engine.eval("R = net.reachSet{end};", nargout=0)

        if reach == "exact-star":
            lb_out = np.ones(self.output_size) * 1000
            ub_out = np.ones(self.output_size) * -1000
            R_len = int(self.engine.eval("length(R);", nargout=1))
            for i in range(1, R_len + 1):
                lb_t, ub_t = self.engine.eval(f"R({i}).getRanges;", nargout=2)
                lb_out = np.minimum(lb_out, np.squeeze(np.array(lb_t)))
                ub_out = np.maximum(ub_out, np.squeeze(np.array(ub_t)))
        else:
            lb_out, ub_out = self.engine.eval("R.getRanges;", nargout=2)
            lb_out = np.squeeze(np.array(lb_out))
            ub_out = np.squeeze(np.array(ub_out))

        return (lb_out, ub_out), res

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down MATLAB parallel pool and engine."""
        engine = getattr(self, "engine", None)
        if engine is None:
            return
        try:
            engine.eval("poolobj = gcp('nocreate');", nargout=0)
            if bool(engine.eval("~isempty(poolobj)", nargout=1)):
                engine.eval("delete(poolobj);", nargout=0)
        except Exception as e:
            logger.warning("Pool cleanup error: %s", e)
        if getattr(self, "_owns_engine", False):
            try:
                engine.quit()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
