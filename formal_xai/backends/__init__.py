"""Verification backends for formal_xai.

Each backend implements :class:`VerificationBackend` and provides
``get_ranges(img, ub, lb, target)`` for reachability analysis.

Available backends:
- **NNV** (MATLAB): ``formal_xai.backends.nnv``
- **Marabou**: ``formal_xai.backends.marabou``

Both are optional dependencies; import errors are handled gracefully.
"""

from formal_xai.backends.base import VerificationBackend

__all__ = ["VerificationBackend"]

# Lazy imports — only fail when actually used, not on package import.
# We catch (ImportError, OSError) because MATLAB Engine may raise OSError
# when shared library versions are incompatible.
try:
    from formal_xai.backends.nnv import NNVBackend
    __all__.append("NNVBackend")
except (ImportError, OSError):
    pass

try:
    from formal_xai.backends.marabou import MarabouBackend
    __all__.append("MarabouBackend")
except (ImportError, OSError):
    pass
