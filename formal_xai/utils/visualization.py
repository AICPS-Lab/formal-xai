"""Visualization and printing helpers."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

try:
    from colorama import Fore, Style

    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False


# ---------------------------------------------------------------------------
# Coloured console output
# ---------------------------------------------------------------------------

_COLOR_MAP = {
    "red": Fore.RED if _HAS_COLORAMA else "",
    "green": Fore.GREEN if _HAS_COLORAMA else "",
    "yellow": Fore.YELLOW if _HAS_COLORAMA else "",
    "blue": Fore.BLUE if _HAS_COLORAMA else "",
    "magenta": Fore.MAGENTA if _HAS_COLORAMA else "",
    "cyan": Fore.CYAN if _HAS_COLORAMA else "",
    "white": Fore.WHITE if _HAS_COLORAMA else "",
}

_RESET = Style.RESET_ALL if _HAS_COLORAMA else ""


def printc(*args, color: str = "red") -> None:
    """Print coloured text to the console.

    Args:
        *args: Values forwarded to ``print``.
        color: Colour name (red, green, yellow, blue, magenta, cyan, white).
    """
    prefix = _COLOR_MAP.get(color, "")
    print(prefix, *args, _RESET)


# ---------------------------------------------------------------------------
# Attribution colourmap
# ---------------------------------------------------------------------------

def get_custom_cmap() -> LinearSegmentedColormap:
    """Return a blue-white-red diverging colourmap for attribution overlays."""
    colours = ["blue", "white", "red"]
    return LinearSegmentedColormap.from_list("bwr_custom", colours, N=256)
