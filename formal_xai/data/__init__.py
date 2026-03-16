"""Dataset loaders for formal_xai.

Supports three data modalities:
- **Image**: MNIST, GTSRB, EMNIST, OrganMNIST  (``formal_xai.data.image``)
- **Tabular**: HELOC and generic CSV  (``formal_xai.data.tabular``)
- **Time Series**: TaxiNet and generic 1-D signals  (``formal_xai.data.timeseries``)
"""

from formal_xai.data.image import (
    get_sample_by_class,
    randomly_select_sample_by_class,
    randomly_select_sample,
    Selector,
    SimpleEMNIST,
    PermuteTransform,
    process_dataset_gtrsb,
)

__all__ = [
    "get_sample_by_class",
    "randomly_select_sample_by_class",
    "randomly_select_sample",
    "Selector",
    "SimpleEMNIST",
    "PermuteTransform",
    "process_dataset_gtrsb",
]
