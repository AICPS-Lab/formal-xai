# formal-xai

**ViTaX** — *Towards Verified and Targeted Explanations through Formal Methods*

Formally verified attribution explanations for neural networks.

## Overview

ViTaX produces attribution maps that are backed by formal verification — it doesn't just estimate feature importance, it *proves* which features are sufficient to change the model's prediction under L∞ perturbation.

## Visualizations

Here are visualizations showing what ViTaX does (left side shows the original image embedded with the generation of explanation, right side shows the robustness checking in logits space):

### MNIST explanation from 2 to 3
<img src="mnist_2to3.gif" alt="MNIST" width="600"/>

- Time taken: 14.06 seconds
- Solver avg time: 0.96 seconds
- A (important) set: 130 | B (irrelevant) set: 654

### GTSRB explanation from speed limit 50 to 60
<img src="gtsrb_50to60.gif" alt="GTSRB" width="600"/>

- Time taken: 86.98 seconds
- Solver avg time: 6.45 seconds
- A (important) set: 104 | B (irrelevant) set: 2248

### EMNIST explanation from n to m (robust example)
<img src="emnist_ntom.gif" alt="EMNIST n to m" width="600"/>

- Time taken: 37.19 seconds
- Solver avg time: 3.08 seconds
- A (important) set: 35 | B (irrelevant) set: 749

### EMNIST explanation from n to C (less robust example)
<img src="emnist_ntoC.gif" alt="EMNIST n to C" width="600"/>

- Time taken: 106.67 seconds
- Solver avg time: 8.70 seconds
- A (important) set: 93 | B (irrelevant) set: 691

## Algorithm

1. **Rank** features by heuristic importance (Saliency, IG, DeepLift, SHAP, or Random)
2. **Binary search** over ranked features to find the minimal sensitive set
3. **Verify** robustness at each step via a formal verification backend (NNV/MATLAB or Marabou)

<img src="ViTaX.png" alt="ViTaX Framework" width="800"/>

### Data Modalities

| Modality | Examples | Loader |
|----------|----------|--------|
| Image | MNIST, GTSRB, EMNIST, OrganMNIST | `formal_xai.data.image` |
| Tabular | HELOC, CSV datasets | `formal_xai.data.tabular` |
| Time Series | TaxiNet, generic 1-D signals | `formal_xai.data.timeseries` |

## Installation

```bash
pip install -e .                # Core only
pip install -e ".[captum]"      # + Captum for IG/DeepLift/SHAP heuristics
pip install -e ".[all]"         # + all optional Python deps
```

### NNV Backend (MATLAB)

Requires MATLAB with the [NNV toolbox](https://github.com/verivital/nnv):

```bash
pip install matlabengine
```

### Marabou Backend

Requires [Maraboupy](https://github.com/NeuralNetworkVerification/Marabou).

## Quick Start

### CLI

```bash
# Default (MNIST, MLP, NNV backend, saliency heuristic)
python experiments/run_vitax.py

# Custom parameters
python experiments/run_vitax.py --backend marabou --heuristic ig --epsilon 0.05
python experiments/run_vitax.py --target 3 --counterfactual 7
python experiments/run_vitax.py --help   # see all options
```

### Python API

```python
from formal_xai.vitax import VitaX
from formal_xai.models import MLP
import torch

# Load model
model = MLP(input_size=784, output_size=10, input_channels=1)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Create VitaX explainer
verifier = VitaX(
    model_path="model.onnx",
    backend="nnv",
    reach_method="approx-star",
    heuristic_method="sa",
    epsilon=25/255,
    num_classes=10,
)

# Generate verified attribution
attr, is_robust = verifier.explain(
    model, image, target=7, class_to_check=3,
    return_robustness=True,
)
```

## Package Structure

```
formal_xai/
├── vitax/          # Core VitaX method
│   ├── explainer.py    # VitaX class
│   └── heuristic.py    # Feature ranker
├── backends/       # Verification backends
│   ├── base.py         # Abstract interface
│   ├── nnv.py          # NNV / MATLAB
│   └── marabou.py      # VeriX / Marabou
├── baselines/      # Baseline explainers
│   ├── lime_explainer.py
│   ├── anchors_explainer.py
│   ├── prototype_explainer.py
│   └── tsa_explainer.py
├── models/         # Model architectures
│   ├── mlp.py
│   └── cnn.py
├── data/           # Dataset loaders
│   ├── image.py
│   ├── tabular.py
│   └── timeseries.py
└── utils/          # Utilities
    ├── device.py
    ├── seed.py
    ├── visualization.py
    └── math.py
```

## Baselines

| Method | Class | Description |
|--------|-------|-------------|
| LIME | `LIMEExplainer` | Local surrogate linear model |
| Anchors | `AnchorsExplainer` | Rule-based sufficient conditions |
| Prototypes | `PrototypeExplainer` | Example-based nearest prototypes |
| TSA | `TSAExplainer` | Targeted semi-factual adversarial |

## License

MIT
