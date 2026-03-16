"""MLP model architectures."""

import torch.nn as nn


class MLP(nn.Module):
    """Standard 3-layer MLP (input → 128 → 64 → output)."""

    def __init__(self, input_size: int, output_size: int, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.fc1 = nn.Linear(input_size * input_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = nn.Flatten(start_dim=1)(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x


class SmallMLP(nn.Module):
    """Compact 2-layer MLP (input → output+5 → output)."""

    def __init__(self, input_size: int, output_size: int, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.fc1 = nn.Linear(input_size * input_channels, output_size + 5)
        self.fc2 = nn.Linear(output_size + 5, output_size)

    def forward(self, x):
        x = nn.Flatten(start_dim=1)(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP_DENSE(nn.Module):
    """Dense 3-layer MLP with narrow hidden layers (input → 10 → 10 → output)."""

    def __init__(self, input_size: int, output_size: int, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.fc1 = nn.Linear(input_size * input_channels, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_size)

    def forward(self, x):
        x = nn.Flatten(start_dim=1)(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_DENSE_LARGE(nn.Module):
    """Dense 3-layer MLP with wider hidden layers (input → 30 → 30 → output)."""

    def __init__(self, input_size: int, output_size: int, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.fc1 = nn.Linear(input_size * input_channels, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, output_size)

    def forward(self, x):
        x = nn.Flatten(start_dim=1)(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
