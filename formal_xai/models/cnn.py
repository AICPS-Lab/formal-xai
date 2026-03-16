"""CNN model architectures."""

import torch.nn as nn


class CNN(nn.Module):
    """Single-conv CNN with average pooling (conv1 → AvgPool → FC).

    Designed for 28×28 input images.
    """

    def __init__(self, input_channel: int, output_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * 14 * 14, output_size)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
        x = x.reshape(-1, 8 * 14 * 14)
        x = self.fc1(x)
        return x


class CNN_DENSE(nn.Module):
    """Two-conv CNN with dense head.

    Args:
        input_channel: Number of input channels.
        output_size: Number of output classes.
        input_size: Spatial dimensions ``(H, W)`` of the input.
    """

    def __init__(self, input_channel: int, output_size: int, input_size=(28, 28)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4 * input_size[0] * input_size[1], 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.Flatten(start_dim=1)(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_taxi(nn.Module):
    """Fully-connected model for TaxiNet regression (27×54 input).

    Despite the name, this is an MLP applied to flattened spatial input
    and outputs a single regression value.
    """

    def __init__(self, input_channel: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(27 * 54 * input_channel, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, output_size)

    def forward(self, x):
        x = nn.Flatten(start_dim=1)(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
