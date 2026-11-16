"""Fully connected factored eikonal networks and configurable input features."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor, nn


class AdaptiveAtan(nn.Module):
    """Neuron-wise L-LAAF activation atan(n * a_k * z), a_k initially 1/n."""

    def __init__(self, width: int, scale: float = 10.0) -> None:
        super().__init__()
        if width <= 0 or scale <= 0:
            raise ValueError("Adaptive activation width and scale must be positive")
        self.scale = float(scale)
        self.slopes = nn.Parameter(torch.full((width,), 1.0 / self.scale))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.atan(self.scale * self.slopes * inputs)


class Atan(nn.Module):
    """Ordinary arctangent activation."""

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.atan(inputs)


class Sine(nn.Module):
    """Sinusoidal activation."""

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sin(inputs)


class FourierFeatures(nn.Module):
    """Random sin/cos features of normalized coordinates; B is checkpointed."""

    def __init__(self, input_dim: int, size: int = 32, scale: float = 1.0,
                 include_raw: bool = True) -> None:
        super().__init__()
        if input_dim <= 0 or size <= 0 or scale <= 0:
            raise ValueError("Fourier dimensions and scale must be positive")
        self.register_buffer("frequencies", torch.randn(input_dim, size) * scale)
        self.include_raw = include_raw
        self.output_dim = 2 * size + (input_dim if include_raw else 0)

    def forward(self, inputs: Tensor) -> Tensor:
        phase = 2.0 * math.pi * inputs @ self.frequencies
        encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return torch.cat((inputs, encoded), dim=-1) if self.include_raw else encoded


def activation_layer(kind: str, width: int, scale: float) -> nn.Module:
    """Build one activation; adaptive parameters are independent per layer."""
    choices = {"tanh": nn.Tanh, "atan": Atan, "sin": Sine}
    if kind == "adaptive_atan":
        return AdaptiveAtan(width, scale)
    if kind not in choices:
        raise ValueError("Unknown activation: " + kind)
    return choices[kind]()


class EikonalMLP(nn.Module):
    """Predict tau(x) or tau(x,xs); coordinates supplied in physical km.

    Normalization is part of the graph, so coordinate derivatives obtained
    with autograd automatically have the correct physical scaling.
    """

    def __init__(self, bounds: Sequence[Sequence[float]], hidden_layers: int = 10,
                 width: int = 20, activation: str = "adaptive_atan",
                 activation_scale: float = 10.0, fourier: Optional[Dict[str, Any]] = None,
                 surrogate: bool = False) -> None:
        super().__init__()
        if hidden_layers < 1 or width < 1:
            raise ValueError("MLP depth and width must be positive")
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32)
        if tuple(bounds_tensor.shape) != (3, 2) or bool(torch.any(bounds_tensor[:, 1] <= bounds_tensor[:, 0])):
            raise ValueError("bounds must contain three increasing pairs")
        self.register_buffer("lower", bounds_tensor[:, 0])
        self.register_buffer("extent", bounds_tensor[:, 1] - bounds_tensor[:, 0])
        self.surrogate = surrogate
        input_dim = 6 if surrogate else 3
        options = fourier or {"enabled": False}
        self.encoding: nn.Module = nn.Identity()
        if options.get("enabled", False):
            self.encoding = FourierFeatures(input_dim, options.get("size", 32),
                                            options.get("scale", 1.0), options.get("include_raw", True))
            input_dim = self.encoding.output_dim
        layers = []
        for layer_index in range(hidden_layers):
            linear = nn.Linear(input_dim if layer_index == 0 else width, width)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, activation_layer(activation, width, activation_scale)))
        output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(output.weight, gain=0.1)
        nn.init.ones_(output.bias)
        layers.append(output)  # A linear output; the loss enforces positivity.
        self.layers = nn.Sequential(*layers)

    def forward(self, points: Tensor, sources: Optional[Tensor] = None) -> Tensor:
        coordinates = 2.0 * (points - self.lower) / self.extent - 1.0
        if self.surrogate:
            if sources is None:
                raise ValueError("Surrogate prediction requires source coordinates")
            normalized_sources = 2.0 * (sources - self.lower) / self.extent - 1.0
            normalized_sources = torch.broadcast_to(normalized_sources, coordinates.shape)
            coordinates = torch.cat((coordinates, normalized_sources), dim=-1)
        return self.layers(self.encoding(coordinates))


def build_network(config: Dict[str, Any]) -> EikonalMLP:
    """Build the architecture described in the effective YAML config."""
    return EikonalMLP(config["domain"]["bounds"], **config["network"],
                       surrogate=config["surrogate"]["enabled"])
