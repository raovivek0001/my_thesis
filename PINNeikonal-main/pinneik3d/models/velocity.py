"""Isotropic velocities in km/s evaluated at physical coordinates in km."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


class VelocityModel:
    """Interface shared by analytic functions and sampled velocity volumes."""

    def numpy(self, points: np.ndarray) -> np.ndarray:
        """Return velocities with shape ``points.shape[:-1]``."""
        raise NotImplementedError

    def torch(self, points: Any) -> Any:
        """Return a differentiable tensor with shape ``(..., 1)``."""
        raise NotImplementedError

    def __call__(self, points: Any) -> Any:
        if isinstance(points, np.ndarray):
            return self.numpy(points)
        return self.torch(points)


@dataclass
class ConstantGradient(VelocityModel):
    """Linear velocity ``v0 + gradient . (x - origin)``."""

    v0: float = 2.0
    gradient: Sequence[float] = (0.0, 0.0, 0.5)
    origin: Sequence[float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.v0) or self.v0 <= 0 or len(self.gradient) != 3 or len(self.origin) != 3:
            raise ValueError("v0 must be finite/positive; gradient and origin need 3 entries")
        if not np.all(np.isfinite(self.gradient)) or not np.all(np.isfinite(self.origin)):
            raise ValueError("Gradient and origin coordinates must be finite")

    def numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        return self.v0 + np.sum((points - np.asarray(self.origin)) * self.gradient, axis=-1)

    def torch(self, points: Any) -> Any:
        g = points.new_tensor(self.gradient)
        origin = points.new_tensor(self.origin)
        return self.v0 + ((points - origin) * g).sum(dim=-1, keepdim=True)


@dataclass
class GaussianAnomaly:
    """A smooth additive velocity perturbation."""

    center: Sequence[float]
    sigma: Sequence[float]
    amplitude: float

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.sigma) != 3 or min(self.sigma) <= 0:
            raise ValueError("Gaussian centers/sigmas need 3 entries and positive sigmas")
        if not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.sigma)) or not np.isfinite(self.amplitude):
            raise ValueError("Gaussian parameters must be finite")


@dataclass
class SmoothHeterogeneous(VelocityModel):
    """A linear gradient with Gaussian anomalies; no closed-form traveltime."""

    background: ConstantGradient = field(default_factory=ConstantGradient)
    anomalies: List[GaussianAnomaly] = field(default_factory=lambda: [
        GaussianAnomaly((0.6, 0.7, 0.8), (0.22, 0.30, 0.25), -0.35),
        GaussianAnomaly((1.4, 1.2, 1.3), (0.30, 0.22, 0.28), 0.45),
        GaussianAnomaly((1.1, 0.4, 1.6), (0.18, 0.25, 0.20), -0.20),
    ])

    def numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        velocity = self.background.numpy(points)
        for anomaly in self.anomalies:
            q = (points - anomaly.center) / anomaly.sigma
            velocity = velocity + anomaly.amplitude * np.exp(-0.5 * np.sum(q * q, axis=-1))
        return velocity

    def torch(self, points: Any) -> Any:
        import torch

        velocity = self.background.torch(points)
        for anomaly in self.anomalies:
            q = (points - points.new_tensor(anomaly.center)) / points.new_tensor(anomaly.sigma)
            velocity = velocity + anomaly.amplitude * torch.exp(-0.5 * (q * q).sum(-1, keepdim=True))
        return velocity


def build_velocity(config: Dict[str, Any], bounds: Sequence[Sequence[float]]) -> VelocityModel:
    """Construct a model from the ``velocity`` YAML section."""
    kind = config.get("kind", "constant_gradient")
    if kind == "volume":
        from .volume import GriddedVelocity

        return GriddedVelocity.load(
            config["path"], bounds, shape=config.get("shape"),
            dtype=config.get("dtype", "float32"), order=config.get("order", "C"),
            axis_order=config.get("axis_order", "xyz"),
            velocity_scale=float(config.get("velocity_scale", 1.0)),
        )
    background = ConstantGradient(
        v0=float(config.get("v0", 2.0)),
        gradient=config.get("gradient", [0.0, 0.0, 0.5]),
        origin=config.get("origin", [0.0, 0.0, 0.0]),
    )
    if kind == "constant_gradient":
        model: VelocityModel = background
    elif kind == "heterogeneous":
        anomalies = config.get("anomalies")
        model = SmoothHeterogeneous(background) if anomalies is None else SmoothHeterogeneous(
            background, [GaussianAnomaly(**a) for a in anomalies]
        )
    else:
        raise ValueError("Unknown velocity kind: " + str(kind))
    # Corners exactly bound a linear field. Subtracting all negative amplitudes
    # gives a conservative lower bound for a Gaussian mixture over the domain.
    corners = np.array(np.meshgrid(*bounds, indexing="ij")).reshape(3, -1).T
    lower_bound = float(np.min(background.numpy(corners)))
    if isinstance(model, SmoothHeterogeneous):
        lower_bound += sum(min(0.0, a.amplitude) for a in model.anomalies)
    if lower_bound <= 0:
        raise ValueError("Velocity positivity cannot be guaranteed over the domain")
    return model
