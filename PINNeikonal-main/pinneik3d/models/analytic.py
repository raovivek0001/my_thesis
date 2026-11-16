"""Stable analytic first-arrival times in a constant velocity gradient."""

from __future__ import annotations

from typing import Any

import numpy as np

from .velocity import ConstantGradient


def traveltime_numpy(points: np.ndarray, source: np.ndarray, model: ConstantGradient) -> np.ndarray:
    """Evaluate ``acosh(1+g²r²/(2 v vs))/|g|`` without cancellation.

    The equivalent expression ``2 asinh(|g| r/(2 sqrt(v vs)))/|g|``
    is accurate near the source. Both coordinate and source arrays broadcast.
    """
    points, source = np.asarray(points), np.asarray(source)
    radius = np.linalg.norm(points - source, axis=-1)
    v, vs = model.numpy(points), model.numpy(source)
    if np.any(v <= 0) or np.any(vs <= 0):
        raise ValueError("The analytic solution requires strictly positive velocities")
    magnitude = float(np.linalg.norm(model.gradient))
    if magnitude == 0.0:
        return radius / vs
    return 2.0 * np.arcsinh(magnitude * radius / (2.0 * np.sqrt(v * vs))) / magnitude


def traveltime_torch(points: Any, source: Any, model: ConstantGradient) -> Any:
    """Autograd-compatible analytic solution, including the zero-gradient limit."""
    import torch

    radius = torch.linalg.vector_norm(points - source, dim=-1, keepdim=True)
    v, vs = model.torch(points), model.torch(source)
    magnitude = float(np.linalg.norm(model.gradient))
    if magnitude == 0.0:
        return radius / vs
    return 2.0 * torch.asinh(magnitude * radius / (2.0 * torch.sqrt(v * vs))) / magnitude


def tau_numpy(points: np.ndarray, source: np.ndarray, model: ConstantGradient, eps: float = 1e-12) -> np.ndarray:
    """Analytic multiplicative factor with its continuous source value set to 1."""
    radius = np.linalg.norm(np.asarray(points) - source, axis=-1)
    t0 = radius / model.numpy(np.asarray(source))
    time = traveltime_numpy(points, source, model)
    return np.divide(time, t0, out=np.ones_like(time), where=radius > eps)


def tau_torch(points: Any, source: Any, model: ConstantGradient, eps: float = 1e-8) -> Any:
    """Differentiable analytic factor away from the source."""
    import torch

    radius = torch.linalg.vector_norm(points - source, dim=-1, keepdim=True)
    t0 = radius / model.torch(source)
    safe_t0 = torch.where(radius > eps, t0, torch.ones_like(t0))
    ratio = traveltime_torch(points, source, model) / safe_t0
    return torch.where(radius > eps, ratio, torch.ones_like(ratio))
