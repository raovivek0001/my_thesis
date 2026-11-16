"""First-coordinate-derivative factored residual and gradient balancing."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .models import VelocityModel


def background_factor(points: Tensor, sources: Tensor, source_velocity: Tensor,
                      eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Return exact-distance T0 and its guarded analytic gradient.

    T0(xs)=0 is retained exactly. Only the gradient denominator is clamped;
    adding epsilon to the distance itself would violate the source condition.
    """
    displacement = points - sources
    radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    t0 = radius / source_velocity
    gradient = displacement / (radius.clamp_min(eps) * source_velocity)
    return t0, gradient


def residual_from_tau(tau: Tensor, points: Tensor, sources: Tensor,
                      velocity: VelocityModel, eps: float = 1e-8,
                      create_graph: bool = True) -> Tensor:
    """Evaluate the expanded multiplicative residual using only grad(tau).

    Fixed velocity and T0 coefficients are detached. This avoids unnecessary
    velocity derivatives and unsupported double backward through grid_sample.
    Parameter gradients still flow through grad(tau) during training.
    """
    gradient_tau = torch.autograd.grad(tau, points, grad_outputs=torch.ones_like(tau),
                                       create_graph=create_graph, retain_graph=True)[0]
    with torch.no_grad():
        v = velocity.torch(points).float()
        vs = velocity.torch(sources).float()
        t0, gradient_t0 = background_factor(points.float(), sources.float(), vs, eps)
    # Residual arithmetic stays in float32 during AMP. Preserve float64 in
    # numerical verification so tests are not limited by single precision.
    dtype = torch.float64 if tau.dtype == torch.float64 else torch.float32
    if dtype == torch.float64:
        with torch.no_grad():
            v, vs = velocity.torch(points), velocity.torch(sources)
            t0, gradient_t0 = background_factor(points, sources, vs, eps)
    tau, gradient_tau = tau.to(dtype), gradient_tau.to(dtype)
    return (t0.square() * gradient_tau.square().sum(-1, keepdim=True)
            + tau.square() * gradient_t0.square().sum(-1, keepdim=True)
            + 2.0 * t0 * tau * (gradient_t0 * gradient_tau).sum(-1, keepdim=True)
            - v.reciprocal().square())


def loss_terms(model: nn.Module, points: Tensor, sources: Tensor,
               boundary_sources: Tensor, velocity: VelocityModel,
               eps: float = 1e-8, create_graph: bool = True,
               residual_normalization: str = "none") -> Dict[str, Tensor]:
    """Compute mean(R²), (mean(ReLU(-tau)))², and mean((tau(xs)-1)²).

    The positivity term squares the mean, as requested in the formulation;
    it is deliberately different from mean(ReLU(-tau)²).
    Optional relative normalization replaces R by v² R, retaining PDE roots
    while changing their weighting to dimensionless relative residuals.
    """
    tau = model(points, sources)
    residual = residual_from_tau(tau, points, sources, velocity, eps, create_graph)
    if residual_normalization == "relative":
        # v^2 R = v^2 |grad T|^2 - 1: dimensionless, with the same PDE roots.
        # Detachment is essential for gridded velocity double backward.
        with torch.no_grad():
            scale = velocity.torch(points).to(residual.dtype).square()
        residual = residual * scale
    elif residual_normalization != "none":
        raise ValueError("Unknown residual normalization: " + residual_normalization)
    source_tau = model(boundary_sources, boundary_sources)
    return {"pde": residual.square().mean(),
            "positivity": torch.relu(-tau.float()).mean().square(),
            "source": (source_tau.float() - 1.0).square().mean()}


class GradientLossBalancer:
    """Wang-style gradient-statistic weights with exponential smoothing.

    PDE weight is fixed. For each auxiliary loss i, estimate
    lambda_i = max(|grad L_pde|) / mean(|grad L_i|), then apply an EMA.
    Statistics are taken over trainable parameters, including zero entries
    for parameters unused by a term. A zero-gradient term retains its weight.
    """

    def __init__(self, weights: Optional[Mapping[str, float]] = None, enabled: bool = True,
                 beta: float = 0.9, min_weight: float = 1e-3, max_weight: float = 1e3,
                 update_every: int = 50) -> None:
        self.weights = dict(weights or {"pde": 1.0, "positivity": 1.0, "source": 1.0})
        self.enabled = enabled
        self.beta = beta
        self.min_weight, self.max_weight = min_weight, max_weight
        self.update_every = update_every

    def update(self, terms: Mapping[str, Tensor], parameters: Iterable[nn.Parameter]) -> None:
        """Inspect back-propagated gradients without modifying parameter .grad."""
        if not self.enabled:
            return
        params = tuple(p for p in parameters if p.requires_grad)
        reference = torch.autograd.grad(terms["pde"], params, retain_graph=True, allow_unused=True)
        maxima = [g.detach().abs().max() for g in reference if g is not None]
        if not maxima:
            return
        numerator = torch.stack(maxima).max().item() * self.weights["pde"]
        if numerator <= 1e-15:
            return
        count = sum(p.numel() for p in params)
        for name in ("positivity", "source"):
            if self.weights.get(name, 0.0) == 0.0:
                continue
            gradients = torch.autograd.grad(terms[name], params, retain_graph=True, allow_unused=True)
            sums = [g.detach().abs().sum() for g in gradients if g is not None]
            denominator = torch.stack(sums).sum().item() / max(count, 1) if sums else 0.0
            if denominator > 1e-15:
                estimate = max(self.min_weight, min(self.max_weight, numerator / denominator))
                self.weights[name] = self.beta * self.weights[name] + (1.0 - self.beta) * estimate

    def total(self, terms: Mapping[str, Tensor]) -> Tensor:
        """Weighted objective with constant weights for the current step."""
        return sum(self.weights[name] * term for name, term in terms.items())

    def state_dict(self) -> Dict[str, float]:
        return dict(self.weights)

    def load_state_dict(self, state: Mapping[str, float]) -> None:
        self.weights = dict(state)
