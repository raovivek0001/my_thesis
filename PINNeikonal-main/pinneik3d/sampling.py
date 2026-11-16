"""Uniform exploration plus periodically refreshed high-residual collocations."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def ranked_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Select largest finite residual magnitudes, preserving deterministic ties."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < count <= len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Residual scores must be finite and selection count valid")
    return np.argsort(-np.abs(values), kind="stable")[:count].copy()


def sample_uniform(bounds: Any, lattice: Any, batch_size: int, source_batch_size: int,
                   generator: Any, device: Any) -> Tuple[Any, Any, Any]:
    """Sample physical coordinates and source indices using the saved CPU RNG."""
    import torch

    points = bounds[:, 0] + torch.rand(batch_size, 3, generator=generator)*(bounds[:, 1]-bounds[:, 0])
    indices = torch.randint(len(lattice), (batch_size,), generator=generator)
    sources = lattice[indices]
    boundary_indices = torch.randint(len(lattice), (min(source_batch_size, len(lattice)),), generator=generator)
    return (points.to(device).requires_grad_(True), sources.to(device), lattice[boundary_indices].to(device))


class ResidualSampler:
    """Keep uniform coverage while reusing a refreshed pool of hard PDE points.

    Candidate scoring is chunked to the optimization batch size, uses only
    coordinate gradients, and never reads analytic/FSM reference values.
    The pool and scoring count are checkpointed alongside the collocation RNG.
    """

    def __init__(self, bounds: Any, lattice: Any, batch_size: int,
                 source_batch_size: int, generator: Any, device: Any,
                 settings: Dict[str, Any], eps: float) -> None:
        self.bounds, self.lattice = bounds, lattice
        self.batch_size, self.source_batch_size = batch_size, source_batch_size
        self.generator, self.device = generator, device
        self.settings, self.eps = settings, eps
        self.pool_points, self.pool_sources = None, None
        self.candidate_points_scored = 0

    def refresh(self, model: Any, velocity: Any) -> None:
        """Rank a fresh uniform candidate pool by abs(v^2 R)."""
        import torch
        from .losses import residual_from_tau

        candidate_count = self.batch_size*self.settings["candidate_multiplier"]
        point_chunks, source_chunks, score_chunks = [], [], []
        was_training = model.training
        model.eval()
        try:
            for start in range(0, candidate_count, self.batch_size):
                count = min(self.batch_size, candidate_count-start)
                points, sources, _ = sample_uniform(self.bounds, self.lattice, count,
                    self.source_batch_size, self.generator, self.device)
                with torch.enable_grad():
                    tau = model(points, sources)
                    residual = residual_from_tau(tau, points, sources, velocity,
                                                 self.eps, create_graph=False)
                with torch.no_grad():
                    scores = (residual*velocity.torch(points).square()).abs().reshape(-1)
                point_chunks.append(points.detach().cpu())
                source_chunks.append(sources.detach().cpu())
                score_chunks.append(scores.detach().cpu().numpy())
                del points, sources, tau, residual, scores
        finally:
            model.train(was_training)
        selected = ranked_indices(np.concatenate(score_chunks), self.batch_size)
        self.pool_points = torch.cat(point_chunks)[selected].clone()
        self.pool_sources = torch.cat(source_chunks)[selected].clone()
        self.candidate_points_scored += candidate_count

    def sample(self, model: Any, velocity: Any, iteration: int) -> Tuple[Any, Any, Any]:
        """Draw a minibatch; inactive sampling has the original uniform RNG path."""
        import torch

        enabled = (self.settings["enabled"] and self.settings["uniform_fraction"] < 1
                   and iteration > self.settings["warmup"])
        if enabled and (self.pool_points is None or iteration % self.settings["update_every"] == 0):
            self.refresh(model, velocity)
        points, sources, boundaries = sample_uniform(self.bounds, self.lattice, self.batch_size,
            self.source_batch_size, self.generator, self.device)
        if enabled:
            hard_count = int(self.batch_size*(1-self.settings["uniform_fraction"]))
            indices = torch.randint(len(self.pool_points), (hard_count,), generator=self.generator)
            with torch.no_grad():
                points[:hard_count] = self.pool_points[indices].to(self.device)
                sources[:hard_count] = self.pool_sources[indices].to(self.device)
        return points, sources, boundaries

    def state_dict(self) -> Dict[str, Any]:
        """Return only CPU tensors so checkpoints can move between devices."""
        return {"pool_points": self.pool_points, "pool_sources": self.pool_sources,
                "candidate_points_scored": self.candidate_points_scored}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the hard pool without drawing new random candidates."""
        self.pool_points, self.pool_sources = state["pool_points"], state["pool_sources"]
        self.candidate_points_scored = int(state["candidate_points_scored"])
