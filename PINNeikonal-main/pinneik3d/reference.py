"""Analytic or twice-finer FSM references with an explicit error floor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .fsm3d import solve_fsm
from .grids import GridSize, grid_axes, grid_batches, grid_shape, sample_velocity_grid
from .models import ConstantGradient, VelocityModel, traveltime_numpy


@dataclass
class Reference:
    """Queryable reference; reference generation is timed separately."""

    velocity: VelocityModel
    bounds: Sequence[Sequence[float]]
    source: np.ndarray
    kind: str
    generation_seconds: float
    metadata: Dict[str, Any]
    interpolator: Any = None

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate analytic times or interpolate the finer FSM field."""
        if self.kind == "analytic":
            return traveltime_numpy(points, self.source, self.velocity)
        return self.interpolator(np.asarray(points).reshape(-1, 3)).reshape(np.asarray(points).shape[:-1])

    def grid(self, size: GridSize, batch_size: int = 262144) -> np.ndarray:
        """Materialize a reference grid without a full coordinate tensor."""
        output = np.empty(int(np.prod(grid_shape(size))), dtype=np.float64)
        for start, stop, points in grid_batches(self.bounds, size, batch_size, np.float64):
            output[start:stop] = self.evaluate(points)
        return output.reshape(grid_shape(size))


def make_reference(velocity: VelocityModel, bounds: Sequence[Sequence[float]], source: Sequence[float],
                   coarse_size: GridSize, config: Dict[str, Any]) -> Reference:
    """Build an analytic reference when available, otherwise FSM at h/2.

    Twice finer means 2*(n-1)+1 nodes per axis: 51 -> 101, 101 -> 201,
    and 201 -> 401. It does not mean 2*n nodes. The finer solver's convergence
    tolerance controls iteration error, not its first-order discretization error.
    """
    started = time.perf_counter()
    source_array = np.asarray(source, dtype=np.float64)
    kind = config["evaluation"]["reference"]
    if kind == "analytic" and not isinstance(velocity, ConstantGradient):
        raise ValueError("An analytic reference exists only for constant-gradient velocity")
    if kind == "analytic" or (kind == "auto" and isinstance(velocity, ConstantGradient)):
        return Reference(velocity, bounds, source_array, "analytic", time.perf_counter()-started,
                         {"reference_kind": "analytic", "reference_grid": None})
    shape = grid_shape(coarse_size)
    fine_shape: Tuple[int, ...] = tuple(2*(n-1)+1 for n in shape)
    volume = sample_velocity_grid(velocity, bounds, fine_shape)
    settings = config["fsm"]
    result = solve_fsm(volume, bounds, source, float(velocity.numpy(source_array)),
                       tolerance=settings["tolerance"], max_cycles=settings["max_cycles"],
                       backend=settings["backend"])
    interpolation = RegularGridInterpolator(grid_axes(bounds, fine_shape), result.traveltime,
                                             bounds_error=True)
    return Reference(velocity, bounds, source_array, "fsm_fine", time.perf_counter()-started,
                     {"reference_kind": "fsm_fine", "reference_grid": list(fine_shape),
                      "reference_cycles": result.cycles, "reference_final_change": result.final_change,
                      "reference_backend": result.backend, "reference_injection": result.source_injection,
                      "reference_discretization": "first_order_h_over_2"}, interpolation)
