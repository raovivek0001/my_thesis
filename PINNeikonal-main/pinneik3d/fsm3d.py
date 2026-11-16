"""First-order multiplicatively factored 3D fast sweeping, compiled by Numba.

The Godunov derivative along axis i is
  max((T0/h_i + p_i)*tau - T0*tau_minus/h_i,
      (T0/h_i - p_i)*tau - T0*tau_plus/h_i, 0),
where p_i = d_i T0. Its squared norm equals the squared slowness.
This discretizes tau, rather than running an unfactored solver and dividing
its output by T0. Eight in-place Gauss-Seidel passes form one sweep cycle.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .grids import grid_axes

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Keep a diagnostic Python kernel importable without Numba."""
        def decorate(function):
            return function
        return decorate

_INF = 1e30
_WARMED = False


@njit(cache=True)
def _root_three(c0, b0, c1, b1, c2, b2, slowness):
    """Exact positive root of sum(max(c_i*tau-b_i,0)^2)=s²."""
    r0 = b0 / c0 if c0 > 0.0 else _INF
    r1 = b1 / c1 if c1 > 0.0 else _INF
    r2 = b2 / c2 if c2 > 0.0 else _INF
    if r0 > r1:
        c0, c1, b0, b1, r0, r1 = c1, c0, b1, b0, r1, r0
    if r1 > r2:
        c1, c2, b1, b2, r1, r2 = c2, c1, b2, b1, r2, r1
    if r0 > r1:
        c0, c1, b0, b1, r0, r1 = c1, c0, b1, b0, r1, r0
    if c0 <= 0.0:
        return _INF
    root = (b0 + slowness) / c0
    if c1 <= 0.0 or root <= r1:
        return root
    aa = c0*c0 + c1*c1
    bb = c0*b0 + c1*b1
    cc = b0*b0 + b1*b1 - slowness*slowness
    root = (bb + math.sqrt(max(0.0, bb*bb - aa*cc))) / aa
    if c2 <= 0.0 or root <= r2:
        return root
    aa += c2*c2
    bb += c2*b2
    cc += b2*b2
    return (bb + math.sqrt(max(0.0, bb*bb - aa*cc))) / aa


@njit(cache=True)
def _node_update(tau, times, i, j, k, t0, px, py, pz, hx, hy, hz, slowness):
    """Choose Godunov branches and solve their piecewise quadratic exactly."""
    nx, ny, nz = tau.shape
    c0m, c0p, b0m, b0p = 0.0, 0.0, 0.0, 0.0
    c1m, c1p, b1m, b1p = 0.0, 0.0, 0.0, 0.0
    c2m, c2p, b2m, b2p = 0.0, 0.0, 0.0, 0.0
    if i > 0 and times[i-1, j, k] < _INF/2:
        c0m, b0m = max(0.0, t0/hx + px), t0/hx * tau[i-1, j, k]
    if i+1 < nx and times[i+1, j, k] < _INF/2:
        c0p, b0p = max(0.0, t0/hx - px), t0/hx * tau[i+1, j, k]
    if j > 0 and times[i, j-1, k] < _INF/2:
        c1m, b1m = max(0.0, t0/hy + py), t0/hy * tau[i, j-1, k]
    if j+1 < ny and times[i, j+1, k] < _INF/2:
        c1p, b1p = max(0.0, t0/hy - py), t0/hy * tau[i, j+1, k]
    if k > 0 and times[i, j, k-1] < _INF/2:
        c2m, b2m = max(0.0, t0/hz + pz), t0/hz * tau[i, j, k-1]
    if k+1 < nz and times[i, j, k+1] < _INF/2:
        c2p, b2p = max(0.0, t0/hz - pz), t0/hz * tau[i, j, k+1]
    # A one-direction solution bounds the full root from above.
    upper = _INF
    if c0m > 0.0:
        upper = min(upper, (b0m+slowness)/c0m)
    if c0p > 0.0:
        upper = min(upper, (b0p+slowness)/c0p)
    if c1m > 0.0:
        upper = min(upper, (b1m+slowness)/c1m)
    if c1p > 0.0:
        upper = min(upper, (b1p+slowness)/c1p)
    if c2m > 0.0:
        upper = min(upper, (b2m+slowness)/c2m)
    if c2p > 0.0:
        upper = min(upper, (b2p+slowness)/c2p)
    if upper >= _INF/2:
        return _INF
    # Each pair of lines crosses at most once. Starting from the upper bound
    # and resolving the winning branches decreases the bound monotonically.
    for _ in range(12):
        c0, b0 = (c0m, b0m) if c0m*upper-b0m >= c0p*upper-b0p else (c0p, b0p)
        c1, b1 = (c1m, b1m) if c1m*upper-b1m >= c1p*upper-b1p else (c1p, b1p)
        c2, b2 = (c2m, b2m) if c2m*upper-b2m >= c2p*upper-b2p else (c2p, b2p)
        updated = _root_three(c0, b0, c1, b1, c2, b2, slowness)
        if abs(updated-upper) <= 1e-13 * max(1.0, abs(upper)):
            return updated
        upper = updated
    return upper


@njit(cache=True)
def _sweep_kernel(velocity, tau, times, fixed, xs, ys, zs, x0, y0, z0,
                  hx, hy, hz, source_velocity, tolerance, max_cycles):
    nx, ny, nz = velocity.shape
    delta = _INF
    for cycle in range(max_cycles):
        delta = 0.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    irange = range(nx) if sx == 1 else range(nx-1, -1, -1)
                    jrange = range(ny) if sy == 1 else range(ny-1, -1, -1)
                    krange = range(nz) if sz == 1 else range(nz-1, -1, -1)
                    for i in irange:
                        dx = x0+i*hx-xs
                        for j in jrange:
                            dy = y0+j*hy-ys
                            for k in krange:
                                if fixed[i, j, k]:
                                    continue
                                dz = z0+k*hz-zs
                                radius = math.sqrt(dx*dx + dy*dy + dz*dz)
                                t0 = radius/source_velocity
                                denominator = max(radius, 1e-15)*source_velocity
                                candidate = _node_update(tau, times, i, j, k, t0,
                                                         dx/denominator, dy/denominator, dz/denominator,
                                                         hx, hy, hz, 1.0/velocity[i, j, k])
                                new_time = t0*candidate
                                if new_time < times[i, j, k]:
                                    delta = max(delta, times[i, j, k]-new_time)
                                    times[i, j, k] = new_time
                                    tau[i, j, k] = candidate
        if delta <= tolerance:
            return cycle+1, delta
    return max_cycles, delta


@dataclass
class FSMResult:
    """Solver output; seconds excludes JIT compilation and velocity sampling."""

    traveltime: np.ndarray
    tau: np.ndarray
    cycles: int
    final_change: float
    converged: bool
    seconds: float
    backend: str
    source_injection: str


def warmup_fsm() -> None:
    """Compile once, outside benchmark timing, using a tiny constant medium."""
    global _WARMED
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for fair FSM timing; install requirements.txt")
    if not _WARMED:
        velocity = np.full((3, 3, 3), 2.0)
        tau, times = np.full_like(velocity, _INF), np.full_like(velocity, _INF)
        fixed = np.zeros(velocity.shape, dtype=np.bool_)
        fixed[1, 1, 1], tau[1, 1, 1], times[1, 1, 1] = True, 1.0, 0.0
        _sweep_kernel(velocity, tau, times, fixed, 1., 1., 1., 0., 0., 0.,
                      1., 1., 1., 2., 1e-9, 5)
        _WARMED = True


def solve_fsm(velocity: np.ndarray, bounds: Sequence[Sequence[float]], source: Sequence[float],
              source_velocity: Optional[float] = None, tolerance: float = 1e-8,
              max_cycles: int = 100, backend: str = "numba",
              require_convergence: bool = True) -> FSMResult:
    """Solve the factored eikonal equation on a uniform, possibly anisotropic grid.

    Grid-aligned sources pin only the source node with tau=1. Off-grid sources
    keep their true position and pin a shrinking ball of radius one grid-cell
    diagonal using a local linear-velocity approximation. This ensures
    nonnegative stencil coefficients outside the injection region and retains
    first-order consistency. No source snapping or analytic global seeding is
    used. The Python backend is exclusively for numerical diagnostics.
    """
    if backend not in ("numba", "python"):
        raise ValueError("backend must be numba or python")
    if backend == "numba":
        warmup_fsm()
    elif NUMBA_AVAILABLE:
        # A dispatcher has a Python body, but it calls compiled helpers. Use
        # this backend only when Numba is absent; it must never enter timings.
        raise ValueError("Use backend=numba when Numba is installed")
    velocity = np.ascontiguousarray(velocity, dtype=np.float64)
    domain = np.asarray(bounds, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
    if velocity.ndim != 3 or min(velocity.shape) < 3:
        raise ValueError("FSM requires at least three nodes on each axis")
    if not np.all(np.isfinite(velocity)) or np.min(velocity) <= 0:
        raise ValueError("FSM velocities must be finite and positive")
    if domain.shape != (3, 2) or np.any(domain[:, 1] <= domain[:, 0]):
        raise ValueError("Invalid domain bounds")
    if source_array.shape != (3,) or np.any(source_array < domain[:, 0]) or np.any(source_array > domain[:, 1]):
        raise ValueError("Source must lie inside the domain")
    if tolerance <= 0 or max_cycles <= 0:
        raise ValueError("Convergence tolerance and max_cycles must be positive")
    if source_velocity is None:
        from .grids import interpolate_field
        source_velocity = float(interpolate_field(velocity, bounds, source_array))
    if source_velocity <= 0 or not np.isfinite(source_velocity):
        raise ValueError("Source velocity must be finite and positive")
    started = time.perf_counter()
    steps = (domain[:, 1]-domain[:, 0]) / (np.asarray(velocity.shape)-1)
    grid_index = (source_array-domain[:, 0])/steps
    on_grid = bool(np.all(np.abs(grid_index-np.round(grid_index)) < 1e-9))
    tau, times = np.full_like(velocity, _INF), np.full_like(velocity, _INF)
    fixed = np.zeros(velocity.shape, dtype=np.bool_)
    if on_grid:
        index = tuple(np.round(grid_index).astype(int))
        fixed[index], tau[index], times[index] = True, 1.0, 0.0
        injection = "single_grid_node"
    else:
        axes = grid_axes(bounds, velocity.shape)
        dx = axes[0][:, None, None]-source_array[0]
        dy = axes[1][None, :, None]-source_array[1]
        dz = axes[2][None, None, :]-source_array[2]
        radius_squared = dx*dx + dy*dy + dz*dz
        fixed[:] = radius_squared <= np.sum(steps*steps)*(1.0+1e-12)
        radius = np.sqrt(radius_squared[fixed])
        tau[fixed] = 2.0*source_velocity/(source_velocity+velocity[fixed])
        times[fixed] = radius/source_velocity*tau[fixed]
        injection = "off_grid_local_ball"
    cycles, delta = _sweep_kernel(velocity, tau, times, fixed, *source_array,
                                  *domain[:, 0], *steps, source_velocity, tolerance, max_cycles)
    elapsed = time.perf_counter()-started
    converged = bool(delta <= tolerance and np.all(times < _INF/2))
    if require_convergence and not converged:
        raise RuntimeError("FSM did not converge: cycles=%d, final change=%.3e" % (cycles, delta))
    return FSMResult(times, tau, int(cycles), float(delta), converged, elapsed, backend, injection)


def solve_fmm(velocity: np.ndarray, bounds: Sequence[Sequence[float]], source: Sequence[float],
              order: int = 2) -> Tuple[np.ndarray, float]:
    """Optional scikit-fmm comparison for a point source exactly on a grid node.

    The level-set field is zero only at the source node and positive elsewhere.
    Off-grid sources are rejected rather than silently snapped to a node.
    Runtime includes point-source field initialization but excludes velocity
    sampling, which the benchmark adds to its end-to-end cost.
    """
    import skfmm

    volume = np.asarray(velocity, dtype=np.float64)
    domain = np.asarray(bounds, dtype=float)
    point = np.asarray(source, dtype=float)
    if volume.ndim != 3 or min(volume.shape) < 3 or not np.all(np.isfinite(volume)) or np.min(volume) <= 0:
        raise ValueError("FMM requires a finite positive 3D velocity volume")
    if domain.shape != (3, 2) or np.any(domain[:, 1] <= domain[:, 0]):
        raise ValueError("Invalid FMM bounds")
    if point.shape != (3,) or np.any(point < domain[:, 0]) or np.any(point > domain[:, 1]):
        raise ValueError("FMM source must lie inside the domain")
    steps = (domain[:, 1]-domain[:, 0])/(np.asarray(volume.shape)-1)
    fractional = (point-domain[:, 0])/steps
    if not np.all(np.abs(fractional-np.round(fractional)) < 1e-9):
        raise ValueError("Optional FMM supports only grid-aligned point sources")
    if order not in (1, 2):
        raise ValueError("scikit-fmm order must be 1 or 2")
    started = time.perf_counter()
    axes = grid_axes(bounds, volume.shape)
    phi = np.sqrt((axes[0][:, None, None]-point[0])**2
                  + (axes[1][None, :, None]-point[1])**2
                  + (axes[2][None, None, :]-point[2])**2)
    phi[tuple(np.round(fractional).astype(int))] = 0.0
    times = skfmm.travel_time(phi, volume, dx=tuple(steps), order=order)
    if np.ma.isMaskedArray(times) and np.any(np.ma.getmaskarray(times)):
        raise RuntimeError("FMM returned unreachable/masked nodes")
    return np.asarray(times), time.perf_counter()-started
