"""Memory-bounded Cartesian grids with an explicit xyz array convention."""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

GridSize = Union[int, Sequence[int]]


def grid_shape(size: GridSize) -> Tuple[int, int, int]:
    """Convert a cube size or a three-axis shape into a validated tuple."""
    shape = (size, size, size) if isinstance(size, int) else tuple(size)
    if len(shape) != 3 or any(n < 2 or not isinstance(n, (int, np.integer)) for n in shape):
        raise ValueError("A grid requires three integer dimensions >= 2")
    return tuple(int(n) for n in shape)


def grid_axes(bounds: Sequence[Sequence[float]], size: GridSize) -> Tuple[np.ndarray, ...]:
    """Return inclusive, increasing x, y, z node coordinates."""
    return tuple(np.linspace(a, b, n, dtype=np.float64) for (a, b), n in zip(bounds, grid_shape(size)))


def grid_batches(bounds: Sequence[Sequence[float]], size: GridSize, batch_size: int,
                 dtype: np.dtype = np.float32) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Generate C-order xyz chunks without allocating an (N**3,3) grid."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    shape = grid_shape(size)
    axes = grid_axes(bounds, size)
    total = int(np.prod(shape))
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = np.arange(start, stop, dtype=np.int64)
        indices = (flat // (shape[1] * shape[2]), (flat // shape[2]) % shape[1], flat % shape[2])
        points = np.column_stack([axis[index] for axis, index in zip(axes, indices)]).astype(dtype)
        yield start, stop, points


def sample_velocity_grid(model: object, bounds: Sequence[Sequence[float]], size: GridSize,
                         batch_size: int = 262144) -> np.ndarray:
    """Evaluate a velocity field in chunks; the returned array is float64 xyz."""
    result = np.empty(int(np.prod(grid_shape(size))), dtype=np.float64)
    for start, stop, points in grid_batches(bounds, size, batch_size, np.float64):
        result[start:stop] = model.numpy(points)
    return result.reshape(grid_shape(size))


def interpolate_field(field: np.ndarray, bounds: Sequence[Sequence[float]], points: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of an xyz field for reference evaluation."""
    interpolator = RegularGridInterpolator(grid_axes(bounds, field.shape), field, bounds_error=True)
    return interpolator(np.asarray(points).reshape(-1, 3)).reshape(np.asarray(points).shape[:-1])
