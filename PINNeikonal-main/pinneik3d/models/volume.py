"""Explicit array ordering and trilinear interpolation for 3D velocity data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .velocity import VelocityModel


class GriddedVelocity(VelocityModel):
    """Store ``volume[ix, iy, iz]``; interpolate at coordinates in km.

    Torch's 5D grid_sample expects (N,C,D,H,W), so xyz volume axes map
    to D,H,W and query coordinates must be ordered (z,y,x). The mapping
    is tested with a non-symmetric linear field.
    """

    def __init__(self, volume: np.ndarray, bounds: Sequence[Sequence[float]]) -> None:
        volume = np.asarray(volume, dtype=np.float32)
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if volume.ndim != 3 or min(volume.shape) < 2:
            raise ValueError("Velocity volume must be 3D with at least two nodes per axis")
        if bounds_array.shape != (3, 2) or np.any(bounds_array[:, 1] <= bounds_array[:, 0]):
            raise ValueError("bounds must be three increasing [min,max] pairs")
        if not np.all(np.isfinite(volume)) or np.min(volume) <= 0:
            raise ValueError("All velocity samples must be finite and positive")
        self.volume = np.ascontiguousarray(volume)
        self.bounds = bounds_array
        self.axes = tuple(np.linspace(a, b, n) for (a, b), n in zip(bounds_array, volume.shape))
        self._numpy_interpolator = RegularGridInterpolator(self.axes, self.volume, bounds_error=True)
        self._tensor_cache: Dict[Tuple[str, str], Any] = {}

    @classmethod
    def load(cls, path: str, bounds: Sequence[Sequence[float]],
             shape: Optional[Sequence[int]] = None, dtype: str = "float32",
             order: str = "C", axis_order: str = "xyz", velocity_scale: float = 1.0) -> "GriddedVelocity":
        """Load .npy or raw .bin; raw shape/dtype/endianness/order are explicit.

        ``axis_order`` describes the file's array axes, e.g. zyx. Use
        velocity_scale=0.001 for volumes provided in m/s.
        """
        filename = Path(path)
        if sorted(axis_order) != ["x", "y", "z"] or order not in ("C", "F"):
            raise ValueError("axis_order must permute xyz and order must be C or F")
        if filename.suffix.lower() == ".npy":
            volume = np.load(str(filename), allow_pickle=False)
        elif filename.suffix.lower() == ".bin":
            if shape is None or len(shape) != 3:
                raise ValueError("A .bin volume requires its three file-axis dimensions")
            data = np.fromfile(str(filename), dtype=np.dtype(dtype))
            if data.size != int(np.prod(shape)):
                raise ValueError("Binary byte count does not match shape and dtype")
            volume = data.reshape(tuple(shape), order=order)
        else:
            raise ValueError("Supported volume formats: .npy and .bin")
        volume = np.transpose(volume, [axis_order.index(a) for a in "xyz"])
        return cls(volume * velocity_scale, bounds)

    def numpy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        return self._numpy_interpolator(points.reshape(-1, 3)).reshape(points.shape[:-1])

    def torch(self, points: Any) -> Any:
        import torch
        import torch.nn.functional as F

        # Velocities stay in float32 during AMP; interpolation is fixed physics.
        key = (str(points.device), str(points.dtype))
        if key not in self._tensor_cache:
            self._tensor_cache[key] = torch.as_tensor(
                self.volume, device=points.device, dtype=points.dtype
            )[None, None]
        bounds = points.new_tensor(self.bounds)
        normalized = 2.0 * (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0
        if bool(torch.any((normalized.detach() < -1.00001) | (normalized.detach() > 1.00001))):
            raise ValueError("Velocity query lies outside the volume bounds")
        grid = normalized[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3)
        values = F.grid_sample(self._tensor_cache[key], grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return values.reshape(*points.shape[:-1], 1)
