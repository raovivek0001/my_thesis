"""Volume ordering, units, interpolation, and analytic velocity checks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pinneik3d.common import PROJECT_ROOT
from pinneik3d.models import ConstantGradient, GriddedVelocity, SmoothHeterogeneous, build_velocity


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.bounds = [[0, 2], [0, 3], [0, 4]]
        axes = [np.linspace(a, b, n) for (a, b), n in zip(self.bounds, (5, 7, 9))]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        self.volume = 2 + 0.1*x + 0.2*y + 0.3*z
        self.points = np.random.RandomState(3).uniform([0, 0, 0], [2, 3, 4], (100, 3))

    def test_numpy_trilinear_interpolation_and_bounds(self):
        model = GriddedVelocity(self.volume, self.bounds)
        np.testing.assert_allclose(model.numpy(self.points), 2 + self.points @ [0.1, 0.2, 0.3], rtol=1e-7)
        with self.assertRaises(ValueError):
            model.numpy(np.array([[3., 0., 0.]]))

    def test_binary_zyx_loader_and_unit_conversion(self):
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as directory:
            filename = Path(directory) / "v.bin"
            file_volume = np.transpose(self.volume, (2, 1, 0)) * 1000
            file_volume.astype("<f4").tofile(str(filename))
            model = GriddedVelocity.load(str(filename), self.bounds, file_volume.shape,
                                         dtype="<f4", axis_order="zyx", velocity_scale=0.001)
            np.testing.assert_allclose(model.volume, self.volume, rtol=2e-7)

    def test_velocity_positivity(self):
        self.assertGreater(np.min(SmoothHeterogeneous().numpy(self.points)), 0)
        with self.assertRaises(ValueError):
            build_velocity({"v0": 0.1, "gradient": [0, 0, -1]}, self.bounds)

    def test_error_slice_passes_through_off_grid_source(self):
        from pinneik3d.benchmarks.plotting import source_slice

        source = [.87, 1.13, 1.73]
        sliced = source_slice(self.volume, self.bounds, source, normal_axis=2)
        x, y = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 3, 7), indexing="ij")
        np.testing.assert_allclose(sliced, 2+.1*x+.2*y+.3*source[2], atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_torch_grid_sample_order_and_gradient(self):
        import torch

        points = torch.tensor(self.points, dtype=torch.float32, requires_grad=True)
        model = GriddedVelocity(self.volume, self.bounds)
        velocity = model.torch(points)
        np.testing.assert_allclose(velocity.detach().numpy().ravel(), model.numpy(self.points), rtol=2e-7)
        gradient = torch.autograd.grad(velocity.sum(), points)[0]
        np.testing.assert_allclose(gradient.numpy(), np.tile([.1, .2, .3], (100, 1)), atol=1e-6)
