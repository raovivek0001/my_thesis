"""Independent finite-difference checks of the mathematical formulation."""

import importlib.util
import unittest

import numpy as np

from pinneik3d.models import ConstantGradient, tau_numpy, traveltime_numpy


def finite_gradient(function, points, step=1e-5):
    derivatives = []
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        derivatives.append((function(points + delta) - function(points - delta)) / (2 * step))
    return np.stack(derivatives, axis=-1)


class AnalyticTests(unittest.TestCase):
    def setUp(self):
        self.model = ConstantGradient(2.0, (0.15, -0.1, 0.5))
        self.source = np.array([0.8, 0.9, 0.7])
        self.points = np.random.RandomState(42).uniform(0.05, 1.95, (300, 3))

    def test_analytic_satisfies_eikonal(self):
        gradient = finite_gradient(lambda p: traveltime_numpy(p, self.source, self.model), self.points)
        relative_residual = np.sum(gradient ** 2, axis=-1) * self.model.numpy(self.points) ** 2 - 1
        self.assertLess(np.max(np.abs(relative_residual)), 2e-7)

    def test_factored_residual_analytic_tau_finite_difference(self):
        tau = tau_numpy(self.points, self.source, self.model)
        gradient_tau = finite_gradient(lambda p: tau_numpy(p, self.source, self.model), self.points)
        displacement = self.points - self.source
        radius = np.linalg.norm(displacement, axis=-1)
        vs = self.model.numpy(self.source)
        t0 = radius / vs
        gradient_t0 = displacement / (radius[:, None] * vs)
        residual = (t0**2 * np.sum(gradient_tau**2, axis=-1)
                    + tau**2 * np.sum(gradient_t0**2, axis=-1)
                    + 2*t0*tau*np.sum(gradient_t0*gradient_tau, axis=-1)
                    - 1/self.model.numpy(self.points)**2)
        self.assertLess(np.max(np.abs(residual)), 2e-8)

    def test_source_and_zero_gradient_limit(self):
        self.assertEqual(float(traveltime_numpy(self.source, self.source, self.model)), 0.0)
        self.assertEqual(float(tau_numpy(self.source, self.source, self.model)), 1.0)
        constant = ConstantGradient(2.3, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(traveltime_numpy(self.points, self.source, constant),
                                   np.linalg.norm(self.points-self.source, axis=-1)/2.3)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_analytic_torch_autograd(self):
        import torch
        from pinneik3d.models import traveltime_torch

        points = torch.tensor(self.points, dtype=torch.float64, requires_grad=True)
        source = torch.tensor(self.source, dtype=torch.float64)
        time = traveltime_torch(points, source, self.model)
        gradient = torch.autograd.grad(time.sum(), points)[0]
        residual = gradient.square().sum(-1, keepdim=True) - self.model.torch(points).reciprocal().square()
        self.assertLess(residual.abs().max().item(), 1e-12)


if __name__ == "__main__":
    unittest.main()
