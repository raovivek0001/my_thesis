"""Verify physical coordinate gradients and neuron-wise adaptive parameters."""

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class NetworkTests(unittest.TestCase):
    def test_adaptive_activation_initialization_and_trainability(self):
        import torch
        from pinneik3d.networks import AdaptiveAtan

        activation = AdaptiveAtan(5, 10)
        inputs = torch.linspace(-2, 2, 5, requires_grad=True)
        torch.testing.assert_close(activation(inputs), torch.atan(inputs))
        activation(inputs).sum().backward()
        self.assertIsNotNone(activation.slopes.grad)
        self.assertEqual(activation.slopes.numel(), 5)

    def test_fourier_surrogate_gradients_and_checkpoint(self):
        import torch
        from pinneik3d.networks import EikonalMLP

        torch.manual_seed(4)
        options = dict(hidden_layers=2, width=12, surrogate=True,
                       fourier={"enabled": True, "size": 8, "scale": 1.5})
        network = EikonalMLP([[0, 2], [0, 3], [0, 4]], **options)
        points = torch.tensor([[.7, 1.1, 1.3]], requires_grad=True)
        source = torch.tensor([[.4, .5, .6]])
        gradient = torch.autograd.grad(network(points, source).sum(), points)[0]
        differences = []
        for axis in range(3):
            delta = torch.zeros_like(points)
            delta[0, axis] = 1e-3
            differences.append(((network(points+delta, source)-network(points-delta, source))/(2e-3)).item())
        np.testing.assert_allclose(gradient.numpy().ravel(), differences, atol=2e-4, rtol=2e-2)
        clone = EikonalMLP([[0, 2], [0, 3], [0, 4]], **options)
        clone.load_state_dict(network.state_dict())
        torch.testing.assert_close(clone(points, source), network(points, source))
