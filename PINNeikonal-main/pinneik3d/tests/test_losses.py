"""Numerical residual and positivity loss tests, including source guarding."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class LossTests(unittest.TestCase):
    def test_factored_residual_analytic_tau_autograd(self):
        import torch
        from pinneik3d.losses import residual_from_tau
        from pinneik3d.models import ConstantGradient, tau_torch

        torch.manual_seed(1)
        points = (torch.rand(200, 3, dtype=torch.float64)*1.8+.1).requires_grad_()
        source = torch.tensor([.7, .8, .9], dtype=torch.float64).expand_as(points)
        model = ConstantGradient(2., (.1, -.1, .5))
        tau = tau_torch(points, source, model)
        residual = residual_from_tau(tau, points, source, model)
        self.assertLess(residual.abs().max().item(), 2e-12)

    def test_source_background_is_finite_and_exactly_zero(self):
        import torch
        from pinneik3d.losses import background_factor

        source = torch.tensor([[1., 1., 1.]])
        t0, gradient = background_factor(source, source, torch.tensor([[2.5]]))
        self.assertEqual(t0.item(), 0.)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_loss_squares_mean_negative_factor(self):
        import torch
        from pinneik3d.losses import loss_terms
        from pinneik3d.models import ConstantGradient

        class Trial(torch.nn.Module):
            def forward(self, points, sources=None):
                return points[:, :1] - 1

        points = torch.tensor([[0., .2, .3], [2., .2, .3]], requires_grad=True)
        sources = torch.tensor([[1., 1., 1.]]).expand_as(points)
        terms = loss_terms(Trial(), points, sources, sources[:1], ConstantGradient())
        self.assertAlmostEqual(terms["positivity"].item(), .25)

    def test_gradient_balancing_uses_gradient_statistics(self):
        import torch
        from pinneik3d.losses import GradientLossBalancer

        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        terms = {"pde": parameter.square().sum(), "source": parameter.sum().square(),
                 "positivity": torch.relu(-parameter).mean().square()}
        balancer = GradientLossBalancer(beta=0.)
        balancer.update(terms, [parameter])
        self.assertAlmostEqual(balancer.weights["source"], 4/6)
        self.assertEqual(balancer.weights["positivity"], 1.)
