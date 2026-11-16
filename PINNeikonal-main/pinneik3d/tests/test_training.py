"""Short real optimizer tests, checkpoint determinism, and surrogate separation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pinneik3d.common import PROJECT_ROOT, deep_merge, load_config
from pinneik3d.surrogate import evaluation_sources, training_sources


class SourceTests(unittest.TestCase):
    def test_64_sources_and_distant_disjoint_holdout(self):
        config = load_config("configs/surrogate.yaml")
        lattice = training_sources(config)
        self.assertEqual(lattice.shape, (64, 3))
        held_out = evaluation_sources(config)
        self.assertTrue(all(r["nearest_training_source_km"] > 0 for r in held_out))
        far = next(r for r in held_out if r["kind"] == "far")
        self.assertTrue(far["outside_training_source_box"])
        self.assertGreater(far["nearest_training_source_km"], .65)

    def test_training_source_cannot_be_labeled_held_out(self):
        config = load_config("configs/surrogate.yaml")
        config["surrogate"]["held_out"] = [[.4, .4, .4]]
        with self.assertRaises(ValueError):
            evaluation_sources(config)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TrainingTests(unittest.TestCase):
    def tiny_config(self):
        from pinneik3d.fsm3d import NUMBA_AVAILABLE
        return load_config("configs/smoke.yaml", {
            "seeds": [42], "network": {"hidden_layers": 2, "width": 8},
            "loss": {"adaptive": {"enabled": False}},
            "training": {"iterations": 4, "batch_size": 32, "eval_every": 2,
                         "log_every": 2, "checkpoint_every": 2},
            "evaluation": {"grid_size": 5, "validation_grid_size": 5,
                           "plots": False, "save_field": False, "timing_repeats": 1},
            "fsm": {"backend": "numba" if NUMBA_AVAILABLE else "python"},
        })

    def test_checkpoint_resume_matches_uninterrupted_weights(self):
        import torch
        from pinneik3d.train import train_one

        config = self.tiny_config()
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as directory:
            root = Path(directory)
            full = train_one(config, 42, root / "full")
            checkpoint = full.run_dir / "checkpoint_iter_000002.pt"
            resumed_config = deep_merge(config, {"training": {"resume_checkpoint": str(checkpoint)}})
            resumed = train_one(resumed_config, 42, root / "resumed")
            for key, value in full.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(np.isfinite(resumed.metrics["relative_l2"]))

    def test_gridded_physics_backpropagates_without_grid_sample_double_backward(self):
        import torch
        from pinneik3d.losses import loss_terms
        from pinneik3d.models import GriddedVelocity
        from pinneik3d.networks import EikonalMLP

        axis = np.linspace(0, 2, 5)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        velocity = GriddedVelocity(2+.1*x+.2*y+.3*z, [[0, 2]]*3)
        model = EikonalMLP([[0, 2]]*3, hidden_layers=2, width=8)
        points = torch.rand(32, 3, requires_grad=True)*1.8+.1
        points = points.detach().requires_grad_(True)
        sources = torch.ones_like(points)
        terms = loss_terms(model, points, sources, sources[:1], velocity)
        sum(terms.values()).backward(inputs=tuple(model.parameters()))
        self.assertIsNone(points.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_lbfgs_uses_fixed_weights_and_evaluates_closure(self):
        from pinneik3d.train import train_one

        config = deep_merge(self.tiny_config(), {"training": {
            "iterations": 2, "lbfgs": {"enabled": True, "iterations": 2, "max_iter": 2,
                                         "batch_size": 32, "history_size": 5}}})
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as directory:
            result = train_one(config, 42, directory)
            rows = [row for row in result.history if row["stage"] == "lbfgs"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["loss_weights"], rows[1]["loss_weights"])
            self.assertGreaterEqual(result.metrics["lbfgs_closure_evaluations"], 2)
