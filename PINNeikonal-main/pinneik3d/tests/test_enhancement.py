"""Accuracy workflow budgets, residual scaling, sampling state and selection."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pinneik3d.common import DEFAULTS, PROJECT_ROOT, deep_merge, validate_config
from pinneik3d.compare import comparison_configs, summarize_pairs
from pinneik3d.sampling import ranked_indices


class EnhancementConfigTests(unittest.TestCase):
    def test_comparison_matches_optimizer_budgets_and_grids(self):
        baseline, enhanced = comparison_configs("configs/smoke.yaml", "configs/enhanced_smoke.yaml")
        for section in ("domain", "velocity", "surrogate", "runtime", "evaluation", "seeds", "fsm"):
            self.assertEqual(baseline[section], enhanced[section])
        for key in ("iterations", "batch_size", "lr", "lr_min", "lbfgs", "eval_every"):
            self.assertEqual(baseline["training"][key], enhanced["training"][key])
        self.assertFalse(baseline["training"]["sampling"]["enabled"])
        self.assertTrue(enhanced["training"]["sampling"]["enabled"])

    def test_invalid_sampling_and_normalization_are_rejected(self):
        for override in ({"loss": {"residual_normalization": "unknown"}},
                         {"training": {"sampling": {"uniform_fraction": 0.0}}},
                         {"training": {"sampling": {"update_every": 0}}},
                         {"training": {"sampling": {"candidate_multiplier": 1.5}}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_config(deep_merge(DEFAULTS, override))

    def test_ranked_candidates_use_magnitude_and_reject_nonfinite(self):
        np.testing.assert_array_equal(ranked_indices([1., -9., 9., 2.], 2), [1, 2])
        with self.assertRaises(ValueError):
            ranked_indices([1., np.nan], 1)

    def test_summary_reports_regression_and_requires_paired_seeds(self):
        rows = [{"variant": variant, "seed": seed, "relative_l2": error,
                 "max_abs_error": error, "mean_abs_error": error,
                 "training_seconds": 1., "inference_seconds": .1, "candidate_points_scored": 0}
                for variant, error in (("baseline", .01), ("enhanced", .02)) for seed in (42, 43, 44)]
        summary = summarize_pairs(rows)
        self.assertFalse(summary["accuracy_improved"])
        self.assertAlmostEqual(summary["relative_l2_reduction_percent"], -100.)
        with self.assertRaises(ValueError):
            summarize_pairs(rows[:-1])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class EnhancementTrainingTests(unittest.TestCase):
    def test_relative_residual_is_invariant_to_velocity_units(self):
        import torch
        from pinneik3d.losses import loss_terms
        from pinneik3d.models import ConstantGradient

        class Trial(torch.nn.Module):
            def forward(self, points, sources=None):
                return 1.1+.02*points[:, :1]

        points = torch.tensor([[.2, .3, .4], [1.4, 1.6, 1.8]], requires_grad=True)
        sources = torch.ones_like(points)
        values = []
        for scale in (1., 1000.):
            velocity = ConstantGradient(2*scale, (0., 0., .5*scale))
            values.append(loss_terms(Trial(), points, sources, sources[:1], velocity,
                                     residual_normalization="relative")["pde"])
        torch.testing.assert_close(values[0], values[1], rtol=2e-5, atol=1e-7)

    def test_adaptive_sampling_resumes_exactly_and_best_weights_load(self):
        import torch
        from pinneik3d.common import load_config
        from pinneik3d.fsm3d import NUMBA_AVAILABLE
        from pinneik3d.train import load_model_checkpoint, train_one

        config = load_config("configs/enhanced_smoke.yaml", {
            "seeds": [42], "network": {"hidden_layers": 2, "width": 8},
            "loss": {"adaptive": {"enabled": False}},
            "training": {"iterations": 4, "batch_size": 16, "eval_every": 2,
                         "log_every": 2, "checkpoint_every": 2,
                         "sampling": {"warmup": 0, "update_every": 2, "candidate_multiplier": 2},
                         "lbfgs": {"enabled": False}},
            "evaluation": {"grid_size": 5, "validation_grid_size": 5,
                           "plots": False, "save_field": False, "timing_repeats": 1},
            "fsm": {"backend": "numba" if NUMBA_AVAILABLE else "python"},
        })
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as directory:
            root = Path(directory)
            full = train_one(config, 42, root / "full")
            checkpoint = full.run_dir / "checkpoint_iter_000002.pt"
            resumed = train_one(deep_merge(config, {"training": {"resume_checkpoint": str(checkpoint)}}),
                                42, root / "resumed")
            for name, value in full.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(full.metrics["candidate_points_scored"], 96)
            self.assertEqual(full.metrics["candidate_points_scored"], resumed.metrics["candidate_points_scored"])
            observed = min(row["validation_relative_l2"] for row in full.history if "validation_relative_l2" in row)
            self.assertEqual(full.metrics["selected_validation"]["relative_l2"], observed)
            restored, _, _ = load_model_checkpoint(full.run_dir / "best_model.pt")
            for name, value in full.model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
