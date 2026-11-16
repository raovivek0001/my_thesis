"""Accuracy, refinement, point sources, and convergence checks for factored FSM."""

import importlib.util
import unittest

import numpy as np

from pinneik3d.fsm3d import NUMBA_AVAILABLE, solve_fsm
from pinneik3d.grids import grid_batches, sample_velocity_grid
from pinneik3d.models import ConstantGradient, traveltime_numpy


class FSMTests(unittest.TestCase):
    bounds = [[0, 2], [0, 2], [0, 2]]
    backend = "numba" if NUMBA_AVAILABLE else "python"

    def field_error(self, n, source, model):
        volume = sample_velocity_grid(model, self.bounds, n)
        result = solve_fsm(volume, self.bounds, source, float(model.numpy(np.asarray(source))),
                           tolerance=1e-10, backend=self.backend)
        points = next(grid_batches(self.bounds, n, n**3, np.float64))[2]
        exact = traveltime_numpy(points, np.asarray(source), model).reshape((n, n, n))
        relative = np.linalg.norm(result.traveltime-exact) / np.linalg.norm(exact)
        return result, float(relative), float(np.max(np.abs(result.traveltime-exact)))

    def test_constant_medium_is_exact_after_factorization(self):
        result, relative, maximum = self.field_error(9, [1, 1, 1], ConstantGradient(2, [0, 0, 0]))
        self.assertTrue(result.converged)
        self.assertLess(relative, 1e-12)
        self.assertLess(maximum, 1e-12)
        self.assertEqual(result.traveltime[4, 4, 4], 0)

    def test_constant_gradient_first_order_convergence(self):
        model = ConstantGradient(2, [0, 0, .5])
        coarse, coarse_error, coarse_max = self.field_error(9, [1, 1, 1], model)
        fine, fine_error, fine_max = self.field_error(17, [1, 1, 1], model)
        self.assertLess(coarse_error, .025)
        self.assertLess(fine_error, .015)
        self.assertLess(fine_max, .04)
        self.assertLess(fine_error, .75*coarse_error)
        self.assertLess(fine_max, coarse_max)

    def test_off_grid_source_is_preserved(self):
        source = [.87, 1.13, .93]
        result, relative, maximum = self.field_error(9, source, ConstantGradient())
        self.assertEqual(result.source_injection, "off_grid_local_ball")
        self.assertLess(relative, .04)
        self.assertGreater(np.min(result.traveltime), 0)  # No falsely snapped zero.

    def test_unconverged_result_is_not_silently_accepted(self):
        volume = np.full((5, 5, 5), 2.)
        with self.assertRaises(RuntimeError):
            solve_fsm(volume, self.bounds, [1, 1, 1], max_cycles=1, backend=self.backend)

    def test_anisotropic_grid_spacing_in_constant_medium(self):
        bounds = [[0, 2], [0, 3], [0, 4]]
        result = solve_fsm(np.full((9, 7, 5), 2.), bounds, [1, 1.5, 2], backend=self.backend)
        points = next(grid_batches(bounds, (9, 7, 5), 9*7*5, np.float64))[2]
        exact = np.linalg.norm(points-[1, 1.5, 2], axis=-1)/2
        np.testing.assert_allclose(result.traveltime.ravel(), exact, atol=2e-12)

    def test_heterogeneous_reference_uses_twice_as_many_cells(self):
        from pinneik3d.common import load_config
        from pinneik3d.models import SmoothHeterogeneous
        from pinneik3d.reference import make_reference

        config = load_config("configs/smoke.yaml", {"fsm": {"backend": self.backend},
                                                   "evaluation": {"reference": "fsm"}})
        reference = make_reference(SmoothHeterogeneous(), self.bounds, [1, 1, 1], 5, config)
        self.assertEqual(reference.kind, "fsm_fine")
        self.assertEqual(reference.metadata["reference_grid"], [9, 9, 9])
        field = reference.grid(5)
        self.assertTrue(np.all(np.isfinite(field)))
        self.assertGreaterEqual(np.min(field), 0)
        self.assertEqual(field[2, 2, 2], 0)

    @unittest.skipUnless(importlib.util.find_spec("skfmm"), "Optional scikit-fmm is not installed")
    def test_optional_fmm_point_source(self):
        from pinneik3d.fsm3d import solve_fmm
        field, seconds = solve_fmm(np.full((17, 17, 17), 2.), self.bounds, [1, 1, 1])
        points = next(grid_batches(self.bounds, 17, 17**3, np.float64))[2]
        exact = np.linalg.norm(points-[1, 1, 1], axis=-1)/2
        self.assertEqual(field[8, 8, 8], 0.)
        self.assertLess(np.linalg.norm(field.ravel()-exact)/np.linalg.norm(exact), .06)
