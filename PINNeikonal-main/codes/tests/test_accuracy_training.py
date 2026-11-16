"""Check full-batch gradients and per-source sampling in the 2D refinement."""

import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from codes.accuracy_training import lbfgs_refine, stratified_source_indices


class AccuracyTrainingTests(unittest.TestCase):
    def test_source_sampling_is_balanced_disjoint_and_repeatable(self):
        source_x = np.repeat([.1, 2.1, 4., 5.9], 20)
        source_z = np.ones(80)
        first = stratified_source_indices(source_x, source_z, 7)
        np.testing.assert_array_equal(first, stratified_source_indices(source_x, source_z, 7))
        self.assertEqual(len(first), 28)
        self.assertEqual(len(np.unique(first)), 28)
        self.assertTrue(all(np.sum(source_x[first] == source) == 7 for source in np.unique(source_x)))

    def test_full_batch_gradient_matches_finite_difference_and_refines(self):
        import tensorflow as tf
        from sciann import Functional, Variable, SciModel
        from sciann.utils import K

        K.clear_session()
        tf.random.set_seed(7)
        x = Variable("accuracy_x", dtype="float64")
        value = Functional("accuracy_value", [x], [5], "tanh", dtype="float64")
        model = SciModel([x], [value])
        coordinates = np.linspace(-1, 1, 21).reshape(-1, 1)
        desired = .7+.2*coordinates
        refined, report = lbfgs_refine(model, [coordinates], [desired], iterations=50,
                                       log_every=100, verify_gradient=True)
        self.assertLess(report["directional_gradient_relative_error"], 1e-4)
        self.assertLess(report["final_loss"], report["initial_loss"]*.001)
        prediction = value.eval(refined, [coordinates])
        self.assertLess(np.max(np.abs(prediction-desired)), .003)


if __name__ == "__main__":
    unittest.main()
