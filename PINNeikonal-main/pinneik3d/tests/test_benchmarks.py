"""Accuracy statistics, censored times, and strict amortization inequality."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pinneik3d.benchmarks.metrics import (aggregate_records, break_even_sources, error_metrics,
                                         first_threshold_times, threshold_key)


class BenchmarkTests(unittest.TestCase):
    def test_error_metrics(self):
        scores = error_metrics(np.array([1., 3.]), np.array([1., 2.]), chunk_size=1)
        self.assertAlmostEqual(scores["relative_l2"], 1/np.sqrt(5))
        self.assertEqual(scores["max_abs_error"], 1.)
        self.assertEqual(scores["mean_abs_error"], .5)
        with self.assertRaises(ValueError):
            error_metrics(np.array([np.nan]), np.array([1.]))

    def test_thresholds_remain_censored(self):
        history = [{"elapsed_seconds": 1., "validation_relative_l2": .1},
                   {"elapsed_seconds": 3., "validation_relative_l2": .008}]
        values = first_threshold_times(history, [.01, .001])
        self.assertEqual(values[threshold_key(.01)], 3.)
        self.assertIsNone(values[threshold_key(.001)])

    def test_mean_sample_std_and_success_count(self):
        records = [{"seed": i, "grid": 51, "error": float(i+1),
                    "target_time": None if i == 2 else float(i+1)} for i in range(3)]
        row = aggregate_records(records, ["grid"], ["error", "target_time"])[0]
        self.assertEqual(row["error_mean"], 2.)
        self.assertEqual(row["error_std"], 1.)
        self.assertEqual(row["target_time_count"], 2)
        self.assertEqual(row["seed_count"], 3)

    def test_break_even_strict_inequality(self):
        self.assertEqual(break_even_sources(100., 1., 11.), 11)
        self.assertIsNone(break_even_sources(100., 11., 11.))
        self.assertIsNone(break_even_sources(100., 12., 11.))

    def test_suite_enforces_three_seeds_and_exact_budgets(self):
        from pinneik3d.common import load_config
        from pinneik3d.benchmarks.run import suite_plan

        config = load_config("configs/benchmark_smoke.yaml")
        jobs = suite_plan(config)
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j["seed"] for j in jobs}, {42, 43, 44})
        self.assertTrue(all(j["config"]["training"]["iterations"]*j["config"]["training"]["batch_size"] == 4096 for j in jobs))
        config["seeds"] = [42]
        with self.assertRaises(ValueError):
            suite_plan(config)

    def test_unrepresentable_budget_is_rejected(self):
        from pinneik3d.common import load_config
        from pinneik3d.benchmarks.run import suite_plan

        config = load_config("configs/benchmark_smoke.yaml")
        config["benchmarks"]["collocation_budgets"] = [4097]
        with self.assertRaises(ValueError):
            suite_plan(config)

    def test_worker_crash_preserves_only_matching_completed_rows(self):
        from pinneik3d.common import PROJECT_ROOT, load_config, write_json
        from pinneik3d.benchmarks.run import read_worker_result, suite_plan

        job = suite_plan(load_config("configs/benchmark_smoke.yaml"))[0]
        config = job["config"]
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as directory:
            filename = Path(directory) / "worker_results.json"
            actual = {"status": "completed", "seed": 42, "grid_n": 11, "hardware": {}}
            write_json(filename, {"status": "in_progress", "records": [actual], "breaks": [],
                                  "run_spec_hash": config["benchmarks"]["run_spec_hash"]})
            payload = read_worker_result(filename, config, 42, 137)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual(payload["records"][0], actual)
            self.assertEqual(payload["records"][1]["status"], "failed")
            write_json(filename, {"status": "completed", "records": [actual], "breaks": [], "run_spec_hash": "stale"})
            stale = read_worker_result(filename, config, 42, 137)
            self.assertEqual(len(stale["records"]), 1)
            self.assertEqual(stale["records"][0]["status"], "failed")
