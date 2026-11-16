"""Run numerical/unit validation and honestly report unavailable dependencies.

Examples: python validate.py --smoke; python validate.py --math-only --smoke.
The math-only flag explicitly excludes PyTorch training verification.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pinneik3d"

import numpy as np

from .common import PROJECT_ROOT, load_config, project_path, source_hash, write_csv, write_json
from .benchmarks.measurements import hardware_metadata
from .benchmarks.metrics import aggregate_records, error_metrics
from .fsm3d import NUMBA_AVAILABLE, solve_fsm
from .grids import grid_batches, sample_velocity_grid
from .models import build_velocity, tau_numpy, traveltime_numpy


def syntax_and_configs() -> Dict[str, Any]:
    """Parse all project Python and all YAML configs without writing bytecode."""
    files = sorted(path for path in PROJECT_ROOT.rglob("*.py")
                   if ".venv" not in path.parts and "results" not in path.parts)
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
    configs = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    for path in configs:
        load_config(path)
    from .benchmarks.run import suite_plan
    full_plan = suite_plan(load_config("configs/benchmark.yaml"))
    return {"python_files_parsed": len(files), "yaml_configs_validated": len(configs),
            "full_benchmark_training_runs": len(full_plan), "python_syntax_minimum": "3.8"}


def unit_validation() -> Dict[str, Any]:
    """Run the complete unittest suite; retain failure traces and skip reasons."""
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"),
                                                top_level_dir=str(PROJECT_ROOT.parent))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"tests_run": result.testsRun, "tests_passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
            "tests_skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
            "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
            "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
            "executed_tests_passed": result.wasSuccessful(),
            "all_tests_verified": result.wasSuccessful() and not result.skipped}


def _finite_gradient(function: Any, points: np.ndarray, step: float = 1e-5) -> np.ndarray:
    components = []
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        components.append((function(points+delta)-function(points-delta))/(2*step))
    return np.stack(components, axis=-1)


def mathematical_smoke() -> Dict[str, Any]:
    """Three-seed 31³ numerical smoke; Python FSM times never count as baseline timings."""
    from .benchmarks.plotting import plot_fields

    config = load_config("configs/smoke.yaml")
    velocity = build_velocity(config["velocity"], config["domain"]["bounds"])
    source = np.asarray(config["domain"]["source"])
    n = config["evaluation"]["grid_size"]
    backend = "numba" if NUMBA_AVAILABLE else "python"
    folder = project_path("results/validation", output=True)
    rows = []
    artifacts: List[str] = []
    for index, seed in enumerate(config["seeds"]):
        started = time.perf_counter()
        points = np.random.RandomState(seed).uniform(.05, 1.95, (300, 3))
        gradient = _finite_gradient(lambda x: traveltime_numpy(x, source, velocity), points)
        equation = np.sum(gradient**2, axis=-1)*velocity.numpy(points)**2-1
        tau = tau_numpy(points, source, velocity)
        gradient_tau = _finite_gradient(lambda x: tau_numpy(x, source, velocity), points)
        displacement = points-source
        radius = np.linalg.norm(displacement, axis=-1)
        vs = float(velocity.numpy(source))
        t0 = radius/vs
        gradient_t0 = displacement/(radius[:, None]*vs)
        residual = (t0**2*np.sum(gradient_tau**2, axis=-1)
                    + tau**2*np.sum(gradient_t0**2, axis=-1)
                    + 2*t0*tau*np.sum(gradient_t0*gradient_tau, axis=-1)
                    - 1/velocity.numpy(points)**2)
        volume = sample_velocity_grid(velocity, config["domain"]["bounds"], n)
        fsm = solve_fsm(volume, config["domain"]["bounds"], source, vs,
                        tolerance=config["fsm"]["tolerance"], max_cycles=config["fsm"]["max_cycles"], backend=backend)
        exact = np.empty(n**3, dtype=np.float64)
        for start, stop, coordinates in grid_batches(config["domain"]["bounds"], n, 262144, np.float64):
            exact[start:stop] = traveltime_numpy(coordinates, source, velocity)
        exact = exact.reshape((n, n, n))
        row = {"seed": seed, "grid_n": n, "fsm_backend": backend,
               "max_eikonal_relative_residual": float(np.max(np.abs(equation))),
               "max_factored_residual": float(np.max(np.abs(residual))),
               "fsm_cycles": fsm.cycles, "fsm_converged": fsm.converged,
               "fsm_seconds": fsm.seconds if backend == "numba" else None,
               "diagnostic_wall_seconds": time.perf_counter()-started,
               "timing_valid_for_baseline": backend == "numba", **error_metrics(fsm.traveltime, exact),
               "config": config}
        if row["max_eikonal_relative_residual"] > 2e-7 or row["max_factored_residual"] > 2e-8 or row["relative_l2"] > .015:
            raise AssertionError("Numerical smoke failed its analytic/FSM accuracy checks")
        rows.append(row)
        print("Numerical smoke seed=%d grid=%d^3 backend=%s relL2=%.6g wall=%.2fs" % (
            seed, n, backend, row["relative_l2"], row["diagnostic_wall_seconds"]), flush=True)
        if index == 0:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(folder / "field.npz"), reference=exact, fsm=fsm.traveltime,
                                source=source, bounds=config["domain"]["bounds"], reference_kind="analytic")
            artifacts = plot_fields(None, exact, fsm.traveltime, config["domain"]["bounds"], source,
                                    folder / "figures", "analytic")
    hardware = hardware_metadata()
    for row in rows:
        row["hardware"] = hardware
    write_csv(folder / "mathematical_smoke.csv", rows)
    write_json(folder / "mathematical_smoke.json", rows)
    aggregate = aggregate_records(rows, ["grid_n", "fsm_backend"],
        ["relative_l2", "max_abs_error", "mean_abs_error", "max_eikonal_relative_residual", "max_factored_residual"])
    write_csv(folder / "aggregate.csv", aggregate)
    write_json(folder / "aggregate.json", aggregate)
    return {"status": "passed", "scope": "analytic equation, analytic tau, numerical factored FSM; no PINN inference",
            "seeds": config["seeds"], "runs": rows, "aggregate": aggregate, "artifacts": artifacts}


def _validation_summary(validation: Dict[str, Any], smoke: Optional[Dict[str, Any]]) -> None:
    """Write a factual status table; do not produce unmeasured training numbers."""
    unit = validation["unit_tests"]
    hardware = validation["hardware"]
    lines = ["# PINNeik3D validation status", "", "This file reports checks actually performed in this workspace. Full PyTorch performance experiments have not been measured unless training results are explicitly included below.", "",
             "| Check | Result |\n|---|---|",
             "| Python syntax | %d files parsed with Python 3.8 grammar |" % validation["static_checks"]["python_files_parsed"],
             "| YAML configurations | %d validated |" % validation["static_checks"]["yaml_configs_validated"],
             "| Executed unit tests | %d passed; %d failed; %d errors |" % (unit["tests_passed"], len(unit["failures"]), len(unit["errors"])),
             "| Skipped checks | %d; see validation.json for individual reasons |" % len(unit["tests_skipped"]),
             "| Dependency availability | Missing: %s |" % (", ".join(validation["missing_training_dependencies"]) or "none"),
             "| Python | %s |" % hardware["python_version"],
             "| GPU detected | %s |" % (", ".join(hardware["gpu_models_detected"]) or "none"),
             "| PyTorch / CUDA | %s / %s |" % (hardware["torch_version"] or "unavailable", hardware["cuda_version"] or "unavailable")]
    if smoke:
        math = smoke.get("mathematical_smoke")
        if math:
            row = math["aggregate"][0]
            lines += ["", "Numerical 31³ smoke, mean ± sample standard deviation over seeds 42, 43, and 44:", "",
                      "| Backend | Relative L2 | Max absolute error (s) | Mean absolute error (s) |",
                      "|---|---:|---:|---:|"]
            lines.append("| %s | %.6g ± %.3g | %.6g ± %.3g | %.6g ± %.3g |" % (
                row["fsm_backend"], row["relative_l2_mean"], row["relative_l2_std"],
                row["max_abs_error_mean"], row["max_abs_error_std"], row["mean_abs_error_mean"], row["mean_abs_error_std"]))
            if row["fsm_backend"] == "python":
                lines += ["", "This is the diagnostic Python implementation of the factored stencil. Its wall times are excluded from FSM benchmark columns; the Numba-compiled kernel is not verified in this environment."]
            lines += ["", "Actual analytic/FSM error-slice and contour figures: `validation/figures/error_slices.png` and `validation/figures/traveltime_contours.png` (also PDF). No PINN field is fabricated."]
        training = smoke["training_smoke"]
        lines += ["", "| Training smoke | Status |\n|---|---|", "| 200 Adam iterations, 31³ evaluation, 3 CPU seeds | %s |" % training["status"]]
        if training["status"] == "completed":
            lines += ["| Total training smoke wall time | %.3f s; under 120 s: %s |" % (training["wall_seconds"], training["under_two_minutes"])]
        else:
            lines += ["", training["reason"], "", "The CPU training-under-two-minutes target, mixed precision, GPU timings/memory, full 51³/101³/201³ scaling, surrogate accuracy, transfer speedups, and cost break-even remain unverified."]
    lines += ["", "Run `python validate.py --smoke` after installing requirements. Run the full commands in README.md to replace this status report with measured paper tables. The full benchmark YAML defines %d independent training runs, each setting with three seeds." % validation["static_checks"]["full_benchmark_training_runs"], "",
              "Implemented choices: stable asinh form of the acosh analytic solution; guarded gradient denominator with exact T0(xs)=0; positivity uses square of the mean as specified; off-grid FSM uses a shrinking local injection region; optional scikit-fmm is used only for grid-aligned sources."]
    project_path("results/summary.md", output=True).write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 31³ numerical smoke and training when dependencies exist")
    parser.add_argument("--math-only", action="store_true", help="Explicitly verify only the available mathematical/numerical scope")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        static = syntax_and_configs()
        units = unit_validation()
        missing = [module for module in ("torch", "numba") if importlib.util.find_spec(module) is None]
        validation = {"status": "passed" if units["executed_tests_passed"] and not missing else "partial",
                      "scope": "math_only" if args.math_only else "full", "static_checks": static,
                      "unit_tests": units, "missing_training_dependencies": missing,
                      "source_hash": source_hash(),
                      "hardware": hardware_metadata(), "elapsed_seconds": time.perf_counter()-started}
        evidence = None
        if args.smoke and units["executed_tests_passed"]:
            numerical = mathematical_smoke()
            if missing or args.math_only:
                training = {"status": "not_requested" if args.math_only else "blocked",
                            "reason": "Training was explicitly excluded by --math-only." if args.math_only else
                                      "PyTorch/Numba are unavailable. Dependency downloads were blocked by network socket permissions (WinError 10013) during this workspace session.",
                            "wall_seconds": None, "under_two_minutes": None}
            else:
                from .train import run_training
                train_started = time.perf_counter()
                records = run_training(load_config("configs/smoke.yaml"))
                wall_seconds = time.perf_counter()-train_started
                training = {"status": "completed", "runs": records, "wall_seconds": wall_seconds,
                            "under_two_minutes": wall_seconds < 120}
            evidence = {"mathematical_smoke": numerical, "training_smoke": training,
                        "hardware": validation["hardware"]}
            write_json("results/smoke_evidence.json", evidence)
        elif project_path("results/smoke_evidence.json").is_file():
            evidence = json.loads(project_path("results/smoke_evidence.json").read_text(encoding="utf-8"))
            validation["previous_smoke_evidence_retained"] = True
        validation["elapsed_seconds"] = time.perf_counter()-started
        write_json("results/validation.json", validation)
        _validation_summary(validation, evidence)
        print(json.dumps({"validation_status": validation["status"], "tests_passed": units["tests_passed"],
                          "tests_skipped": len(units["tests_skipped"]), "missing": missing,
                          "report": str(project_path("results/summary.md"))}, indent=2))
        if not units["executed_tests_passed"]:
            return 1
        return 0 if args.math_only or not missing else 2
    except (AssertionError, RuntimeError, ValueError, ImportError, OSError) as exc:
        print("PINNeik3D validation: "+str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
