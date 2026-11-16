"""Reproducible scaling studies with one fresh worker process per seed/run."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pinneik3d.benchmarks"

import numpy as np
import yaml

from ..common import (Config, config_hash, deep_merge, load_config, project_path,
                      source_hash, validate_config, write_csv, write_json)
from ..grids import grid_shape, sample_velocity_grid
from ..reference import make_reference
from ..surrogate import evaluation_sources
from .measurements import hardware_fingerprint, hardware_metadata
from .metrics import break_even_sources, error_metrics


def _spec_hash(config: Config) -> str:
    specification = copy.deepcopy(config)
    specification.pop("config_path", None)
    specification.get("benchmarks", {}).pop("run_spec_hash", None)
    return config_hash(specification)


def suite_plan(config: Config) -> List[Dict[str, Any]]:
    """Validate the matrix and create exact collocation budgets and run specs."""
    options = config.get("benchmarks", {})
    required = ("experiments", "grid_sizes", "batch_sizes", "collocation_budgets", "precisions", "devices", "modes")
    if any(not options.get(key) for key in required):
        raise ValueError("benchmarks must define nonempty experiment/grid/batch/budget/precision/device/mode lists")
    if len(config["seeds"]) < 3:
        raise ValueError("Every benchmark setting requires at least three independent seeds")
    if config["fsm"]["backend"] != "numba":
        raise ValueError("Benchmarks require the compiled numba FSM backend")
    if any(not isinstance(n, int) or n < 3 for n in options["grid_sizes"]):
        raise ValueError("Benchmark grid_sizes must be integer cube sizes >=3")
    if any(not isinstance(n, int) or n <= 0 for n in options["batch_sizes"]+options["collocation_budgets"]):
        raise ValueError("Batch sizes and collocation budgets must be positive integers")
    if any(budget % batch for budget in options["collocation_budgets"] for batch in options["batch_sizes"]):
        raise ValueError("Each collocation budget must be exactly divisible by every batch size")
    if set(options["precisions"])-{"float32", "mixed"} or set(options["modes"])-{"single", "surrogate"}:
        raise ValueError("Unknown benchmark precision or mode")
    if set(options["devices"])-{"auto", "cpu", "cuda"}:
        raise ValueError("Benchmark devices must be auto, cpu, or cuda")
    if options.get("inference_repeats", 3) <= 0 or options.get("accuracy_target", .01) <= 0:
        raise ValueError("Inference repetitions and accuracy target must be positive")
    smallest_grid = min(options["grid_sizes"])
    root = project_path(config["output_dir"], output=True)
    code_identity = source_hash()
    jobs = []
    for experiment_path in options["experiments"]:
        preset = load_config(experiment_path)
        if preset["training"].get("initial_checkpoint") or preset["training"].get("resume_checkpoint"):
            raise ValueError("Scaling presets must start from random initialization")
        experiment = preset["name"]
        for batch, budget, precision, device, mode, seed in itertools.product(
            options["batch_sizes"], options["collocation_budgets"], options["precisions"],
            options["devices"], options["modes"], config["seeds"],
        ):
            iterations = budget//batch
            identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", "%s_%s_%s_%s_b%d_c%d_s%d" % (
                experiment, mode, precision, device, batch, budget, seed))
            representative = (seed == config["seeds"][0] and batch == options["batch_sizes"][0]
                              and budget == max(options["collocation_budgets"])
                              and precision == options["precisions"][0] and device == options["devices"][0])
            overrides = {
                "name": experiment+"_"+mode, "seeds": config["seeds"],
                "output_dir": str(root / "runs" / identifier),
                "runtime": {"device": device, "precision": precision,
                            "num_threads": config["runtime"]["num_threads"],
                            "deterministic": config["runtime"]["deterministic"]},
                "training": {"iterations": iterations, "batch_size": batch,
                    "eval_every": min(preset["training"]["eval_every"], max(1, iterations//20)),
                    "log_every": min(preset["training"]["log_every"], max(1, iterations//20)),
                    "checkpoint_every": min(preset["training"]["checkpoint_every"], iterations),
                    "lbfgs": {"enabled": options.get("enable_lbfgs", False)}},
                "evaluation": {"grid_size": smallest_grid,
                    "validation_grid_size": min(smallest_grid, preset["evaluation"]["validation_grid_size"]),
                    "batch_size": config["evaluation"]["batch_size"],
                    "plots": representative, "save_field": representative and options.get("save_fields", True),
                    "timing_repeats": options.get("inference_repeats", 3)},
                "surrogate": {"enabled": mode == "surrogate"}, "fsm": config["fsm"],
                "benchmarks": {**options, "experiment": experiment, "mode": mode,
                    "run_id": identifier, "representative": representative, "source_hash": code_identity},
            }
            effective = deep_merge(preset, overrides)
            validate_config(effective)
            evaluation_sources(effective)
            effective["benchmarks"]["run_spec_hash"] = _spec_hash(effective)
            jobs.append({"run_id": identifier, "seed": seed, "config": effective})
    return jobs


def _optional_fmm(velocity: Any, source: Sequence[float], config: Config,
                   n: int, reference: np.ndarray) -> Dict[str, Any]:
    if not config["benchmarks"].get("optional_fmm", True):
        return {"fmm_status": "disabled", "fmm_seconds": None}
    if importlib.util.find_spec("skfmm") is None:
        return {"fmm_status": "unavailable", "fmm_seconds": None}
    from ..fsm3d import solve_fmm
    try:
        # Warm the compiled extension outside the timed production solve.
        solve_fmm(np.full((3, 3, 3), 2.), [[0, 2]]*3, [1, 1, 1])
        started = time.perf_counter()
        volume = sample_velocity_grid(velocity, config["domain"]["bounds"], n)
        field, _ = solve_fmm(volume, config["domain"]["bounds"], source)
        seconds = time.perf_counter()-started
        return {"fmm_status": "completed", "fmm_seconds": seconds,
                **{"fmm_"+key: value for key, value in error_metrics(field, reference).items()}}
    except ValueError as exc:
        return {"fmm_status": "skipped", "fmm_seconds": None, "fmm_reason": str(exc)}
    except (RuntimeError, OSError) as exc:
        return {"fmm_status": "failed", "fmm_seconds": None, "fmm_reason": str(exc)}


def benchmark_worker(config: Config, seed: int) -> Dict[str, Any]:
    """Run one training process, then reuse its network across all grid sizes."""
    from ..train import _coarse_fsm, timed_grid_inference, train_one
    from .plotting import plot_fields

    options = config["benchmarks"]
    folder = project_path(config["output_dir"], output=True)
    result_path = folder / "worker_results.json"
    records: List[Dict[str, Any]] = []
    breaks: List[Dict[str, Any]] = []
    result = None
    context: Dict[str, Any] = {"stage": "training"}
    write_json(result_path, {"status": "in_progress", "run_spec_hash": options["run_spec_hash"],
                            "records": [], "breaks": [], "context": context})
    try:
        if options.get("source_hash") != source_hash():
            raise RuntimeError("Project source changed after planning this run; regenerate the benchmark plan")
        result = train_one(config, seed, folder)
        cases = evaluation_sources(config)
        for n in sorted(options["grid_sizes"]):
            source_rows = []
            for index, source_record in enumerate(cases):
                context = {"stage": "grid_evaluation", "grid_n": n, "source_id": source_record["source_id"]}
                source = source_record["source"]
                row = {**result.metrics, **source_record, "experiment": options["experiment"],
                       "mode": options["mode"], "run_id": options["run_id"], "seed": seed,
                       "grid_n": n, "grid_nodes": n**3, "grid_shape": [n, n, n],
                       "device": str(result.device), "status": "completed"}
                if n != config["evaluation"]["grid_size"] or index != 0:
                    prediction, inference = timed_grid_inference(result.model, source, result.velocity,
                        result.device, config, n, options.get("inference_repeats", 3))
                    reference = make_reference(result.velocity, config["domain"]["bounds"], source, n, config)
                    exact = reference.grid(n)
                    fsm, baseline = _coarse_fsm(result.velocity, source, config, n)
                    row.update(inference)
                    row.update(baseline)
                    row.update(error_metrics(prediction, exact))
                    row.update({"fsm_"+key: value for key, value in error_metrics(fsm, exact).items()})
                    row["reference"] = reference.metadata
                    row["final_reference_seconds"] = reference.generation_seconds
                    row.update(_optional_fmm(result.velocity, source, config, n, exact))
                    if options["representative"] and options.get("save_fields", True):
                        np.savez_compressed(str(folder / ("field_%d_%s.npz" % (n, source_record["source_id"]))),
                            prediction=prediction, reference=exact, fsm=fsm,
                            source=source, bounds=config["domain"]["bounds"], reference_kind=reference.kind)
                    if options["representative"]:
                        plot_fields(prediction, exact, fsm, config["domain"]["bounds"], source,
                                    folder / "figures" / ("%d_%s" % (n, source_record["source_id"])), reference.kind)
                    del prediction, exact, fsm, reference
                elif options.get("optional_fmm", True):
                    reference = make_reference(result.velocity, config["domain"]["bounds"], source, n, config)
                    exact = reference.grid(n)
                    row.update(_optional_fmm(result.velocity, source, config, n, exact))
                    del reference, exact
                row["pinn_total_seconds"] = row["surrogate_setup_training_seconds"]+row["inference_seconds"]
                records.append(row)
                source_rows.append(row)
                write_json(result_path, {"status": "in_progress", "run_spec_hash": options["run_spec_hash"],
                                         "records": records, "breaks": breaks, "context": context})
                print("%s grid=%d source=%s PINN=%.3es FSM=%.3es relL2=%.3e" % (
                    options["run_id"], n, source_record["source_id"], row["inference_seconds"],
                    row["fsm_seconds"], row["relative_l2"]), flush=True)
            if options["mode"] == "surrogate":
                training = result.metrics["surrogate_setup_training_seconds"]
                inference = float(np.mean([row["inference_seconds"] for row in source_rows]))
                fsm_cost = float(np.mean([row["fsm_seconds"] for row in source_rows]))
                target = options.get("accuracy_target", .01)
                qualifies = all(row["relative_l2"] <= target and row["fsm_relative_l2"] <= target for row in source_rows)
                crossing = break_even_sources(training, inference, fsm_cost)
                breaks.append({"run_id": options["run_id"], "experiment": options["experiment"],
                    "seed": seed, "grid_n": n, "batch_size": config["training"]["batch_size"],
                    "collocation_budget": result.metrics["collocation_budget"],
                    "precision": config["runtime"]["precision"], "device": str(result.device),
                    "training_seconds": training, "inference_seconds": inference, "fsm_seconds": fsm_cost,
                    "cost_break_even_sources": crossing, "break_even_sources": crossing if qualifies else None,
                    "accuracy_qualified": qualifies, "accuracy_target": target,
                    "worst_held_out_relative_l2": max(r["relative_l2"] for r in source_rows),
                    "config": config, "hardware": result.metrics["hardware"]})
        payload = {"status": "completed", "run_spec_hash": options["run_spec_hash"],
                   "records": records, "breaks": breaks}
        write_json(result_path, payload)
        return payload
    except Exception as exc:
        failure = {"status": "failed", "seed": seed, "run_id": options["run_id"],
                   "experiment": options["experiment"], "mode": options["mode"],
                   "precision": config["runtime"]["precision"], "device": config["runtime"]["device"],
                   "batch_size": config["training"]["batch_size"],
                   "collocation_budget": config["training"]["iterations"]*config["training"]["batch_size"],
                   "error": type(exc).__name__+": "+str(exc), "failed_context": context,
                   "config": config, "hardware": result.metrics["hardware"] if result is not None else hardware_metadata()}
        records.append(failure)
        payload = {"status": "failed", "run_spec_hash": options["run_spec_hash"],
                   "records": records, "breaks": breaks, "error": failure["error"]}
        write_json(result_path, payload)
        print(failure["error"], file=sys.stderr, flush=True)
        return payload


def read_worker_result(path: Any, config: Config, seed: int, return_code: int) -> Dict[str, Any]:
    """Preserve valid partial rows, and reject stale or incomplete worker data."""
    filename = project_path(path)
    expected = config["benchmarks"]["run_spec_hash"]
    payload = None
    if filename.is_file():
        try:
            candidate = json.loads(filename.read_text(encoding="utf-8"))
            if candidate.get("run_spec_hash") == expected:
                payload = candidate
        except (OSError, json.JSONDecodeError):
            pass
    if payload is not None and payload.get("status") == "completed" and return_code == 0:
        return payload
    if payload is not None and payload.get("status") == "failed":
        return payload
    records = payload.get("records", []) if payload is not None else []
    options = config["benchmarks"]
    failure = {"status": "failed", "seed": seed, "run_id": options["run_id"],
               "experiment": options["experiment"], "mode": options["mode"],
               "precision": config["runtime"]["precision"], "device": config["runtime"]["device"],
               "batch_size": config["training"]["batch_size"],
               "collocation_budget": config["training"]["iterations"]*config["training"]["batch_size"],
               "error": "Worker exited %d without a complete matching result" % return_code,
               "failed_context": payload.get("context") if payload is not None else None,
               "config": config, "hardware": records[0].get("hardware", {}) if records else hardware_metadata()}
    return {"status": "failed", "run_spec_hash": expected,
            "records": [*records, failure], "breaks": payload.get("breaks", []) if payload is not None else []}


def run_benchmarks(config: Config) -> List[Dict[str, Any]]:
    """Execute the matrix, persist each result, and make seed-aggregate plots."""
    from ..train import _require_torch
    from ..fsm3d import NUMBA_AVAILABLE
    jobs = suite_plan(config)
    _require_torch()
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for fair baseline timings")
    from .plotting import plot_benchmark_records, plot_break_even
    from .report import benchmark_report
    root = project_path(config["output_dir"], output=True)
    options = config["benchmarks"]
    host_identity = hardware_fingerprint(hardware_metadata())
    records, breaks = [], []
    for index, job in enumerate(jobs):
        effective = job["config"]
        folder = project_path(effective["output_dir"], output=True)
        filename = root / "configs" / (job["run_id"]+".yaml")
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        result_path = folder / "worker_results.json"
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            previous = None
        reuse = bool(options.get("reuse_completed", True) and previous and previous["status"] == "completed"
                     and previous["run_spec_hash"] == effective["benchmarks"]["run_spec_hash"]
                     and previous.get("records") and hardware_fingerprint(previous["records"][0].get("hardware", {})) == host_identity)
        print("Benchmark %d/%d: %s%s" % (index+1, len(jobs), job["run_id"], " (reusing completed run)" if reuse else ""), flush=True)
        if reuse:
            payload = previous
        elif options.get("memory_isolation", True):
            write_json(result_path, {"status": "in_progress", "records": [], "breaks": [],
                                    "run_spec_hash": effective["benchmarks"]["run_spec_hash"]})
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(filename),
                       "--seed", str(job["seed"])]
            process = subprocess.run(command, env=environment, check=False)
            payload = read_worker_result(result_path, effective, job["seed"], process.returncode)
        else:
            payload = benchmark_worker(load_config(filename), job["seed"])
        records.extend(payload["records"])
        breaks.extend(payload["breaks"])
        write_csv(root / "runs.csv", records)
        write_json(root / "runs.json", records)
        write_csv(root / "break_even.csv", breaks)
        write_json(root / "break_even.json", breaks)
        if payload["status"] != "completed" and options.get("fail_fast", False):
            benchmark_report(records, breaks, root)
            raise RuntimeError("Benchmark stopped after failed worker; partial results are saved")
    benchmark_report(records, breaks, root)
    plot_benchmark_records(records, root / "figures")
    if breaks:
        # A separate figure for each comparable setting, never pooling budgets.
        groups = sorted({(r["experiment"], r["precision"], r["device"], r["batch_size"], r["collocation_budget"]) for r in breaks})
        for group in groups:
            selected = [r for r in breaks if (r["experiment"], r["precision"], r["device"], r["batch_size"], r["collocation_budget"]) == group]
            label = "%s_%s_%s_b%d_c%d" % group
            plot_break_even(selected, root / "figures" / label)
    return records


def regenerate_report(config: Config) -> None:
    """Rebuild report/aggregate figures from saved measurements, without training."""
    from .plotting import plot_benchmark_records, plot_break_even
    from .report import benchmark_report

    root = project_path(config["output_dir"], output=True)
    records = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    break_path = root / "break_even.json"
    breaks = json.loads(break_path.read_text(encoding="utf-8")) if break_path.is_file() else []
    benchmark_report(records, breaks, root)
    plot_benchmark_records(records, root / "figures")
    groups = {(r["experiment"], r["precision"], r["device"], r["batch_size"], r["collocation_budget"]) for r in breaks}
    for group in sorted(groups):
        selected = [r for r in breaks if (r["experiment"], r["precision"], r["device"], r["batch_size"], r["collocation_budget"]) == group]
        plot_break_even(selected, root / "figures" / ("%s_%s_%s_b%d_c%d" % group))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.worker:
            payload = benchmark_worker(config, args.seed)
            return 0 if payload["status"] == "completed" else 2
        if args.dry_run:
            jobs = suite_plan(config)
            print(json.dumps({"training_runs": len(jobs), "seeds_per_setting": len(config["seeds"]),
                "grid_sizes": config["benchmarks"]["grid_sizes"], "fresh_processes": config["benchmarks"].get("memory_isolation", True),
                "runs": [{"run_id": j["run_id"], "seed": j["seed"],
                          "iterations": j["config"]["training"]["iterations"],
                          "batch_size": j["config"]["training"]["batch_size"]} for j in jobs]}, indent=2))
            return 0
        if args.report_only:
            regenerate_report(config)
        else:
            run_benchmarks(config)
        return 0
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        print("PINNeik3D benchmark: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
