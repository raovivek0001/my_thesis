"""Six-input source surrogates with disjoint held-out and distant sources."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pinneik3d"

import numpy as np
from scipy.spatial import cKDTree

from .common import Config, load_config, project_path, write_csv, write_json
from .benchmarks.metrics import break_even_sources, error_metrics
from .reference import make_reference


def training_sources(config: Config) -> np.ndarray:
    """Return a regular source lattice, default 4*4*4=64, in physical km."""
    if not config["surrogate"]["enabled"]:
        return np.asarray([config["domain"]["source"]], dtype=np.float64)
    options = config["surrogate"]
    counts = options["source_lattice"]
    bounds = np.asarray(options["source_bounds"], dtype=np.float64)
    domain = np.asarray(config["domain"]["bounds"], dtype=np.float64)
    if len(counts) != 3 or any(not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("source_lattice must contain three positive integers")
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("Invalid source_bounds")
    if np.any(bounds[:, 0] < domain[:, 0]) or np.any(bounds[:, 1] > domain[:, 1]):
        raise ValueError("Training sources must lie inside the physical domain")
    if any(n > 1 and a == b for n, (a, b) in zip(counts, bounds)):
        raise ValueError("A source axis with multiple nodes requires a nonzero interval")
    axes = [np.linspace(a, b, n) if n > 1 else np.asarray([(a+b)/2])
            for n, (a, b) in zip(counts, bounds)]
    return np.asarray(list(itertools.product(*axes)), dtype=np.float64)


def evaluation_sources(config: Config) -> List[Dict[str, Any]]:
    """Held-out sources plus a source maximally distant from the regular lattice.

    For a Cartesian lattice, nearest-source distance separates by coordinate.
    The largest empty point occurs at a domain boundary or an axis midpoint
    between adjacent training coordinates; testing their products finds it
    exactly. The distant case may test extrapolation outside the source box.
    """
    lattice = training_sources(config)
    if not config["surrogate"]["enabled"]:
        return [{"source_id": "single", "source": lattice[0].tolist(), "kind": "single",
                 "nearest_training_source_km": 0.0, "outside_training_source_box": False}]
    domain = np.asarray(config["domain"]["bounds"], dtype=np.float64)
    source_bounds = np.asarray(config["surrogate"]["source_bounds"], dtype=np.float64)
    held_out = config["surrogate"]["held_out"]
    if not held_out:
        raise ValueError("Surrogate training requires at least one held-out source")
    records = []
    tree = cKDTree(lattice)
    for index, source in enumerate(held_out):
        point = np.asarray(source, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)) or np.any(point < domain[:, 0]) or np.any(point > domain[:, 1]):
            raise ValueError("Each held-out source must be inside the physical domain")
        distance = float(tree.query(point)[0])
        if distance < 1e-8:
            raise ValueError("Held-out sources must not occur in the training lattice")
        if any(np.linalg.norm(point-np.asarray(r["source"])) < 1e-8 for r in records):
            raise ValueError("Held-out sources must be distinct")
        records.append({"source_id": "held_out_%d" % index, "source": point.tolist(),
                        "kind": "held_out", "nearest_training_source_km": distance,
                        "outside_training_source_box": bool(np.any(point < source_bounds[:, 0]) or
                                                             np.any(point > source_bounds[:, 1]))})
    far = config["surrogate"]["far_source"]
    if isinstance(far, str):
        if far != "auto":
            raise ValueError("far_source must be auto or three coordinates")
        candidates = []
        for axis in range(3):
            coordinates = np.unique(lattice[:, axis])
            midpoints = (coordinates[:-1]+coordinates[1:])/2
            candidates.append(np.concatenate((domain[axis], midpoints)))
        points = np.asarray(list(itertools.product(*candidates)))
        distances = tree.query(points)[0]
        far_point = points[np.argmax(distances)]
    else:
        far_point = np.asarray(far, dtype=np.float64)
    if far_point.shape != (3,) or not np.all(np.isfinite(far_point)) or np.any(far_point < domain[:, 0]) or np.any(far_point > domain[:, 1]):
        raise ValueError("far_source must lie inside the physical domain")
    distance = float(tree.query(far_point)[0])
    if distance < 1e-8:
        raise ValueError("far_source must not occur in the training lattice")
    far_record = {"source_id": "far", "source": far_point.tolist(), "kind": "far",
                  "nearest_training_source_km": distance,
                  "outside_training_source_box": bool(np.any(far_point < source_bounds[:, 0]) or
                                                       np.any(far_point > source_bounds[:, 1]))}
    duplicate = next((i for i, r in enumerate(records)
                      if np.linalg.norm(far_point-np.asarray(r["source"])) < 1e-8), None)
    if duplicate is None:
        records.append(far_record)
    else:
        records[duplicate] = far_record
    return records


def run_surrogate(config: Config) -> List[Dict[str, Any]]:
    """Train all seeds, evaluate held-out sources, and compute cost break-even."""
    from .train import _coarse_fsm, timed_grid_inference, train_one
    from .benchmarks.plotting import plot_break_even, plot_fields
    from .benchmarks.report import surrogate_report

    config = copy.deepcopy(config)
    config["surrogate"]["enabled"] = True
    if len(config["seeds"]) < 3:
        raise ValueError("Surrogate experiments require at least three seeds")
    cases = evaluation_sources(config)
    records: List[Dict[str, Any]] = []
    break_records = []
    target = float(config.get("benchmarks", {}).get("accuracy_target", .01))
    for seed in config["seeds"]:
        result = train_one(config, seed)
        source_records = []
        for index, source_record in enumerate(cases):
            source = source_record["source"]
            row = {**source_record, "name": config["name"], "seed": seed,
                   "grid_n": config["evaluation"]["grid_size"],
                   "training_seconds": result.metrics["training_seconds"],
                   "surrogate_setup_training_seconds": result.metrics["surrogate_setup_training_seconds"],
                   "batch_size": config["training"]["batch_size"],
                   "collocation_budget": result.metrics["collocation_budget"],
                   "precision": config["runtime"]["precision"], "status": "completed",
                   "config": config, "hardware": result.metrics["hardware"],
                   "train_source_count": len(training_sources(config))}
            if index == 0:
                for key in ("relative_l2", "max_abs_error", "mean_abs_error", "inference_seconds",
                            "fsm_seconds", "fsm_relative_l2", "fsm_max_abs_error", "fsm_mean_abs_error"):
                    row[key] = result.metrics[key]
            else:
                prediction, inference = timed_grid_inference(result.model, source, result.velocity,
                    result.device, config, repeats=config["evaluation"].get("timing_repeats", 3))
                reference = make_reference(result.velocity, config["domain"]["bounds"], source,
                                           config["evaluation"]["grid_size"], config)
                exact = reference.grid(config["evaluation"]["grid_size"])
                fsm, baseline = _coarse_fsm(result.velocity, source, config, config["evaluation"]["grid_size"])
                row.update(inference)
                row.update(baseline)
                row.update(error_metrics(prediction, exact))
                row.update({"fsm_"+k: v for k, v in error_metrics(fsm, exact).items()})
                row["reference"] = reference.metadata
                if config["evaluation"]["save_field"]:
                    np.savez_compressed(str(result.run_dir / (source_record["source_id"]+".npz")),
                        prediction=prediction, reference=exact, fsm=fsm, source=source,
                        bounds=config["domain"]["bounds"], reference_kind=reference.kind)
                if config["evaluation"]["plots"]:
                    plot_fields(prediction, exact, fsm, config["domain"]["bounds"], source,
                                result.run_dir / "figures" / source_record["source_id"], reference.kind)
            source_records.append(row)
            records.append(row)
        if any(row.get("fsm_seconds") is None for row in source_records):
            raise RuntimeError("Break-even needs compiled FSM timings; Python diagnostic times are excluded")
        inference_mean = float(np.mean([row["inference_seconds"] for row in source_records]))
        fsm_mean = float(np.mean([row["fsm_seconds"] for row in source_records]))
        training = result.metrics["surrogate_setup_training_seconds"]
        cost_count = break_even_sources(training, inference_mean, fsm_mean)
        qualified = all(row["relative_l2"] <= target and row["fsm_relative_l2"] <= target for row in source_records)
        break_records.append({"name": config["name"], "seed": seed,
            "grid_n": config["evaluation"]["grid_size"], "training_seconds": training,
            "inference_seconds": inference_mean, "fsm_seconds": fsm_mean,
            "cost_break_even_sources": cost_count, "break_even_sources": cost_count if qualified else None,
            "accuracy_qualified": qualified, "accuracy_target": target,
            "worst_held_out_relative_l2": max(row["relative_l2"] for row in source_records),
            "source_count_measured": len(source_records), "config": config, "hardware": result.metrics["hardware"]})
        write_csv(result.run_dir / "held_out.csv", source_records)
        write_json(result.run_dir / "held_out.json", source_records)
        del result
    folder = project_path(config["output_dir"], output=True)
    write_csv(folder / "held_out.csv", records)
    write_json(folder / "held_out.json", records)
    write_csv(folder / "break_even.csv", break_records)
    write_json(folder / "break_even.json", break_records)
    plot_break_even(break_records, folder / "figures")
    surrogate_report(records, break_records, folder)
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/surrogate.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, {"surrogate": {"enabled": True}})
        if args.dry_run:
            print(json.dumps({"config": config, "training_sources": training_sources(config).tolist(),
                              "evaluation_sources": evaluation_sources(config)}, indent=2))
            return 0
        run_surrogate(config)
        return 0
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        print("PINNeik3D surrogate: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
