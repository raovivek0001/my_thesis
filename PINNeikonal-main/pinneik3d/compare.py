"""Measure baseline versus enhanced accuracy with matched optimization budgets.

Run ``python -m pinneik3d.compare --smoke`` from the repository root.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pinneik3d"

import numpy as np

from .common import Config, load_config, project_path, validate_config, write_csv, write_json


def comparison_configs(baseline_path: Any, enhanced_path: Any) -> Tuple[Config, Config]:
    """Use identical physics, seeds, precision, grids and optimizer budgets.

    Sampling, network, loss normalization, clipping and model selection may
    differ. Candidate-scoring overhead is measured, not hidden in the budget.
    """
    baseline, enhanced = load_config(baseline_path), load_config(enhanced_path)
    for section in ("domain", "velocity", "surrogate"):
        if baseline[section] != enhanced[section]:
            raise ValueError("Comparison requires identical " + section)
    if len(enhanced["seeds"]) < 3:
        raise ValueError("An accuracy comparison requires at least three seeds")
    for section in ("seeds", "runtime", "evaluation", "fsm"):
        baseline[section] = copy.deepcopy(enhanced[section])
    for key in ("iterations", "batch_size", "source_batch_size", "lr", "lr_min", "schedule",
                "log_every", "eval_every", "checkpoint_every", "lbfgs"):
        baseline["training"][key] = copy.deepcopy(enhanced["training"][key])
    for config in (baseline, enhanced):
        if config["training"]["initial_checkpoint"] or config["training"]["resume_checkpoint"]:
            raise ValueError("Comparison requires fresh initialization for both variants")
        validate_config(config)
    return baseline, enhanced


def summarize_pairs(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize measured paired seeds without promising an improvement."""
    grouped = {variant: {row["seed"]: row for row in rows if row["variant"] == variant}
               for variant in ("baseline", "enhanced")}
    seeds = sorted(grouped["baseline"])
    if len(seeds) < 3 or seeds != sorted(grouped["enhanced"]):
        raise ValueError("At least three matching baseline/enhanced seeds are required")
    if len(rows) != 2*len(seeds):
        raise ValueError("Duplicate seed measurements are not allowed")
    fields = ("relative_l2", "max_abs_error", "mean_abs_error", "training_seconds",
              "inference_seconds", "candidate_points_scored")
    result: Dict[str, Any] = {"seed_count": len(seeds), "seeds": seeds, "variants": {}}
    for variant, values in grouped.items():
        statistics = {}
        for field in fields:
            data = np.asarray([values[seed][field] for seed in seeds], dtype=np.float64)
            if not np.all(np.isfinite(data)):
                raise ValueError("Nonfinite measured " + field)
            statistics[field] = {"mean": float(data.mean()), "std": float(data.std(ddof=1))}
        result["variants"][variant] = statistics
    base_error = result["variants"]["baseline"]["relative_l2"]["mean"]
    enhanced_error = result["variants"]["enhanced"]["relative_l2"]["mean"]
    result["accuracy_improved"] = enhanced_error < base_error
    result["relative_l2_reduction_percent"] = 100*(base_error-enhanced_error)/base_error if base_error else None
    result["seeds_with_lower_error"] = sum(grouped["enhanced"][seed]["relative_l2"] <
                                          grouped["baseline"][seed]["relative_l2"] for seed in seeds)
    return result


def write_comparison_report(folder: Path, summary: Dict[str, Any]) -> None:
    """Write a compact table of actual errors and costs."""
    lines = ["# Baseline and enhanced PINN comparison", "",
             "Identical physics, seeds, precision, evaluation grids, Adam updates/batches and L-BFGS budgets.",
             "Adaptive candidate scoring adds work; all of its runtime is included in training time.", "",
             "| Variant | Relative L2 | Max error (s) | Mean error (s) | Training (s) | Inference (s) | Scored candidates |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, statistics in summary["variants"].items():
        values = ["%.6g ± %.3g" % (statistics[field]["mean"], statistics[field]["std"])
                  for field in ("relative_l2", "max_abs_error", "mean_abs_error", "training_seconds",
                                "inference_seconds", "candidate_points_scored")]
        lines.append("| " + variant + " | " + " | ".join(values) + " |")
    reduction = summary["relative_l2_reduction_percent"]
    lines.extend(["", "Lower mean relative L2: **%s**; lower error in **%d/%d** paired seeds." % (
        "yes" if summary["accuracy_improved"] else "no", summary["seeds_with_lower_error"], summary["seed_count"]),
        "Mean relative L2 reduction: %s." % ("%.2f%%" % reduction if reduction is not None else "undefined"),
        "", "Best-weight selection uses the validation grid; reported errors use the separate final grid.",
        "This compares two 3D implementations. The original 2D notebooks solve different problems."])
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="configs/const_grad.yaml")
    parser.add_argument("--enhanced", default="configs/enhanced.yaml")
    parser.add_argument("--smoke", action="store_true", help="Use small baseline/enhanced configs")
    parser.add_argument("--output-dir", default="results/accuracy_comparison")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        baseline, enhanced = comparison_configs(
            "configs/smoke.yaml" if args.smoke else args.baseline,
            "configs/enhanced_smoke.yaml" if args.smoke else args.enhanced)
        folder = project_path(args.output_dir, output=True)
        if args.dry_run:
            print(json.dumps({"training_runs": 2*len(enhanced["seeds"]),
                              "baseline": baseline, "enhanced": enhanced}, indent=2))
            return 0
        from .train import train_one
        rows: List[Dict[str, Any]] = []
        for index, seed in enumerate(enhanced["seeds"]):
            variants = [("baseline", baseline), ("enhanced", enhanced)]
            if index % 2:
                variants.reverse()
            for variant, config in variants:
                result = train_one(config, seed, folder / variant / ("seed_%d" % seed))
                rows.append({"variant": variant, **result.metrics})
                del result
                write_json(folder / "runs.json", rows)
                write_csv(folder / "runs.csv", rows)
        summary = summarize_pairs(rows)
        write_json(folder / "comparison.json", summary)
        write_comparison_report(folder, summary)
        print("Measured comparison: " + str(folder / "summary.md"))
        return 0
    except (RuntimeError, ValueError, ImportError, OSError, FloatingPointError) as exc:
        print("PINNeik3D comparison: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
