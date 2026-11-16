"""Matched-seed source/velocity transfer against random initialization."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pinneik3d"

from .common import Config, deep_merge, load_config, project_path, write_csv, write_json
from .benchmarks.metrics import aggregate_records, threshold_key


def run_transfer(config: Config) -> List[Dict[str, Any]]:
    """Pretrain/reuse weights, then compare the same target with matched batches.

    Collocation randomness uses a separate, identically seeded generator in
    each variant. Both variants use the same optimizer, schedule, architecture,
    precision, iterations, and target medium. Only initial weights differ.
    Pretraining cost is recorded separately from target convergence time.
    """
    from .train import _require_torch, train_one
    from .benchmarks.plotting import plot_transfer
    from .benchmarks.report import training_report
    _require_torch()
    import torch
    if len(config["seeds"]) < 3:
        raise ValueError("Transfer experiments require at least three seeds")
    if config["surrogate"]["enabled"]:
        raise ValueError("Transfer examples use a single-source network; use initial_checkpoint for custom surrogate transfer")
    options = config.get("transfer", {})
    base_config = load_config(options.get("base_config", "configs/const_grad.yaml"))
    if base_config["network"] != config["network"] or base_config["domain"]["bounds"] != config["domain"]["bounds"]:
        raise ValueError("Transfer and pretraining need identical network architecture and domain bounds")
    changes = options.get("changes", [])
    if not changes or len({change["name"] for change in changes}) != len(changes):
        raise ValueError("Provide distinct named transfer changes")
    for change in changes:
        if set(change)-{"name", "domain", "velocity"}:
            raise ValueError("Transfer changes may only change domain.source and velocity")
        if set(change.get("domain", {})) - {"source"}:
            raise ValueError("Transfer domain changes may only change source")
    root = project_path(config["output_dir"], output=True)
    records, histories, pretraining = [], [], []
    for seed in config["seeds"]:
        pretrained_dir = options.get("pretrained_dir")
        candidate_root = project_path(pretrained_dir) if pretrained_dir else project_path(base_config["output_dir"])
        candidate = candidate_root / ("seed_%d" % seed) / "checkpoint.pt"
        if candidate.is_file() and options.get("reuse_pretrained", True):
            state = torch.load(str(candidate), map_location="cpu", weights_only=False)
            previous = state["config"]
            for section in ("domain", "velocity", "network", "surrogate"):
                if previous[section] != base_config[section]:
                    raise ValueError("Existing pretraining checkpoint has a different " + section)
            if state["seed"] != seed or state["stage"] != "complete":
                raise ValueError("Pretrained weights must be completed and have the matching seed")
            pretraining_seconds = float(state["elapsed_seconds"])
            pretraining.append({"seed": seed, "checkpoint": str(candidate), "reused": True,
                                "pretraining_seconds": pretraining_seconds, "config": previous,
                                "hardware": state.get("extra", {}).get("run_metadata", {}).get("hardware", {})})
        else:
            if pretrained_dir:
                raise FileNotFoundError("Requested pretraining checkpoint not found: " + str(candidate))
            base_run = train_one(base_config, seed, root / "base" / ("seed_%d" % seed))
            candidate = base_run.run_dir / "checkpoint.pt"
            pretraining_seconds = base_run.metrics["training_seconds"]
            pretraining.append({"seed": seed, "checkpoint": str(candidate), "reused": False,
                                "pretraining_seconds": pretraining_seconds,
                                "config": base_run.config, "hardware": base_run.metrics["hardware"]})
            del base_run
        for change in changes:
            target = deep_merge(config, {k: v for k, v in change.items() if k != "name"})
            for variant in ("random", "transfer"):
                variant_config = copy.deepcopy(target)
                variant_config["name"] = "transfer_%s_%s" % (change["name"], variant)
                variant_config["training"]["initial_checkpoint"] = str(candidate) if variant == "transfer" else None
                variant_config["training"]["resume_checkpoint"] = None
                result = train_one(variant_config, seed,
                    root / change["name"] / variant / ("seed_%d" % seed))
                records.append({**result.metrics, "change": change["name"], "variant": variant,
                                "pretraining_seconds": pretraining_seconds,
                                "pretraining_checkpoint": str(candidate)})
                histories.append({"seed": seed, "change": change["name"], "variant": variant,
                                  "history": result.history})
                del result
    comparisons = []
    for seed in config["seeds"]:
        for change in changes:
            random_row = next(r for r in records if r["seed"] == seed and r["change"] == change["name"] and r["variant"] == "random")
            transfer_row = next(r for r in records if r["seed"] == seed and r["change"] == change["name"] and r["variant"] == "transfer")
            row = {"seed": seed, "change": change["name"], "config": config,
                   "hardware": transfer_row["hardware"], "pretraining_seconds": transfer_row["pretraining_seconds"]}
            for threshold in config["evaluation"]["thresholds"]:
                key = threshold_key(threshold)
                a, b = random_row[key], transfer_row[key]
                row["random_"+key], row["transfer_"+key] = a, b
                row["speedup_"+format(threshold, ".0e")] = a/b if a is not None and b is not None and b > 0 else None
            comparisons.append(row)
    write_csv(root / "runs.csv", records)
    write_json(root / "runs.json", records)
    write_json(root / "pretraining.json", pretraining)
    write_csv(root / "comparison.csv", comparisons)
    write_json(root / "comparison.json", comparisons)
    speed_keys = ["speedup_"+format(t, ".0e") for t in config["evaluation"]["thresholds"]]
    comparison_aggregate = aggregate_records(comparisons, ["change"], speed_keys)
    write_csv(root / "comparison_aggregate.csv", comparison_aggregate)
    write_json(root / "comparison_aggregate.json", comparison_aggregate)
    training_report(records, root)
    note = ("\nTransfer comparisons use matched seeds and identical resampled collocations. "
            "Target convergence excludes pretraining cost; pretraining.json records that cost. "
            "comparison.csv/json contain speedups only when both variants reached the target, "
            "and comparison_aggregate.csv/json report successful counts.\n")
    summary = root / "summary.md"
    text = summary.read_text(encoding="utf-8")+note
    summary.write_text(text, encoding="utf-8")
    project_path("results/summary.md", output=True).write_text(text, encoding="utf-8")
    plot_transfer(histories, root / "figures")
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transfer.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dry_run:
            base = load_config(config["transfer"]["base_config"])
            print(json.dumps({"config": config, "base_config": base}, indent=2))
            return 0
        run_transfer(config)
        return 0
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        print("PINNeik3D transfer: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
