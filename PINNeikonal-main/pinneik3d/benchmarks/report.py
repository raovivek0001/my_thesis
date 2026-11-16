"""Generate Markdown tables from measured results; never fill missing runs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ..common import project_path, write_csv, write_json
from .metrics import aggregate_records, threshold_key

STANDARD_METRICS = ["training_seconds", "inference_seconds", "fsm_seconds", "relative_l2",
                    "max_abs_error", "mean_abs_error", "fsm_relative_l2", "cpu_peak_rss_bytes",
                    "gpu_peak_allocated_bytes", threshold_key(.01), threshold_key(.001)]


def _format(row: Mapping[str, Any], key: str, scale: float = 1.0) -> str:
    mean, std = row.get(key+"_mean"), row.get(key+"_std")
    if mean is None:
        return "unreached" if key.startswith("time_to_") else "—"
    return "%.4g ± %.3g" % (mean/scale, std/scale) if std is not None else "%.4g (one value)" % (mean/scale)


def _hardware(records: Sequence[Mapping[str, Any]]) -> str:
    hardware = records[0].get("hardware", {}) if records else {}
    return ("| Hardware/software | Value |\n|---|---|\n"
            "| CPU | %s |\n| GPU | %s |\n| Device used | %s |\n| CUDA | %s |\n| PyTorch | %s |\n| Python | %s |\n" % (
                hardware.get("cpu_model", "unrecorded"),
                hardware.get("gpu_model") or ", ".join(hardware.get("gpu_models_detected", [])) or "none detected",
                hardware.get("device", "unrecorded"), hardware.get("cuda_version") or "unavailable",
                hardware.get("torch_version") or "unavailable", hardware.get("python_version", "unrecorded")))


def _write_summary(folder: Any, text: str, *, global_summary: bool = True) -> None:
    root = project_path(folder, output=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text(text, encoding="utf-8")
    if global_summary:
        destination = project_path("results/summary.md", output=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")


def training_report(records: Sequence[Mapping[str, Any]], folder: Any) -> List[Dict[str, Any]]:
    """Tables for ordinary single-source training, averaged across seeds."""
    aggregate = aggregate_records(records, ["name", "grid_nodes", "batch_size", "precision"], STANDARD_METRICS)
    write_csv(project_path(folder) / "aggregate.csv", aggregate)
    write_json(project_path(folder) / "aggregate.json", aggregate)
    lines = ["# PINNeik3D measured training results", "", _hardware(records), "",
             "Values are mean ± sample standard deviation across independent seeds. Each CSV/JSON run includes its complete effective YAML and hardware metadata.", "",
             "| Experiment | Nodes | Batch | Precision | Seeds | Training (s) | Inference (s) | FSM (s) | Relative L2 | Max error (s) | Mean error (s) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["name"], row["grid_nodes"], row["batch_size"], row["precision"], row["seed_count"],
            *[_format(row, key) for key in ("training_seconds", "inference_seconds", "fsm_seconds",
                                            "relative_l2", "max_abs_error", "mean_abs_error")]))
    lines += ["", "| Experiment | Time to 1e-2 (s), reached | Time to 1e-3 (s), reached | CPU peak RSS (MiB) | GPU peak allocated (MiB) |",
              "|---|---:|---:|---:|---:|"]
    for row in aggregate:
        first, second = threshold_key(.01), threshold_key(.001)
        lines.append("| %s | %s, %d/%d | %s, %d/%d | %s | %s |" % (
            row["name"], _format(row, first), row[first+"_count"], row["seed_count"],
            _format(row, second), row[second+"_count"], row["seed_count"],
            _format(row, "cpu_peak_rss_bytes", 2**20), _format(row, "gpu_peak_allocated_bytes", 2**20)))
    lines += ["", "Training time includes resampling, optimization, validation, and periodic checkpoints; reference construction and final evaluation are separate. Inference and FSM times include full-grid data preparation and exclude warmup/JIT. CPU RSS is an observed sampled peak.", "",
              "Unreached thresholds are right-censored at training completion. Means of threshold times include only successful crossings and must be read with their reached counts. A twice-finer FSM reference has a discretization error floor and does not certify analytic 1e-3 accuracy."]
    if any(not r["three_seed_requirement_met"] for r in aggregate):
        lines += ["", "This is a diagnostic run with fewer than three seeds; the full-study reproducibility requirement has not been met."]
    _write_summary(folder, "\n".join(lines)+"\n")
    return aggregate


def surrogate_report(records: Sequence[Mapping[str, Any]], break_records: Sequence[Mapping[str, Any]],
                      folder: Any) -> List[Dict[str, Any]]:
    """Held-out source errors and accuracy-qualified break-even tables."""
    keys = ["relative_l2", "max_abs_error", "mean_abs_error", "inference_seconds", "fsm_seconds"]
    aggregate = aggregate_records(records, ["name", "grid_n", "source_id", "kind"], keys)
    break_aggregate = aggregate_records(break_records, ["grid_n"],
        ["training_seconds", "cost_break_even_sources", "break_even_sources", "worst_held_out_relative_l2"])
    write_csv(project_path(folder) / "aggregate.csv", aggregate)
    write_json(project_path(folder) / "aggregate.json", aggregate)
    write_json(project_path(folder) / "break_even_aggregate.json", break_aggregate)
    lines = ["# PINNeik3D source surrogate", "", _hardware(records), "",
             "| Source | Type | Grid | Seeds | Relative L2 | Max error (s) | Mean error (s) | Inference (s) | FSM (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append("| %s | %s | %s³ | %d | %s | %s | %s | %s | %s |" % (
            row["source_id"], row["kind"], row["grid_n"], row["seed_count"], *[_format(row, key) for key in keys]))
    lines += ["", "| Grid | Setup + training (s) | Cost-only crossing (sources) | Accuracy-qualified crossing (sources) | Finite qualified crossings |",
              "|---:|---:|---:|---:|---:|"]
    for row in break_aggregate:
        lines.append("| %s³ | %s | %s | %s | %d/%d |" % (row["grid_n"],
            _format(row, "training_seconds"), _format(row, "cost_break_even_sources"),
            _format(row, "break_even_sources"), row["break_even_sources_count"], row["seed_count"]))
    lines += ["", "The far source maximizes nearest-training-source distance over the domain for the regular training lattice. Its distance and whether it lies outside the training source box are in held_out.csv/json.", "",
              "The crossing is the smallest integer K satisfying setup + training + K × inference < K × FSM. No finite crossing exists when per-source inference costs at least as much as FSM. Accuracy qualification requires both methods to meet the configured relative-L2 target for every measured held-out source, including the distant case."]
    _write_summary(folder, "\n".join(lines)+"\n")
    return aggregate


def benchmark_report(records: Sequence[Mapping[str, Any]], breaks: Sequence[Mapping[str, Any]],
                      folder: Any) -> List[Dict[str, Any]]:
    """Scaling tables with per-source groups and actual failure status."""
    valid = [row for row in records if row.get("status") == "completed"]
    groups = ["experiment", "mode", "precision", "device", "grid_n", "batch_size", "collocation_budget", "source_id"]
    aggregate = aggregate_records(valid, groups, STANDARD_METRICS)
    write_csv(project_path(folder) / "aggregate.csv", aggregate)
    write_json(project_path(folder) / "aggregate.json", aggregate)
    lines = ["# PINNeik3D computational cost study", "", _hardware(valid), "",
             "Values are mean ± sample standard deviation over seeds; experiment settings and source cases are grouped separately.", "",
             "| Mode/source | Grid | Batch | Collocations | Precision/device | Seeds | Training (s) | Inference (s) | FSM (s) | PINN relative L2 | FSM relative L2 |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        lines.append("| %s/%s | %s³ | %d | %d | %s/%s | %d | %s | %s | %s | %s | %s |" % (
            row["mode"], row["source_id"], row["grid_n"], row["batch_size"], row["collocation_budget"],
            row["precision"], row["device"], row["seed_count"],
            *[_format(row, key) for key in ("training_seconds", "inference_seconds", "fsm_seconds", "relative_l2", "fsm_relative_l2")]))
    lines += ["", "| Mode/source/grid/batch/budget | Max error (s) | Mean error (s) | CPU RSS (MiB) | GPU allocated (MiB) | Time to 1e-2 (s), reached | Time to 1e-3 (s), reached |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        first, second = threshold_key(.01), threshold_key(.001)
        lines.append("| %s/%s/%s/%s/%s | %s | %s | %s | %s | %s, %d/%d | %s, %d/%d |" % (
            row["mode"], row["source_id"], row["grid_n"], row["batch_size"], row["collocation_budget"],
            _format(row, "max_abs_error"), _format(row, "mean_abs_error"),
            _format(row, "cpu_peak_rss_bytes", 2**20), _format(row, "gpu_peak_allocated_bytes", 2**20),
            _format(row, first), row[first+"_count"], row["seed_count"],
            _format(row, second), row[second+"_count"], row["seed_count"]))
    if breaks:
        break_groups = ["experiment", "precision", "device", "grid_n", "batch_size", "collocation_budget"]
        break_aggregate = aggregate_records(breaks, break_groups, ["cost_break_even_sources", "break_even_sources"])
        write_csv(project_path(folder) / "break_even_aggregate.csv", break_aggregate)
        write_json(project_path(folder) / "break_even_aggregate.json", break_aggregate)
        lines += ["", "| Experiment/grid/batch/budget | Cost crossing (sources) | Accuracy-qualified crossing (sources) | Finite qualified crossings |",
                  "|---|---:|---:|---:|"]
        for row in break_aggregate:
            lines.append("| %s/%s/%s/%s | %s | %s | %d/%d |" % (row["experiment"], row["grid_n"],
                row["batch_size"], row["collocation_budget"], _format(row, "cost_break_even_sources"),
                _format(row, "break_even_sources"), row["break_even_sources_count"], row["seed_count"]))
    failures = [row for row in records if row.get("status") != "completed"]
    lines += ["", "Completed measurement rows: %d. Failed/skipped rows: %d. Failed runs remain in runs.csv/json with their actual reason." % (len(valid), len(failures)), "",
              "Training cost is reused for grid evaluations from the same run; this is an inference/grid scaling study and a separate collocation-budget/batch study. Grid nodes are not full-batch collocations. Each configuration uses at least three seeds.", "",
              "Time to target is the first observed validation crossing; unachieved targets are censored, and successful counts accompany their means. Reference FSM uses twice as many cells per axis (51→101, 101→201, 201→401 nodes). Its first-order discretization error limits claims of true accuracy.", "",
              "CUDA timing includes synchronization; JIT and inference warmup are excluded. Full-grid inference and FSM include coordinate/velocity preparation. CPU RSS is sampled at 10 ms plus explicit samples, and GPU memory is torch.cuda.max_memory_allocated."]
    _write_summary(folder, "\n".join(lines)+"\n")
    return aggregate
