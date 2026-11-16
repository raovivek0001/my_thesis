"""Publication plots saved as standalone PNG and PDF artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pinneik3d.benchmarks"

import numpy as np

from ..common import project_path

# Keep matplotlib's font/config cache within the authorized project directory.
os.environ["MPLCONFIGDIR"] = str(project_path("results/.matplotlib", output=True))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": False, "savefig.dpi": 250})


def _save(figure: Any, folder: Any, name: str) -> List[str]:
    root = project_path(folder, output=True)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        filename = root / (name+"."+suffix)
        figure.savefig(str(filename), bbox_inches="tight")
        paths.append(str(filename))
    plt.close(figure)
    return paths


def source_slice(field: np.ndarray, bounds: Sequence[Sequence[float]],
                  source: Sequence[float], normal_axis: int) -> np.ndarray:
    """Interpolate a slice through the actual source, including off-grid sources."""
    index = (source[normal_axis]-bounds[normal_axis][0]) / (
        bounds[normal_axis][1]-bounds[normal_axis][0])*(field.shape[normal_axis]-1)
    lower = max(0, min(int(np.floor(index)), field.shape[normal_axis]-1))
    upper = min(lower+1, field.shape[normal_axis]-1)
    fraction = max(0., min(1., index-lower))
    return (1-fraction)*np.take(field, lower, axis=normal_axis)+fraction*np.take(field, upper, axis=normal_axis)


def plot_fields(prediction: Optional[np.ndarray], reference: np.ndarray, fsm: Optional[np.ndarray],
                 bounds: Sequence[Sequence[float]], source: Sequence[float], folder: Any,
                 reference_label: str = "analytic") -> List[str]:
    """Absolute xy/xz/yz error slices and overlaid traveltime contours."""
    methods = []
    if prediction is not None:
        methods.append(("PINN", prediction, "tab:red", "--"))
    if fsm is not None:
        methods.append(("FSM", fsm, "tab:blue", ":"))
    if not methods:
        raise ValueError("At least one computed field is required")
    planes = [(2, 0, 1, "xy"), (1, 0, 2, "xz"), (0, 1, 2, "yz")]
    figure, axes = plt.subplots(len(methods), 3, figsize=(11, 3.4*len(methods)),
                                squeeze=False, constrained_layout=True)
    maximum = max(float(np.max(np.abs(field-reference))) for _, field, _, _ in methods)
    maximum = max(maximum, 1e-14)
    for row, (label, field, _, _) in enumerate(methods):
        for column, (normal, first, second, name) in enumerate(planes):
            error = np.abs(source_slice(field-reference, bounds, source, normal))
            axis = axes[row, column]
            im = axis.imshow(error.T, origin="lower", extent=[*bounds[first], *bounds[second]],
                             cmap="magma", vmin=0., vmax=maximum, aspect="equal")
            axis.plot(source[first], source[second], "+", color="cyan", markersize=8)
            axis.set(title=label+" absolute error: "+name, xlabel="xyz"[first]+" (km)",
                     ylabel="xyz"[second]+" (km)")
    figure.colorbar(im, ax=axes.ravel().tolist(), label="Absolute error (s)", shrink=.85)
    artifacts = _save(figure, folder, "error_slices")
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    levels = np.linspace(0., float(np.max(reference)), 12)[1:-1]
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="black", label=reference_label)]
    handles.extend(Line2D([0], [0], color=color, linestyle=style, label=label)
                   for label, _, color, style in methods)
    for axis, (normal, first, second, name) in zip(axes, planes):
        first_nodes = np.linspace(*bounds[first], reference.shape[first])
        second_nodes = np.linspace(*bounds[second], reference.shape[second])
        axis.contour(first_nodes, second_nodes, source_slice(reference, bounds, source, normal).T,
                     levels=levels, colors="black", linewidths=.9)
        for label, field, color, style in methods:
            axis.contour(first_nodes, second_nodes, source_slice(field, bounds, source, normal).T,
                         levels=levels, colors=color, linestyles=style, linewidths=.9)
        axis.plot(source[first], source[second], "k+", markersize=8)
        axis.set(title=name+" traveltime contours", xlabel="xyz"[first]+" (km)", ylabel="xyz"[second]+" (km)")
        axis.set_aspect("equal")
    axes[-1].legend(handles=handles, fontsize=8)
    artifacts.extend(_save(figure, folder, "traveltime_contours"))
    return artifacts


def plot_history(history: Sequence[Mapping[str, Any]], folder: Any) -> List[str]:
    """Loss curves and online validation accuracy against measured wall time."""
    artifacts = []
    losses = [row for row in history if row.get("loss_total") is not None]
    if losses:
        figure, axis = plt.subplots(figsize=(6, 3.5))
        times = [row["elapsed_seconds"] for row in losses]
        for name in ("total", "pde", "positivity", "source"):
            values = [row.get("loss_"+name, np.nan) for row in losses]
            values = [value if value is not None and value > 0 else np.nan for value in values]
            axis.semilogy(times, values, label=name)
        axis.set(xlabel="Training wall time (s)", ylabel="Loss", title="Factored eikonal training")
        axis.legend()
        artifacts.extend(_save(figure, folder, "loss_curves"))
    validation = [row for row in history if row.get("validation_relative_l2") is not None]
    if validation:
        figure, axis = plt.subplots(figsize=(6, 3.5))
        axis.semilogy([row["elapsed_seconds"] for row in validation],
                      [max(row["validation_relative_l2"], 1e-16) for row in validation], "o-", markersize=3)
        for target in (.01, .001):
            axis.axhline(target, color="gray", linestyle=":", linewidth=.8)
        axis.set(xlabel="Training wall time (s)", ylabel="Validation relative L2",
                 title="Time to accuracy (worst source for surrogate)")
        artifacts.extend(_save(figure, folder, "time_vs_accuracy"))
    return artifacts


def plot_break_even(records: Sequence[Mapping[str, Any]], folder: Any) -> List[str]:
    """Training plus per-source inference versus repeated FSM solves."""
    if not records:
        return []
    grids = sorted({row["grid_n"] for row in records})
    figure, axes = plt.subplots(1, len(grids), figsize=(5*len(grids), 3.8), squeeze=False)
    for axis, grid in zip(axes.ravel(), grids):
        rows = [row for row in records if row["grid_n"] == grid]
        counts = [row["cost_break_even_sources"] for row in rows if row.get("cost_break_even_sources") is not None]
        maximum = max(100., 2.*max(counts)) if counts else 1000.
        sources = np.unique(np.maximum(1, np.logspace(0, np.log10(maximum), 300).astype(np.int64)))
        surrogate = np.asarray([r["training_seconds"]+sources*r["inference_seconds"] for r in rows])
        sweeping = np.asarray([sources*r["fsm_seconds"] for r in rows])
        for label, costs, color in (("Surrogate", surrogate, "tab:red"), ("FSM", sweeping, "tab:blue")):
            mean = costs.mean(axis=0)
            axis.loglog(sources, mean, label=label, color=color)
            if len(rows) >= 3:
                deviation = costs.std(axis=0, ddof=1)
                axis.fill_between(sources, np.maximum(mean-deviation, 1e-12), mean+deviation,
                                  color=color, alpha=.18)
        qualifies = all(row.get("accuracy_qualified", False) for row in rows)
        title = "%s^3 grid" % grid
        title += "\nAccuracy condition met" if qualifies else "\nCost comparison; accuracy condition unmet"
        axis.set(title=title, xlabel="Number of queried sources", ylabel="Total wall time (s)")
        axis.legend()
    return _save(figure, folder, "break_even")


def plot_benchmark_records(records: Sequence[Mapping[str, Any]], folder: Any) -> List[str]:
    """Plot model/device/precision groups separately, using seed replicates."""
    valid = [row for row in records if row.get("status") == "completed"]
    if not valid:
        return []
    artifacts = []
    identities = list(dict.fromkeys((r["experiment"], r["precision"], r["device"]) for r in valid))
    for index, identity in enumerate(identities):
        selected = [r for r in valid if (r["experiment"], r["precision"], r["device"]) == identity]
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", "%s_%s_%s" % identity)
        target = project_path(folder, output=True) if index == 0 else project_path(folder, output=True) / name
        artifacts.extend(_plot_benchmark_group(selected, target, " / ".join(identity)))
    return artifacts


def _plot_grid_scaling(records: Sequence[Mapping[str, Any]], folder: Any, title: str) -> List[str]:
    """Grid scaling for one model/precision/device/batch/collocation budget."""
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    sources = sorted({(r["mode"], r["source_id"]) for r in records})
    for mode, source in sources:
        rows = [r for r in records if r["mode"] == mode and r["source_id"] == source]
        grids = sorted({r["grid_n"] for r in rows})
        for key, method, style in (("inference_seconds", "PINN", "-"), ("fsm_seconds", "FSM", "--")):
            means, deviations = [], []
            for n in grids:
                values = [r[key] for r in rows if r["grid_n"] == n and r.get(key) is not None]
                means.append(float(np.mean(values)) if values else np.nan)
                deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.)
            axis.errorbar([n**3 for n in grids], means, yerr=deviations, marker="o",
                          linestyle=style, markersize=3, capsize=2,
                          label="%s %s/%s" % (method, mode, source))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(xlabel="Grid nodes", ylabel="Full-grid wall time (s)", title=title)
    axis.legend(fontsize=7)
    return _save(figure, folder, "grid_scaling")


def _plot_benchmark_group(valid: Sequence[Mapping[str, Any]], folder: Any, title: str) -> List[str]:
    """Plot family for a single velocity preset, precision, and device."""
    artifacts = []
    # Training cost/memory are reused across grids: count each seed/run once.
    unique = {}
    for row in valid:
        unique.setdefault(row["run_id"], row)
    training_rows = list(unique.values())
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    groups = sorted({(r["mode"], r["collocation_budget"]) for r in training_rows})
    for group in groups:
        rows = [r for r in training_rows if (r["mode"], r["collocation_budget"]) == group]
        batches = sorted({r["batch_size"] for r in rows})
        label = "%s C=%d" % group
        for axis, key in zip(axes, ("cpu_peak_rss_bytes", "gpu_peak_allocated_bytes")):
            means, deviations = [], []
            for batch in batches:
                values = [r[key]/2**20 for r in rows if r["batch_size"] == batch]
                means.append(np.mean(values))
                deviations.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            axis.errorbar(batches, means, yerr=deviations, marker="o", capsize=3, label=label)
    for axis, panel_title in zip(axes, ("Observed peak CPU RSS", "Peak allocated GPU memory")):
        axis.set(xlabel="Collocation batch size", ylabel="Memory (MiB)", title=panel_title)
        axis.set_xscale("log")
    axes[0].legend(fontsize=7)
    figure.suptitle(title)
    artifacts.extend(_save(figure, folder, "memory_vs_batch_size"))
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for label, time_key, error_key, marker in (("PINN: training + one inference", "pinn_total_seconds", "relative_l2", "o"),
                                               ("FSM", "fsm_seconds", "fsm_relative_l2", "x")):
        times = [r[time_key] for r in valid if r.get(time_key) is not None]
        errors = [max(r[error_key], 1e-16) for r in valid if r.get(time_key) is not None]
        axis.loglog(times, errors, marker, linestyle="none", alpha=.6, label=label)
    axis.set(xlabel="Wall time for one source (s)", ylabel="Full-grid relative L2", title=title)
    axis.legend(fontsize=8)
    artifacts.extend(_save(figure, folder, "time_vs_accuracy"))
    batches = sorted({r["batch_size"] for r in valid})
    budgets = sorted({r["collocation_budget"] for r in valid})
    for batch in batches:
        for budget in budgets:
            selected = [r for r in valid if r["batch_size"] == batch and r["collocation_budget"] == budget]
            if selected:
                target = project_path(folder, output=True) if batch == batches[0] and budget == budgets[-1] else (
                    project_path(folder, output=True) / ("grid_b%d_c%d" % (batch, budget)))
                artifacts.extend(_plot_grid_scaling(selected, target, title+"\nb=%d, C=%d" % (batch, budget)))
    return artifacts


def plot_transfer(histories: Sequence[Mapping[str, Any]], folder: Any) -> List[str]:
    """Actual convergence curves for transferred and random initial weights."""
    changes = sorted({row["change"] for row in histories})
    figure, axes = plt.subplots(1, len(changes), figsize=(5.3*len(changes), 3.7), squeeze=False)
    for axis, change in zip(axes.ravel(), changes):
        for variant, color in (("random", "tab:blue"), ("transfer", "tab:red")):
            curves = [row for row in histories if row["change"] == change and row["variant"] == variant]
            for index, curve in enumerate(curves):
                values = [row for row in curve["history"] if row.get("validation_relative_l2") is not None]
                axis.semilogy([r["elapsed_seconds"] for r in values],
                              [max(r["validation_relative_l2"], 1e-16) for r in values],
                              color=color, alpha=.6, label=variant if index == 0 else None)
        axis.axhline(.01, color="gray", linestyle=":")
        axis.axhline(.001, color="gray", linestyle=":")
        axis.set(title=change, xlabel="Target training wall time (s)", ylabel="Validation relative L2")
        axis.legend()
    return _save(figure, folder, "transfer_convergence")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Regenerate source slices/contours and loss plots from saved run artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", help="Saved field.npz with reference, prediction/FSM, source, bounds")
    parser.add_argument("--history", help="Saved history.json")
    parser.add_argument("--output", required=True, help="Output figure folder inside pinneik3d/")
    args = parser.parse_args(argv)
    if not args.fields and not args.history:
        parser.error("Provide --fields and/or --history")
    try:
        if args.fields:
            with np.load(str(project_path(args.fields)), allow_pickle=False) as fields:
                label = str(fields["reference_kind"].item()) if "reference_kind" in fields else "reference"
                plot_fields(fields["prediction"] if "prediction" in fields else None,
                            fields["reference"], fields["fsm"] if "fsm" in fields else None,
                            fields["bounds"], fields["source"], args.output, label)
        if args.history:
            payload = json.loads(project_path(args.history).read_text(encoding="utf-8"))
            plot_history(payload["history"], args.output)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print("PINNeik3D plotting: "+str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
