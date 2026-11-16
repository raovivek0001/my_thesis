"""YAML-driven mini-batch Adam and fixed-set L-BFGS training.

Run from pinneik3d/: ``python train.py --config configs/smoke.yaml``.
Imports of PyTorch are delayed so --help and --dry-run work without it.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pinneik3d"

import numpy as np
import yaml

from .common import (Config, DEFAULTS, deep_merge, config_hash, configure_runtime, load_config, project_path,
                     seed_everything, source_hash, validate_config, write_csv, write_json)
from .benchmarks.measurements import MemoryMonitor, hardware_metadata, synchronize
from .benchmarks.metrics import error_metrics, first_threshold_times
from .grids import grid_batches, grid_shape, sample_velocity_grid
from .models import VelocityModel, build_velocity
from .reference import Reference, make_reference


@dataclass
class TrainingResult:
    """Trained model, device, measured records, and reproducibility artifacts."""

    model: Any
    device: Any
    velocity: VelocityModel
    config: Config
    seed: int
    run_dir: Path
    history: List[Dict[str, Any]]
    metrics: Dict[str, Any]


def _require_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is not installed. Activate pinneik3d/.venv and install requirements.txt.")


def autocast_context(config: Config, device: Any) -> Any:
    """AMP affects network arithmetic; residuals remain float32."""
    import torch

    if config["runtime"]["precision"] == "float32":
        return contextlib.nullcontext()
    name = config["runtime"]["amp_dtype"]
    if name == "auto":
        name = "float16" if device.type == "cuda" else "bfloat16"
    if device.type == "cpu" and name == "float16":
        raise ValueError("Use bfloat16 or auto for CPU mixed precision")
    if device.type == "cuda" and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; use float16 or auto")
    dtype = torch.float16 if name == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _predict_chunks(model: Any, chunks: Any, source: Sequence[float], velocity: VelocityModel,
                    device: Any, config: Config, output: np.ndarray) -> None:
    import torch
    from .losses import background_factor

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        source_tensor = torch.tensor(source, dtype=torch.float32, device=device).reshape(1, 3)
        source_velocity = velocity.torch(source_tensor).float()
        for start, stop, coordinates in chunks:
            points = torch.from_numpy(coordinates).to(device=device, dtype=torch.float32)
            sources = source_tensor.expand_as(points)
            with autocast_context(config, device):
                tau = model(points, sources)
            t0, _ = background_factor(points, sources, source_velocity, config["loss"]["eps"])
            output[start:stop] = (t0*tau.float()).squeeze(-1).cpu().numpy()
    model.train(was_training)


def predict_points(model: Any, points: np.ndarray, source: Sequence[float], velocity: VelocityModel,
                   device: Any, config: Config) -> np.ndarray:
    """Evaluate traveltime at arbitrary points with bounded device memory."""
    coordinates = np.asarray(points).reshape(-1, 3)
    batch_size = config["evaluation"]["batch_size"]
    chunks = ((start, min(start+batch_size, len(coordinates)),
               coordinates[start:start+batch_size].astype(np.float32, copy=False))
              for start in range(0, len(coordinates), batch_size))
    output = np.empty(len(coordinates), dtype=np.float32)
    _predict_chunks(model, chunks, source, velocity, device, config, output)
    return output.reshape(np.asarray(points).shape[:-1])


def predict_grid(model: Any, source: Sequence[float], velocity: VelocityModel,
                 device: Any, config: Config, size: Any = None) -> np.ndarray:
    """Full-grid xyz inference; coordinate generation and output transfers are included."""
    size = config["evaluation"]["grid_size"] if size is None else size
    shape = grid_shape(size)
    output = np.empty(int(np.prod(shape)), dtype=np.float32)
    chunks = grid_batches(config["domain"]["bounds"], size, config["evaluation"]["batch_size"])
    _predict_chunks(model, chunks, source, velocity, device, config, output)
    return output.reshape(shape)


def timed_grid_inference(model: Any, source: Sequence[float], velocity: VelocityModel,
                         device: Any, config: Config, size: Any = None,
                         repeats: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Warm up, then synchronize and measure complete gridded evaluations."""
    if repeats < 1:
        raise ValueError("Inference repeats must be positive")
    # Warmup includes activation kernels and volume source interpolation.
    warm_points = np.tile(np.asarray(source, dtype=np.float32), (64, 1))
    predict_points(model, warm_points, source, velocity, device, config)
    synchronize(device)
    monitor = MemoryMonitor(device).start()
    timings: List[float] = []
    prediction = None
    try:
        for _ in range(repeats):
            prediction = None
            synchronize(device)
            started = time.perf_counter()
            prediction = predict_grid(model, source, velocity, device, config, size)
            synchronize(device)
            timings.append(time.perf_counter()-started)
            monitor.sample()
    finally:
        memory = monitor.stop()
    assert prediction is not None
    return prediction, {"inference_seconds": float(np.mean(timings)),
                        "inference_repeat_std_seconds": float(np.std(timings, ddof=1)) if repeats > 1 else None,
                        "inference_repeats": repeats, "inference_repeat_seconds": timings,
                        **{"inference_"+key: value for key, value in memory.items()}}


def _sample_collocation(bounds: Any, lattice: Any, batch_size: int, source_batch_size: int,
                         generator: Any, device: Any) -> Tuple[Any, Any, Any]:
    """Resample physical locations and lattice source indices every Adam step."""
    from .sampling import sample_uniform
    return sample_uniform(bounds, lattice, batch_size, source_batch_size, generator, device)


def _validation_cases(velocity: VelocityModel, config: Config) -> List[Tuple[np.ndarray, Reference, np.ndarray, np.ndarray]]:
    """Fixed validation grids; surrogate thresholds require every held-out case."""
    from .surrogate import evaluation_sources

    size = config["evaluation"]["validation_grid_size"]
    total = int(np.prod(grid_shape(size)))
    coordinates = np.concatenate([p for _, _, p in grid_batches(config["domain"]["bounds"], size,
                                                                 min(total, 262144), np.float64)])
    cases = []
    for source_record in evaluation_sources(config):
        source = np.asarray(source_record["source"], dtype=np.float64)
        reference = make_reference(velocity, config["domain"]["bounds"], source, size, config)
        cases.append((source, reference, coordinates, reference.evaluate(coordinates)))
    return cases


def _evaluate_validation(model: Any, velocity: VelocityModel, device: Any, config: Config,
                          cases: Sequence[Any]) -> Dict[str, Any]:
    scores = []
    for source, reference, points, exact in cases:
        prediction = predict_points(model, points, source, velocity, device, config)
        scores.append({"source": source.tolist(), **error_metrics(prediction, exact)})
    return {"validation_relative_l2": max(row["relative_l2"] for row in scores),
            "validation_mean_relative_l2": float(np.mean([row["relative_l2"] for row in scores])),
            "validation_source_errors": scores}


def save_checkpoint(path: Any, model: Any, config: Config, seed: int, stage: str,
                    iteration: int, optimizer: Any, scheduler: Any, scaler: Any,
                    balancer: Any, generator: Any, history: List[Dict[str, Any]],
                    elapsed_seconds: float, extra: Optional[Dict[str, Any]] = None) -> None:
    """Atomic checkpoint containing weights, optimizer, schedule, and every RNG."""
    import random
    import torch

    filename = project_path(path, output=True)
    filename.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format_version": 1, "model": model.state_dict(), "config": config,
                  "seed": seed, "stage": stage, "iteration": iteration,
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict() if scheduler is not None else None,
                  "scaler": scaler.state_dict() if scaler is not None else None,
                  "loss_weights": balancer.state_dict(), "history": history,
                  "elapsed_seconds": elapsed_seconds, "collocation_rng": generator.get_state(),
                  "torch_rng": torch.get_rng_state(),
                  "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                  "extra": extra or {}}
    temporary = filename.with_suffix(filename.suffix+".tmp")
    torch.save(checkpoint, str(temporary))
    temporary.replace(filename)


def load_model_checkpoint(path: Any, device: Optional[Any] = None) -> Tuple[Any, Config, Any]:
    """Load a project's checkpoint for evaluation or transfer learning."""
    _require_torch()
    import torch
    from .networks import build_network

    checkpoint = torch.load(str(project_path(path)), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = checkpoint["config"]
    device = configure_runtime(config) if device is None else device
    model = build_network(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, config, device


def _restore_rng(checkpoint: Dict[str, Any], generator: Any) -> None:
    import random
    import torch

    generator.set_state(checkpoint["collocation_rng"].cpu())
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if checkpoint.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])


def _check_checkpoint_compatibility(checkpoint: Dict[str, Any], config: Config,
                                     resume: bool, seed: int) -> None:
    previous = deep_merge(DEFAULTS, checkpoint["config"])
    if resume and checkpoint.get("checkpoint_kind") == "evaluation":
        raise ValueError("best_model.pt contains selected weights only; resume from checkpoint.pt")
    if previous["network"] != config["network"] or previous["domain"]["bounds"] != config["domain"]["bounds"]:
        raise ValueError("Checkpoint architecture and coordinate bounds must match")
    if previous["surrogate"]["enabled"] != config["surrogate"]["enabled"]:
        raise ValueError("Cannot load a 3-input checkpoint into a 6-input network or vice versa")
    if resume:
        if checkpoint["seed"] != seed:
            raise ValueError("A resumed run must use the checkpoint's seed")
        for section in ("domain", "velocity", "loss", "surrogate", "runtime", "evaluation"):
            if previous[section] != config[section]:
                raise ValueError("Resume requires identical " + section + "; use initial_checkpoint for transfer")
        excluded = {"resume_checkpoint", "initial_checkpoint", "checkpoint_every", "log_every"}
        if {k: v for k, v in previous["training"].items() if k not in excluded} != {
            k: v for k, v in config["training"].items() if k not in excluded
        }:
            raise ValueError("Resume requires the same training budgets and schedule")


def _coarse_fsm(velocity: VelocityModel, source: Sequence[float], config: Config,
                 size: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    from .fsm3d import solve_fsm, warmup_fsm

    settings = config["fsm"]
    if settings["backend"] == "numba":
        warmup_fsm()
    monitor = MemoryMonitor().start()
    started = time.perf_counter()
    try:
        volume = sample_velocity_grid(velocity, config["domain"]["bounds"], size)
        velocity_seconds = time.perf_counter()-started
        result = solve_fsm(volume, config["domain"]["bounds"], source,
                           float(velocity.numpy(np.asarray(source))),
                           tolerance=settings["tolerance"], max_cycles=settings["max_cycles"],
                           backend=settings["backend"])
        seconds = time.perf_counter()-started
    finally:
        memory = monitor.stop()
    valid_timing = settings["backend"] == "numba"
    return result.traveltime, {"fsm_seconds": seconds if valid_timing else None,
                              "fsm_solver_seconds": result.seconds if valid_timing else None,
                              "fsm_velocity_sampling_seconds": velocity_seconds if valid_timing else None,
                              "fsm_backend": result.backend, "fsm_device": "cpu", "fsm_num_threads": 1,
                              "fsm_arithmetic": "float64", "fsm_timing_valid": valid_timing,
                              "fsm_cycles": result.cycles, "fsm_final_change": result.final_change,
                              "fsm_converged": result.converged, "fsm_source_injection": result.source_injection,
                              **{"fsm_"+key: value for key, value in memory.items()}}


def train_one(config: Config, seed: int, run_dir: Optional[Any] = None) -> TrainingResult:
    """Train one reproducible run, evaluate the full grid, and write artifacts.

    training_seconds includes resampling, optimization, online validation,
    and periodic checkpoint I/O. Reference construction and initial model
    setup are separately measured. Final inference, FSM, and plotting are
    outside training_seconds; end_to_end_seconds includes them.
    """
    _require_torch()
    import torch
    from .fsm3d import NUMBA_AVAILABLE
    from .losses import GradientLossBalancer, loss_terms
    from .networks import build_network
    from .surrogate import training_sources
    from .sampling import ResidualSampler

    config = copy.deepcopy(config)
    validate_config(config)
    if config["fsm"]["backend"] == "numba" and not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not installed; install requirements.txt before running this experiment")
    overall_started = time.perf_counter()
    seed_everything(seed, config["runtime"]["deterministic"])
    device = configure_runtime(config)
    # Validate AMP availability before creating long-running references.
    with autocast_context(config, device):
        pass
    folder = project_path(run_dir if run_dir is not None else
                          project_path(config["output_dir"]) / ("seed_%d" % seed), output=True)
    folder.mkdir(parents=True, exist_ok=True)
    model = build_network(config).to(device)
    velocity = build_velocity(config["velocity"], config["domain"]["bounds"])
    lattice = torch.tensor(training_sources(config), dtype=torch.float32)
    bounds_tensor = torch.tensor(config["domain"]["bounds"], dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed((seed+1000003) % 2**32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    settings = config["training"]
    sampler = ResidualSampler(bounds_tensor, lattice, settings["batch_size"],
                              settings["source_batch_size"], generator, device,
                              settings["sampling"], config["loss"]["eps"])
    if settings["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["iterations"],
                                                                eta_min=settings["lr_min"])
    elif settings["schedule"] == "exponential":
        gamma = (settings["lr_min"]/settings["lr"])**(1.0/settings["iterations"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None
    amp_name = config["runtime"]["amp_dtype"]
    use_scaler = config["runtime"]["precision"] == "mixed" and device.type == "cuda" and amp_name != "bfloat16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    balancer = GradientLossBalancer(config["loss"]["weights"], **config["loss"]["adaptive"])
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None
    best_validation: Dict[str, Any] = {}
    checkpoint = None
    start_iteration, elapsed_prior = 0, 0.0
    warm_start = settings.get("initial_checkpoint")
    resume = settings.get("resume_checkpoint")
    if warm_start or resume:
        checkpoint = torch.load(str(project_path(warm_start or resume)), map_location="cpu", weights_only=False)
        _check_checkpoint_compatibility(checkpoint, config, bool(resume), seed)
        model.load_state_dict(checkpoint["model"], strict=True)
        if resume:
            history = copy.deepcopy(checkpoint["history"])
            elapsed_prior = float(checkpoint["elapsed_seconds"])
            balancer.load_state_dict(checkpoint["loss_weights"])
            _restore_rng(checkpoint, generator)
            if "sampling" in checkpoint.get("extra", {}):
                sampler.load_state_dict(checkpoint["extra"]["sampling"])
            best_state = checkpoint.get("extra", {}).get("best_model")
            best_validation = copy.deepcopy(checkpoint.get("extra", {}).get("best_validation", {}))
            if checkpoint["stage"] == "adam":
                start_iteration = int(checkpoint["iteration"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                scaler.load_state_dict(checkpoint["scaler"])
            else:
                start_iteration = settings["iterations"]
    synchronize(device)
    setup_seconds = time.perf_counter()-overall_started
    reference_started = time.perf_counter()
    cases = _validation_cases(velocity, config)
    reference_seconds = time.perf_counter()-reference_started
    metadata = {"seed": seed, "config_hash": config_hash(config), "source_hash": source_hash(), "config": config,
                "hardware": hardware_metadata(device), "train_source_count": len(lattice),
                "train_sources": lattice.numpy().tolist(),
                "validation_sources": [case[0].tolist() for case in cases],
                "validation_references": [case[1].metadata for case in cases],
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "setup_seconds": setup_seconds, "validation_reference_seconds": reference_seconds,
                "initial_checkpoint": warm_start, "resume_checkpoint": resume,
                "timing_definition": "training includes validation and periodic checkpoints; excludes reference construction and final evaluation",
                "precision": config["runtime"]["precision"]}
    write_json(folder / "metadata.json", metadata)
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monitor = MemoryMonitor(device).start()
    training_started = time.perf_counter()
    parameters = tuple(model.parameters())
    closure_evaluations = int(checkpoint.get("extra", {}).get("closure_evaluations", 0)) if resume else 0
    skipped_adam_steps = int(checkpoint.get("extra", {}).get("adam_steps_skipped", 0)) if resume else 0

    def elapsed() -> float:
        synchronize(device)
        return elapsed_prior+time.perf_counter()-training_started

    def record(stage: str, iteration: int, terms: Optional[Dict[str, Any]],
               evaluate: bool, lr: float) -> None:
        nonlocal best_state, best_validation
        row: Dict[str, Any] = {"seed": seed, "stage": stage, "iteration": iteration,
                               "learning_rate": lr, "loss_weights": balancer.state_dict(),
                               "adam_steps_skipped": skipped_adam_steps}
        if terms is not None:
            row.update({"loss_"+name: term.detach().float().item() for name, term in terms.items()})
            row["loss_total"] = balancer.total(terms).detach().float().item()
        if evaluate:
            row.update(_evaluate_validation(model, velocity, device, config, cases))
        row["elapsed_seconds"] = elapsed()
        if settings["select_best"] and evaluate and np.isfinite(row["validation_relative_l2"]):
            if row["validation_relative_l2"] < best_validation.get("relative_l2", float("inf")):
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_validation = {"relative_l2": row["validation_relative_l2"], "stage": stage,
                                   "iteration": iteration, "elapsed_seconds": row["elapsed_seconds"]}
        history.append(row)
        print("seed=%d stage=%s step=%d time=%.2fs loss=%s relL2=%s" % (
            seed, stage, iteration, row["elapsed_seconds"],
            "%.3e" % row["loss_total"] if "loss_total" in row else "--",
            "%.3e" % row["validation_relative_l2"] if "validation_relative_l2" in row else "--"), flush=True)

    def checkpoint_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
        return {**extra, "sampling": sampler.state_dict(),
                "best_model": best_state, "best_validation": best_validation}

    def physics_terms(points: Any, sources: Any, boundaries: Any) -> Dict[str, Any]:
        return loss_terms(model, points, sources, boundaries, velocity, config["loss"]["eps"],
                          residual_normalization=config["loss"]["residual_normalization"])

    active_optimizer, active_scheduler, active_scaler = optimizer, scheduler, scaler
    final_iteration = start_iteration
    lbfgs_extra: Dict[str, Any] = {}
    try:
        if not history:
            record("initial", 0, None, True, settings["lr"])
        for iteration in range(start_iteration+1, settings["iterations"]+1):
            model.train()
            points, sources, boundaries = sampler.sample(model, velocity, iteration)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                terms = physics_terms(points, sources, boundaries)
            if balancer.enabled and iteration % balancer.update_every == 0:
                balancer.update(terms, parameters)
            total = balancer.total(terms)
            if not bool(torch.isfinite(total.detach())):
                raise FloatingPointError("Nonfinite loss; reduce learning rate or use float32")
            scaler.scale(total).backward(inputs=parameters)
            if settings["gradient_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(settings["gradient_clip"]))
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = not use_scaler or scaler.get_scale() >= scale_before_step
            if not step_succeeded:
                skipped_adam_steps += 1
            if scheduler is not None and step_succeeded:
                scheduler.step()
            monitor.sample()
            final_iteration = iteration
            evaluate = iteration % settings["eval_every"] == 0 or iteration == settings["iterations"]
            if iteration % settings["log_every"] == 0 or evaluate:
                record("adam", iteration, terms, evaluate, optimizer.param_groups[0]["lr"])
            if iteration % settings["checkpoint_every"] == 0:
                checkpoint_path = folder / ("checkpoint_iter_%06d.pt" % iteration)
                save_checkpoint(checkpoint_path, model, config, seed, "adam", iteration, optimizer,
                                 scheduler, scaler, balancer, generator, history, elapsed(),
                                 checkpoint_extra({"adam_steps_skipped": skipped_adam_steps, "run_metadata": metadata}))
                save_checkpoint(folder / "checkpoint.pt", model, config, seed, "adam", iteration,
                                 optimizer, scheduler, scaler, balancer, generator, history, elapsed(),
                                 checkpoint_extra({"adam_steps_skipped": skipped_adam_steps, "run_metadata": metadata}))
        lbfgs_settings = settings["lbfgs"]
        completed_resume = bool(resume and checkpoint["stage"] == "complete")
        if lbfgs_settings["enabled"] and not completed_resume:
            if resume and checkpoint["stage"] == "lbfgs":
                extra = checkpoint["extra"]
                fixed_points = extra["fixed_points"].to(device).requires_grad_(True)
                fixed_sources = extra["fixed_sources"].to(device)
                fixed_boundaries = extra["fixed_boundaries"].to(device)
                lbfgs_start = int(checkpoint["iteration"])
            else:
                fixed_points, fixed_sources, _ = _sample_collocation(bounds_tensor, lattice,
                    lbfgs_settings["batch_size"], len(lattice), generator, device)
                fixed_boundaries = lattice.to(device)
                lbfgs_start = 0
            lbfgs = torch.optim.LBFGS(parameters, lr=lbfgs_settings["lr"],
                max_iter=lbfgs_settings["max_iter"], history_size=lbfgs_settings["history_size"],
                tolerance_grad=lbfgs_settings["tolerance_grad"],
                tolerance_change=lbfgs_settings["tolerance_change"], line_search_fn="strong_wolfe")
            if resume and checkpoint["stage"] == "lbfgs":
                lbfgs.load_state_dict(checkpoint["optimizer"])
            active_optimizer, active_scheduler, active_scaler = lbfgs, None, None
            lbfgs_extra = {"fixed_points": fixed_points.detach().cpu(),
                           "fixed_sources": fixed_sources.detach().cpu(),
                           "fixed_boundaries": fixed_boundaries.detach().cpu(), "run_metadata": metadata}

            def closure() -> Any:
                nonlocal closure_evaluations
                lbfgs.zero_grad(set_to_none=True)
                # Fixed collocations, fixed weights, and float32 are essential
                # for a deterministic, stationary L-BFGS line-search objective.
                terms = physics_terms(fixed_points, fixed_sources, fixed_boundaries)
                total = balancer.total(terms)
                if not bool(torch.isfinite(total.detach())):
                    raise FloatingPointError("L-BFGS produced a nonfinite objective")
                total.backward(inputs=parameters)
                closure_evaluations += 1
                monitor.sample()
                return total

            for iteration in range(lbfgs_start+1, lbfgs_settings["iterations"]+1):
                lbfgs.step(closure)
                final_iteration = iteration
                terms = physics_terms(fixed_points, fixed_sources, fixed_boundaries)
                record("lbfgs", iteration, terms, True, lbfgs_settings["lr"])
                lbfgs_extra["closure_evaluations"] = closure_evaluations
                lbfgs_extra["adam_steps_skipped"] = skipped_adam_steps
                save_checkpoint(folder / "checkpoint.pt", model, config, seed, "lbfgs", iteration,
                    lbfgs, None, None, balancer, generator, history, elapsed(), checkpoint_extra(lbfgs_extra))
        training_seconds = elapsed()
    finally:
        memory = monitor.stop()
    save_checkpoint(folder / "checkpoint.pt", model, config, seed, "complete", final_iteration,
        active_optimizer, active_scheduler, active_scaler, balancer, generator, history,
        training_seconds, checkpoint_extra({**lbfgs_extra, "closure_evaluations": closure_evaluations,
                           "adam_steps_skipped": skipped_adam_steps, "run_metadata": metadata}))
    if settings["select_best"] and best_state is not None:
        # Keep checkpoint.pt's last optimizer/weights together for exact resume.
        # Export the selected weights separately for inference and transfer.
        model.load_state_dict(best_state)
        selected_checkpoint = {"format_version": 1, "checkpoint_kind": "evaluation",
                               "model": best_state, "config": config, "seed": seed,
                               "selection": best_validation}
        selected_path = folder / "best_model.pt"
        temporary = selected_path.with_suffix(".pt.tmp")
        torch.save(selected_checkpoint, str(temporary))
        temporary.replace(selected_path)
    write_csv(folder / "history.csv", history)
    write_json(folder / "history.json", {"metadata": metadata, "history": history})
    source = cases[0][0]
    size = config["evaluation"]["grid_size"]
    prediction, inference = timed_grid_inference(model, source, velocity, device, config, size,
                                                  config["evaluation"].get("timing_repeats", 3))
    final_reference = make_reference(velocity, config["domain"]["bounds"], source, size, config)
    exact = final_reference.grid(size)
    fsm_field, fsm_metrics = _coarse_fsm(velocity, source, config, size)
    scores = error_metrics(prediction, exact)
    fsm_scores = {"fsm_"+key: value for key, value in error_metrics(fsm_field, exact).items()}
    metrics = {"name": config["name"], "seed": seed, "status": "completed", "source": source.tolist(),
               "grid_shape": list(grid_shape(size)), "grid_nodes": int(np.prod(grid_shape(size))),
               "batch_size": settings["batch_size"], "adam_iterations": settings["iterations"],
               "collocation_budget": settings["iterations"]*settings["batch_size"],
               "candidate_points_scored": sampler.candidate_points_scored,
               "model_selection": "best_validation" if settings["select_best"] else "last_iteration",
               "selected_validation": best_validation if settings["select_best"] else None,
               "adam_steps_skipped": skipped_adam_steps,
               "lbfgs_closure_evaluations": closure_evaluations,
               "training_seconds": training_seconds, "setup_seconds": setup_seconds,
               "surrogate_setup_training_seconds": setup_seconds+training_seconds,
               "validation_reference_seconds": reference_seconds,
               "final_reference_seconds": final_reference.generation_seconds,
               "validation_grid_shape": list(grid_shape(config["evaluation"]["validation_grid_size"])),
               "validation_eval_every": settings["eval_every"],
               "threshold_definition": "first observed validation crossing; surrogate uses worst held-out source",
               "final_loss_weights": balancer.state_dict(), "reference": final_reference.metadata,
               "config_hash": metadata["config_hash"], "source_hash": metadata["source_hash"],
               "config": config, "hardware": metadata["hardware"],
               "parameter_count": metadata["parameter_count"], "train_source_count": len(lattice),
               "precision": config["runtime"]["precision"],
               **first_threshold_times(history, config["evaluation"]["thresholds"]),
               **memory, **inference, **fsm_metrics, **scores, **fsm_scores}
    if config["evaluation"]["save_field"]:
        np.savez_compressed(str(folder / "field.npz"), prediction=prediction, reference=exact, fsm=fsm_field,
                            source=source, bounds=np.asarray(config["domain"]["bounds"]),
                            reference_kind=final_reference.kind)
    if config["evaluation"]["plots"]:
        from .benchmarks.plotting import plot_fields, plot_history
        plot_fields(prediction, exact, fsm_field, config["domain"]["bounds"], source,
                    folder / "figures", reference_label=final_reference.kind)
        plot_history(history, folder / "figures")
    metrics["end_to_end_seconds"] = time.perf_counter()-overall_started
    write_json(folder / "metrics.json", metrics)
    write_csv(folder / "metrics.csv", [metrics])
    return TrainingResult(model, device, velocity, config, seed, folder, history, metrics)


def run_training(config: Config) -> List[Dict[str, Any]]:
    """Execute all configured seeds and report mean/std of actual runs."""
    from .benchmarks.report import training_report

    rows = []
    for seed in config["seeds"]:
        result = train_one(config, seed)
        rows.append(result.metrics)
        del result
    folder = project_path(config["output_dir"], output=True)
    write_csv(folder / "runs.csv", rows)
    write_json(folder / "runs.json", rows)
    training_report(rows, folder)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI supporting YAML experiments, diagnostic overrides, and resuming."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/const_grad.yaml")
    parser.add_argument("--seed", type=int, help="Single-seed diagnostic override; full experiments use >=3 seeds")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", choices=["float32", "mixed"])
    parser.add_argument("--output-dir", help="Override output directory inside pinneik3d/")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", help="Full-state checkpoint with matching YAML settings")
    parser.add_argument("--initial", help="Initialize weights only for transfer learning")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print effective config without PyTorch/Numba")
    args = parser.parse_args(argv)
    overrides: Config = {"runtime": {}, "training": {}}
    for key in ("device", "precision"):
        if getattr(args, key) is not None:
            overrides["runtime"][key] = getattr(args, key)
    for key in ("iterations", "batch_size"):
        if getattr(args, key) is not None:
            overrides["training"][key] = getattr(args, key)
    if args.seed is not None:
        overrides["seeds"] = [args.seed]
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.resume:
        overrides["training"]["resume_checkpoint"] = args.resume
    if args.initial:
        overrides["training"]["initial_checkpoint"] = args.initial
    try:
        config = load_config(args.config, overrides)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return 0
        run_training(config)
        return 0
    except (RuntimeError, ValueError, ImportError, OSError, FloatingPointError) as exc:
        print("PINNeik3D: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
