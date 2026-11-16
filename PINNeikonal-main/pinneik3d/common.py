"""Configuration, reproducibility, and output helpers shared by the CLIs."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
Config = Dict[str, Any]

DEFAULTS: Config = {
    "name": "const_grad", "seeds": [42, 43, 44], "output_dir": "results/const_grad",
    "runtime": {"device": "auto", "precision": "float32", "amp_dtype": "auto",
                "deterministic": True, "num_threads": 1},
    "domain": {"bounds": [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], "source": [1.0, 1.0, 1.0]},
    "velocity": {"kind": "constant_gradient", "v0": 2.0, "gradient": [0.0, 0.0, 0.5]},
    "network": {"hidden_layers": 10, "width": 20, "activation": "adaptive_atan",
                "activation_scale": 10.0,
                "fourier": {"enabled": False, "size": 32, "scale": 1.0, "include_raw": True}},
    "loss": {"eps": 1e-8, "residual_normalization": "none",
             "weights": {"pde": 1.0, "positivity": 1.0, "source": 1.0},
             "adaptive": {"enabled": True, "update_every": 50, "beta": 0.9,
                          "min_weight": 1e-3, "max_weight": 1e3}},
    "training": {"iterations": 20000, "batch_size": 2048, "lr": 0.001,
                 "lr_min": 1e-5, "schedule": "cosine", "log_every": 100,
                 "eval_every": 500, "checkpoint_every": 2000, "source_batch_size": 16,
                 "gradient_clip": None, "initial_checkpoint": None, "resume_checkpoint": None,
                 "select_best": False,
                 "sampling": {"enabled": False, "warmup": 250, "update_every": 50,
                              "candidate_multiplier": 4, "uniform_fraction": 0.5},
                 "lbfgs": {"enabled": False, "iterations": 50, "max_iter": 20,
                           "batch_size": 8192, "lr": 1.0, "history_size": 50,
                           "tolerance_grad": 1e-9, "tolerance_change": 1e-12}},
    "evaluation": {"grid_size": 51, "validation_grid_size": 21, "batch_size": 65536,
                   "thresholds": [1e-2, 1e-3], "reference": "auto", "refinement": 2,
                   "save_field": True, "plots": True},
    "fsm": {"tolerance": 1e-8, "max_cycles": 100, "backend": "numba"},
    "surrogate": {"enabled": False, "source_lattice": [4, 4, 4],
                  "source_bounds": [[0.4, 1.6], [0.4, 1.6], [0.4, 1.6]],
                  "held_out": [[1.0, 1.0, 1.0], [0.6, 1.0, 1.4]],
                  "far_source": "auto"},
}


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Config:
    """Merge dictionaries without sharing mutable state."""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def project_path(path: Any, *, output: bool = False) -> Path:
    """Resolve project-relative paths and enforce the project's write boundary."""
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    result = result.resolve()
    if output:
        try:
            result.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError("All outputs must stay inside pinneik3d/: " + str(result)) from exc
    return result


def _read_yaml(path: Path, visiting: Sequence[Path]) -> Config:
    if path in visiting:
        raise ValueError("Cyclic YAML extends chain: " + str(path))
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("A config must contain a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        inherited = _read_yaml((path.parent / parent).resolve(), [*visiting, path])
        config = deep_merge(inherited, config)
    return config


def load_config(path: Any, overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Read YAML, resolve an optional extends chain, merge defaults, validate."""
    filename = Path(path)
    if not filename.is_file():
        filename = project_path(path)
    filename = filename.resolve()
    config = deep_merge(DEFAULTS, _read_yaml(filename, []))
    if overrides:
        config = deep_merge(config, overrides)
    config["config_path"] = str(filename)
    if config["velocity"].get("path"):
        config["velocity"]["path"] = str(project_path(config["velocity"]["path"]))
    for key in ("initial_checkpoint", "resume_checkpoint"):
        if config["training"].get(key):
            config["training"][key] = str(project_path(config["training"][key]))
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Reject malformed domains, budgets, optimizers, or experiment settings."""
    bounds = np.asarray(config["domain"]["bounds"], dtype=float)
    source = np.asarray(config["domain"]["source"], dtype=float)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("domain.bounds must contain three finite increasing pairs")
    if source.shape != (3,) or not np.all(np.isfinite(source)) or np.any(source < bounds[:, 0]) or np.any(source > bounds[:, 1]):
        raise ValueError("domain.source must be inside the 3D domain")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    if any(not isinstance(seed, int) or seed < 0 or seed >= 2**32 for seed in config["seeds"]):
        raise ValueError("Each seed must be an integer in [0, 2**32)")
    for key in ("iterations", "batch_size", "log_every", "eval_every", "checkpoint_every", "source_batch_size"):
        value = config["training"][key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError("training." + key + " must be a positive integer")
    if not np.isfinite(config["training"]["lr"]) or config["training"]["lr"] <= 0 or config["training"]["schedule"] not in ("cosine", "exponential", "none"):
        raise ValueError("Invalid Adam learning rate or schedule")
    minimum_lr = config["training"]["lr_min"]
    if not np.isfinite(minimum_lr) or minimum_lr < 0 or minimum_lr > config["training"]["lr"]:
        raise ValueError("lr_min must be finite and between zero and lr")
    if config["training"]["schedule"] == "exponential" and minimum_lr == 0:
        raise ValueError("Exponential schedule requires a positive lr_min")
    if config["training"].get("initial_checkpoint") and config["training"].get("resume_checkpoint"):
        raise ValueError("Use either initial_checkpoint (weights) or resume_checkpoint (full state)")
    if config["runtime"]["precision"] not in ("float32", "mixed"):
        raise ValueError("runtime.precision must be float32 or mixed")
    if config["runtime"]["amp_dtype"] not in ("auto", "float16", "bfloat16"):
        raise ValueError("runtime.amp_dtype must be auto, float16, or bfloat16")
    if config["runtime"]["num_threads"] <= 0:
        raise ValueError("runtime.num_threads must be positive")
    if config["network"]["hidden_layers"] < 1 or config["network"]["width"] < 1:
        raise ValueError("The MLP requires positive depth and width")
    if config["network"]["activation"] not in ("tanh", "atan", "sin", "adaptive_atan"):
        raise ValueError("Unknown activation")
    if config["network"]["activation_scale"] <= 0:
        raise ValueError("activation_scale must be positive")
    fourier = config["network"]["fourier"]
    if fourier["size"] <= 0 or fourier["scale"] <= 0:
        raise ValueError("Fourier size and scale must be positive")
    adaptive = config["loss"]["adaptive"]
    if not 0 <= adaptive["beta"] < 1 or adaptive["update_every"] <= 0:
        raise ValueError("Adaptive beta must be in [0,1) and update_every positive")
    if adaptive["min_weight"] <= 0 or adaptive["max_weight"] < adaptive["min_weight"]:
        raise ValueError("Invalid adaptive weight bounds")
    if config["loss"]["eps"] <= 0 or any(v < 0 for v in config["loss"]["weights"].values()):
        raise ValueError("Loss epsilon must be positive and weights nonnegative")
    if config["loss"]["residual_normalization"] not in ("none", "relative"):
        raise ValueError("residual_normalization must be none or relative")
    sampling = config["training"]["sampling"]
    for key in ("warmup", "update_every", "candidate_multiplier"):
        value = sampling[key]
        minimum = 0 if key == "warmup" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("Invalid sampling." + key)
    if not np.isfinite(sampling["uniform_fraction"]) or not 0.25 <= sampling["uniform_fraction"] <= 1:
        raise ValueError("sampling.uniform_fraction must be in [0.25, 1] to retain domain coverage")
    if not isinstance(config["training"]["select_best"], bool):
        raise ValueError("training.select_best must be boolean")
    for key in ("grid_size", "validation_grid_size"):
        size = config["evaluation"][key]
        shape = [size] * 3 if isinstance(size, int) else size
        if len(shape) != 3 or any(not isinstance(n, int) or n < 3 for n in shape):
            raise ValueError("Evaluation grids need at least three nodes per axis")
    if config["evaluation"]["refinement"] != 2:
        raise ValueError("Reference refinement must be 2 (twice as many cells per axis)")
    if config["evaluation"]["reference"] not in ("auto", "analytic", "fsm"):
        raise ValueError("reference must be auto, analytic, or fsm")
    if config["evaluation"]["batch_size"] <= 0 or any(t <= 0 for t in config["evaluation"]["thresholds"]):
        raise ValueError("Inference batch size and error thresholds must be positive")
    lbfgs = config["training"]["lbfgs"]
    if any(lbfgs[k] <= 0 for k in ("iterations", "max_iter", "batch_size", "history_size", "lr")):
        raise ValueError("L-BFGS budgets, history size, and learning rate must be positive")
    if config["fsm"]["tolerance"] <= 0 or config["fsm"]["max_cycles"] <= 0:
        raise ValueError("FSM convergence parameters must be positive")
    if config["fsm"]["backend"] not in ("numba", "python"):
        raise ValueError("FSM backend must be numba or diagnostic python")
    project_path(config["output_dir"], output=True)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every generator and disable nondeterministic CUDA shortcuts."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def configure_runtime(config: Config) -> Any:
    """Choose the device and explicitly set CPU threading."""
    import torch

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    device_name = config["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This benchmark supports cpu and cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch installation has no available CUDA device")
    return device


def to_jsonable(value: Any) -> Any:
    """Convert NumPy/path values and replace nonfinite scalars by JSON null."""
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Any, data: Any) -> None:
    """Write strict JSON atomically, always within pinneik3d/."""
    filename = project_path(path, output=True)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(to_jsonable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(filename)


def write_csv(path: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write tabular records; nested metadata are stored as JSON in each row."""
    filename = project_path(path, output=True)
    filename.parent.mkdir(parents=True, exist_ok=True)
    records = [to_jsonable(dict(row)) for row in rows]
    keys: List[str] = []
    for row in records:
        keys.extend(key for key in row if key not in keys)
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def config_hash(config: Config) -> str:
    """Stable identifier for the complete effective configuration."""
    serialized = json.dumps(to_jsonable(config), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def source_hash() -> str:
    """Hash project Python sources and the dependency lock for run provenance."""
    digest = hashlib.sha256()
    files = sorted(path for path in PROJECT_ROOT.rglob("*.py")
                   if ".venv" not in path.parts and "results" not in path.parts)
    files.append(PROJECT_ROOT / "requirements.txt")
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
