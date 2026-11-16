"""Hardware metadata, synchronized timers, and process/GPU memory monitoring."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psutil


def hardware_metadata(device: Optional[Any] = None) -> Dict[str, Any]:
    """Record actual hardware and software even when PyTorch is unavailable."""
    metadata: Dict[str, Any] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0], "python_executable": sys.executable,
        "os": platform.platform(), "cpu_model": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_ram_bytes": psutil.virtual_memory().total,
        "device": str(device) if device is not None else "not_selected",
        "torch_version": None, "cuda_version": None, "cuda_available": False,
        "gpu_model": None, "gpu_models_detected": [], "gpu_driver": None,
    }
    packages = {}
    for package in ("torch", "numpy", "scipy", "numba", "PyYAML", "matplotlib", "psutil", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata["package_versions"] = packages
    if importlib.util.find_spec("torch"):
        import torch

        metadata.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        cuda_available=torch.cuda.is_available(), torch_num_threads=torch.get_num_threads())
        if torch.cuda.is_available() and device is not None and device.type == "cuda":
            active = device
            properties = torch.cuda.get_device_properties(active)
            metadata.update(gpu_model=properties.name, gpu_total_memory_bytes=properties.total_memory,
                            gpu_compute_capability=[properties.major, properties.minor])
    try:
        command = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if command.returncode == 0:
            rows = [line.rsplit(",", 1) for line in command.stdout.strip().splitlines()]
            metadata["gpu_models_detected"] = [row[0].strip() for row in rows]
            metadata["gpu_driver"] = rows[0][1].strip() if rows else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return metadata


def hardware_fingerprint(metadata: Dict[str, Any]) -> str:
    """Cache identity excluding timestamp/PID and including versions/hardware."""
    from ..common import config_hash
    keys = ("os", "cpu_model", "logical_cpu_count", "physical_cpu_count", "system_ram_bytes",
            "gpu_models_detected", "gpu_driver", "torch_version", "cuda_version", "cuda_available",
            "python_version", "package_versions")
    return config_hash({key: metadata.get(key) for key in keys})


def synchronize(device: Any) -> None:
    """Make CUDA wall-clock boundaries include queued kernels and transfers."""
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


class MemoryMonitor:
    """Sample process RSS and measure PyTorch's allocated GPU memory peak.

    CPU RSS includes native NumPy/PyTorch allocations. It is sampled (default
    10 ms plus explicit per-step samples), so it is an observed peak, not an
    exact OS high-water mark. GPU values use max_memory_allocated, not reserved
    allocator memory or nvidia-smi's total process allocation.
    """

    def __init__(self, device: Optional[Any] = None, interval: float = 0.01) -> None:
        self.device = device
        self.interval = interval
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.gpu_baseline = 0
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def sample(self) -> None:
        """Capture an explicit RSS sample at an allocation-heavy point."""
        self.peak = max(self.peak, self.process.memory_info().rss)

    def start(self) -> "MemoryMonitor":
        self.started = time.perf_counter()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        if self.device is not None and self.device.type == "cuda":
            import torch
            synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_baseline = torch.cuda.memory_allocated(self.device)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return absolute and baseline-adjusted peaks."""
        if self.device is not None:
            synchronize(self.device)
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        gpu_peak = 0
        if self.device is not None and self.device.type == "cuda":
            import torch
            gpu_peak = torch.cuda.max_memory_allocated(self.device)
        return {"cpu_peak_rss_bytes": self.peak, "cpu_baseline_rss_bytes": self.baseline,
                "cpu_peak_increment_bytes": max(0, self.peak-self.baseline),
                "cpu_rss_sampling_interval_seconds": self.interval,
                "gpu_peak_allocated_bytes": gpu_peak, "gpu_baseline_allocated_bytes": self.gpu_baseline,
                "gpu_peak_increment_bytes": max(0, gpu_peak-self.gpu_baseline),
                "memory_measurement_seconds": time.perf_counter()-self.started}
