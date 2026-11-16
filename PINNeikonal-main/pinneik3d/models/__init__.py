"""Analytic and sampled velocity models; PyTorch imports are lazy."""

from .analytic import tau_numpy, tau_torch, traveltime_numpy, traveltime_torch
from .velocity import ConstantGradient, GaussianAnomaly, SmoothHeterogeneous, VelocityModel, build_velocity
from .volume import GriddedVelocity

__all__ = ["ConstantGradient", "GaussianAnomaly", "SmoothHeterogeneous", "VelocityModel",
           "GriddedVelocity", "build_velocity", "traveltime_numpy", "traveltime_torch",
           "tau_numpy", "tau_torch"]
