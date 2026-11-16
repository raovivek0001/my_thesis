"""Local setup shared by the original SciAnn eikonal notebooks.

This module uses only the standard library so setup can report missing
dependencies before the notebooks attempt scientific-library imports.
"""

import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path


CASES = {
    "vofz": ("inputs/vofz/traveltimes/Tcomp.npy",),
    "vofxz": (
        "inputs/vofxz/traveltimes/Tcomp.npy",
        "models/vofz_model-end.hdf5",
    ),
    "vofxz_surrogate": ("inputs/vofxz_surrogate/traveltimes/Tcomp.npy",),
    "vtiseam": (
        "inputs/vtiseam/model/vzcoarse.npy",
        "inputs/vtiseam/model/vxcoarse.npy",
        "inputs/vtiseam/model/elevation.txt",
        "inputs/vtiseam/traveltimes/Tref.npy",
        "inputs/vtiseam/traveltimes/Tcomp.npy",
        "models/vofz_model-end.hdf5",
    ),
    "marm": (
        "inputs/marm/model/marm_vz.txt",
        "inputs/marm/traveltimes/fmm_or2_marm_s(1,1).txt",
        "inputs/marm/traveltimes/Tcomp.npy",
        "models/vofz_model-end.hdf5",
    ),
}


def prepare_workspace(case):
    """Resolve data paths from this module and create figure/model folders."""
    codes_dir = Path(__file__).resolve().parent
    missing = [name for name in CASES[case] if not (codes_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing input files under {}:\n{}".format(
                codes_dir, "\n".join(missing)
            )
        )
    (codes_dir / "figs" / case).mkdir(parents=True, exist_ok=True)
    (codes_dir / "models").mkdir(parents=True, exist_ok=True)
    os.chdir(str(codes_dir))
    print("Working directory: {}".format(codes_dir))
    return codes_dir


def check_dependencies():
    """Check the documented legacy runtime before loading TensorFlow/SciAnn."""
    requirements = Path(__file__).resolve().parent.parent / "requirements-legacy.txt"
    issues = []
    if sys.version_info[:2] != (3, 8):
        issues.append(
            "The pinned environment requires Python 3.8; this kernel uses {}.{}.".format(
                sys.version_info.major, sys.version_info.minor
            )
        )

    packages = ("numpy", "pandas", "matplotlib", "scipy", "tensorflow", "sciann", "keras", "h5py")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        issues.append("Missing packages: {}.".format(", ".join(missing)))

    # These are the two versions documented by the original repository.
    for package, expected in (("tensorflow", "2.2.0"), ("sciann", "0.5.4.0")):
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        if actual != expected:
            issues.append("{} {} is installed; this implementation requires {}.".format(
                package, actual, expected
            ))

    if issues:
        raise RuntimeError(
            "\n".join(issues)
            + "\nCreate the Python 3.8 environment described in README.md, install "
            + str(requirements)
            + ", select the PINNeikonal (Python 3.8) kernel, and restart the notebook."
        )
    print("Dependency check passed: Python 3.8, TensorFlow 2.2.0, SciAnn 0.5.4.0.")
