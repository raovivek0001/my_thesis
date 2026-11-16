# PINNeik3D validation status

This file reports checks actually performed in this workspace. Full PyTorch performance experiments have not been measured unless training results are explicitly included below.

| Check | Result |
|---|---|
| Python syntax | 30 files parsed with Python 3.8 grammar |
| YAML configurations | 11 validated |
| Executed unit tests | 26 passed; 0 failed; 0 errors |
| Skipped checks | 12; see validation.json for individual reasons |
| Dependency availability | Missing: torch, numba |
| Python | 3.8.10 |
| GPU detected | NVIDIA GeForce GTX 1650 |
| PyTorch / CUDA | unavailable / unavailable |

Numerical 31³ smoke, mean ± sample standard deviation over seeds 42, 43, and 44:

| Backend | Relative L2 | Max absolute error (s) | Mean absolute error (s) |
|---|---:|---:|---:|
| python | 0.00027106 ± 0 | 0.000365809 ± 0 | 8.2833e-05 ± 0 |

This is the diagnostic Python implementation of the factored stencil. Its wall times are excluded from FSM benchmark columns; the Numba-compiled kernel is not verified in this environment.

Actual analytic/FSM error-slice and contour figures: `validation/figures/error_slices.png` and `validation/figures/traveltime_contours.png` (also PDF). No PINN field is fabricated.

| Training smoke | Status |
|---|---|
| 200 Adam iterations, 31³ evaluation, 3 CPU seeds | blocked |

PyTorch/Numba are unavailable. Dependency downloads were blocked by network socket permissions (WinError 10013) during this workspace session.

The CPU training-under-two-minutes target, mixed precision, GPU timings/memory, full 51³/101³/201³ scaling, surrogate accuracy, transfer speedups, and cost break-even remain unverified.

Run `python validate.py --smoke` after installing requirements. Run the full commands in README.md to replace this status report with measured paper tables. The full benchmark YAML defines 216 independent training runs, each setting with three seeds.

Implemented choices: stable asinh form of the acosh analytic solution; guarded gradient denominator with exact T0(xs)=0; positivity uses square of the mean as specified; off-grid FSM uses a shrinking local injection region; optional scikit-fmm is used only for grid-aligned sources.
