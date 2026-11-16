# PINNeik3D

A PyTorch extension of PINNeik to the **3D isotropic eikonal equation**, with
mini-batch collocations, a compiled factored fast sweeping baseline, source
surrogates, transfer learning, and repeat-seed cost measurements. All project
files, checkpoints, plots, caches, and reports stay inside `pinneik3d/`.

**Validation status:** consult [`results/summary.md`](results/summary.md) and
[`results/validation.json`](results/validation.json) for checks actually run in
this workspace. Dependency downloads were blocked by socket permissions during
implementation. The available NumPy/SciPy mathematics and diagnostic FSM can be
checked independently; PyTorch training, Numba compilation, GPU measurements,
and the full performance matrix must be verified after installing dependencies.
No predicted fields or performance results are synthesized to fill missing runs.

## Install in a separate environment

The pinned requirements support **64-bit CPython 3.8–3.11**. Use this environment
for the new project, independently of the original TensorFlow/SciAnn notebooks.
Python 3.12/3.14 require a different dependency lock and are outside this lock.

PowerShell, starting in the existing repository root:

```powershell
cd .\pinneik3d
python --version
python -m venv .venv
$pinneikPython = (Resolve-Path '.venv\Scripts\python.exe').Path
New-Item -ItemType Directory -Force '.venv\tmp' | Out-Null
$env:TEMP = (Resolve-Path '.venv\tmp').Path
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
& $pinneikPython -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
& $pinneikPython -m pip check
```

If `python --version` is outside 3.8–3.11, use an installed supported interpreter
to create the environment. The original workspace's Python 3.8 interpreter is
suitable. For a CPU-only PyTorch wheel, install PyTorch from its CPU index before
installing the rest; the pinned requirement also accepts this local wheel version:

```powershell
& $pinneikPython -m pip install --disable-pip-version-check --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
& $pinneikPython -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
```

For NVIDIA CUDA 12.1 wheels, use `https://download.pytorch.org/whl/cu121` in that
command. Confirm actual support rather than relying on GPU detection alone:

```powershell
& $pinneikPython -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Optional higher-order comparison:

```powershell
& $pinneikPython -m pip install --disable-pip-version-check --no-cache-dir scikit-fmm==2024.9.16
```

scikit-fmm may require a C++ compiler on Windows. It is optional; the required
first-order baseline uses Numba. Its absence is logged explicitly.

On Linux/macOS, with a supported `python3` interpreter:

```bash
cd pinneik3d
python3 -m venv .venv
source .venv/bin/activate
mkdir -p .venv/tmp
export TMPDIR="$PWD/.venv/tmp"
export PYTHONDONTWRITEBYTECODE=1
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
python -m pip check
```

All commands below assume the working directory is `pinneik3d/`. PowerShell
uses `& $pinneikPython`; with an activated environment, replace it by `python`.
From the repository root, the package CLI also works as
`python -m pinneik3d.train --config configs/smoke.yaml` with the same environment.

## Start with validation and the CPU smoke

```powershell
& $pinneikPython -m unittest discover -s tests -t .. -v
& $pinneikPython -m pytest tests -q
& $pinneikPython validate.py --smoke
```

`smoke.yaml` uses the default 10 hidden layers × 20 neurons, 200 Adam iterations,
128 collocations per iteration, a 31³ evaluation grid, and three CPU seeds
(42, 43, 44). It is designed for the under-two-minutes CPU smoke target;
`smoke_evidence.json` records the actual wall time and whether that target was
met. It is a workflow check, not an assurance of reaching 1e-3 accuracy.

Run only the training smoke, or the additional small workflows:

```powershell
& $pinneikPython train.py --config configs/smoke.yaml
& $pinneikPython surrogate.py --config configs/surrogate_smoke.yaml
& $pinneikPython transfer.py --config configs/transfer_smoke.yaml
& $pinneikPython benchmarks/run.py --config configs/benchmark_smoke.yaml
```

The benchmark smoke is a separate, larger workflow test: 12 independent training
runs (two batches × two modes × three seeds) with inference/FSM at 11³, 21³, 31³.
It is not covered by the single CPU training smoke's two-minute target.

Without PyTorch/Numba, use an explicitly limited numerical check:

```powershell
python validate.py --math-only --smoke
```

This can use the workspace's existing NumPy/SciPy interpreter. It runs the
available unit tests and the 31³ factored stencil for three seeds, records
missing dependencies and skips, and plots actual FSM/analytic comparisons.
The pure Python solver is diagnostic: its times are excluded from compiled-FSM
benchmark columns. The full `validate.py --smoke` returns a nonzero status when
required training dependencies are missing.

## Accuracy-focused training and comparison

The enhanced presets are candidates for improved accuracy; no improvement is
claimed until fresh training is measured. The original baseline presets remain
available for reproducible comparisons. From the repository root, using the
separate environment after installing `pinneik3d/requirements.txt`:

```powershell
.\pinneik3d\.venv\Scripts\python.exe -m pinneik3d.compare --smoke --dry-run
.\pinneik3d\.venv\Scripts\python.exe -m pinneik3d.compare --smoke
.\pinneik3d\.venv\Scripts\python.exe -m pinneik3d.train --config configs/enhanced.yaml
.\pinneik3d\.venv\Scripts\python.exe -m pinneik3d.train --config configs/enhanced_heterogeneous.yaml
.\pinneik3d\.venv\Scripts\python.exe -m pinneik3d.compare --output-dir results/accuracy_comparison_full
.\pinneik3d\.venv\Scripts\python.exe -m pinneik3d.compare --baseline configs/heterogeneous.yaml --enhanced configs/enhanced_heterogeneous.yaml --output-dir results/heterogeneous_comparison
```

The enhanced network has six hidden layers of width 64. It uses the
dimensionless residual `v² R`, which has the same PDE roots and reduces the
implicit `v^-4` down-weighting of errors in faster regions. The loss's source
and positivity terms retain their original definitions. Adam's existing cosine
schedule is followed by fixed-set L-BFGS, with float32 residual arithmetic.

After warmup, residual sampling scores a fresh candidate pool every 50 Adam
steps using `abs(v² R)`. Half of each optimization batch still samples uniformly;
the other half draws from the high-residual pool. Candidate scoring uses no
reference labels, is chunked to the optimization batch size, and its time is
included in training time. Candidate-scoring counts are logged separately from
the optimization collocation budget. The heterogeneous preset uses lower
Fourier frequency scale (0.25) as a starting point for these smooth anomalies;
this setting needs tuning for sharper volumes.

`select_best: true` evaluates the weights with the smallest observed validation
relative L2, including the initial model. `best_model.pt` contains those weights
for inference/transfer. `checkpoint.pt` retains the final weights with matching
optimizer/RNG state for exact resume. The residual pool and best-weight state
are preserved in full checkpoints.

The comparison runs three paired seeds with identical physics, precision,
evaluation grids, Adam updates/batches and L-BFGS budgets. It writes measured
`runs.csv`, `runs.json`, `comparison.json` and `summary.md` to its output folder;
the summary reports regressions as well as improvements. Extra scoring, a wider
network, and the number of L-BFGS closure calls can change runtime despite equal
optimizer budgets. The six-run enhanced smoke is a workflow check without a
two-minute runtime promise. The original 2D notebooks solve different problems
and require a separately defined comparison before claiming superiority.

Current enhancement checks and dependency limitations are recorded in
[`results/enhancement_status.md`](results/enhancement_status.md).

## Equations and implementation choices

Coordinates are `(x,y,z)` in **km**, velocity in **km/s**, traveltime in **s**,
and gradient components in **s⁻¹**. Arrays use `volume[ix,iy,iz]` throughout.

\[
|\nabla T|^2 = v(x)^{-2},\quad T(x_s)=0,\quad
T=T_0\tau,\quad T_0=\frac{\|x-x_s\|}{v(x_s)}.
\]

The network approximates `tau`, with physical coordinate normalization to
`[-1,1]` inside the graph. Autograd computes only the first coordinate derivative
needed by the PDE. Parameter optimization differentiates through that derivative:

\[
R=T_0^2\|\nabla\tau\|^2+\tau^2\|\nabla T_0\|^2
 +2T_0\tau(\nabla T_0\cdot\nabla\tau)-v^{-2},\qquad
\nabla T_0=\frac{x-x_s}{\max(\|x-x_s\|,\epsilon)v(x_s)}.
\]

The distance in `T0` is exact, so the reconstructed traveltime is exactly zero
at the source. The PDE is singular at that point; its source condition is imposed
separately with `tau(xs)=1`. The objective follows the specified **square of the
mean** convention for negative factors:

\[
L=\operatorname{mean}(R^2)
 +\big[\operatorname{mean}(\operatorname{ReLU}(-\tau))\big]^2
 +\operatorname{mean}((\tau(x_s)-1)^2).
\]

The last mean averages source conditions in surrogate mode. Weights default to
one; gradient balancing can change the auxiliary weights. Fixed velocity/T0
coefficients are detached from parameter backpropagation, avoiding unwanted
velocity derivatives and `grid_sample` double-backward limitations.

The MLP defaults to 3 inputs, 10 hidden layers of width 20, and a linear output.
`adaptive_atan` uses **one learned slope per neuron per layer**,
`atan(n*a_k*z)`, initialized with `a_k=1/n`. `tanh`, `atan`, and `sin` are also
available. Optional Fourier encoding stores its random frequency matrix in the
checkpoint; `scale` applies to normalized coordinates. Initialization has output
bias one and small output weights, without hard-wiring the solution.

Wang-style balancing compares `max(abs(grad L_pde))` to
`mean(abs(grad L_aux))`, applies an exponential moving average, and caps weights.
Zero-gradient terms retain their weights. No slope-recovery loss is added to the
specified objective.

Adam resamples uniform physical collocations every iteration. L-BFGS uses one
fixed collocation set, all lattice source boundary conditions, fixed loss weights,
and float32 arithmetic during line search. Mixed precision autocasts the network
(CUDA float16 or CPU bfloat16 by default); residual arithmetic stays float32.
Seeds cover Python, NumPy, PyTorch and CUDA; deterministic algorithms are enabled
and TF32/cudnn benchmarking are disabled. Bitwise repeatability is scoped to the
same software versions and hardware; changing precision/device can change results.

## Velocity models and references

`models/velocity.py` provides `ConstantGradient` and smooth Gaussian anomalies.
The default is `v = 2 + 0.5*z` in a 2×2×2 km domain. General three-component
linear gradients and a velocity origin are supported. The analytic reference is

\[
T=\frac1{|g|}\operatorname{acosh}\left(1+
\frac{|g|^2\|x-x_s\|^2}{2v(x)v(x_s)}\right).
\]

For numerical stability, code evaluates its equivalent
`2*asinh(|g|*r/(2*sqrt(v*vs)))/|g|`. Zero gradient uses `r/vs`, and analytic
`tau` is set to one at the source. Tests independently finite-difference the
analytic traveltime and factor, and test the PyTorch residual when available.

For a volume, place a real dataset at `data/velocity.npy` inside this directory
and edit `configs/volume.yaml` for its domain, source, axis order, and units:

```powershell
& $pinneikPython train.py --config configs/volume.yaml
```

`.npy` data must be three-dimensional. For raw `.bin`, specify **file-axis**
`shape`, endianness/dtype (e.g. `<f4`), `order: C` or `F`, `axis_order: xyz` or a
permutation, and `velocity_scale: 0.001` for m/s input. A zyx file is transposed
to xyz. Torch's 5D `grid_sample` uses `(N,C,D,H,W)`; query coordinates are therefore
passed in `(z,y,x)` order with `align_corners=True`. Tests use a nonsymmetric
linear volume to catch axis swaps. SEG/EAGE salt and Overthrust datasets are not
bundled; their dimensions, geometry and units must come from the actual dataset.

The required FSM is a **factored first-order Godunov scheme**, not an unfactored
solver followed by division. It differences `tau`, includes the analytic `grad T0`,
solves the local piecewise quadratic, and uses eight in-place Gauss-Seidel sweep
directions. Numba compiles the serial float64 kernel without fast-math shortcuts.
Stopping uses the maximum change in traveltime over a complete sweep cycle;
unconverged runs raise rather than silently becoming references.

Grid-aligned sources pin one node with `tau=1`. For an off-grid source, its true
position is retained and a ball of radius one cell diagonal uses a local linear
velocity approximation. The ball shrinks on refinement, preserving first-order
consistency. The optional second-order scikit-fmm comparison supports grid-aligned
point sources and logs a skip for off-grid sources.

References are analytic for a constant gradient. Otherwise they use FSM with
**twice as many cells per axis**: 51→101, 101→201, 201→401 nodes. Reference
construction is timed separately. First-order discretization error in that
reference limits claims about true 1e-3 accuracy; the solver's convergence
tolerance controls iteration error, not discretization error.

## Full experiments and exact figure commands

```powershell
& $pinneikPython train.py --config configs/const_grad.yaml
& $pinneikPython train.py --config configs/heterogeneous.yaml
& $pinneikPython surrogate.py --config configs/surrogate.yaml
& $pinneikPython transfer.py --config configs/transfer.yaml
& $pinneikPython benchmarks/run.py --config configs/benchmark.yaml --dry-run
& $pinneikPython benchmarks/run.py --config configs/benchmark.yaml
```

The full benchmark contains **216 training runs** across two velocity presets,
three batches, three collocation budgets, two precisions, two source modes, and
three seeds. Each trained network is reused for 51³, 101³, 201³ inference; grid
resolution is not the training collocation count. Heterogeneous references for
201³ evaluation use 401³ nodes. This matrix is substantial; the dry run prints
its exact settings. Already completed workers are reused only if their effective
specification, source-code hash, and hardware/software fingerprint match.

| Figure | Exact generating command | Output (PNG and PDF) |
|---|---|---|
| Absolute xy/xz/yz error slices | `python train.py --config configs/const_grad.yaml` | `results/const_grad/seed_42/figures/error_slices.*` |
| Analytic/PINN/FSM contour comparison | Same constant-gradient command | `results/const_grad/seed_42/figures/traveltime_contours.*` |
| Loss curves | Same constant-gradient command | `results/const_grad/seed_42/figures/loss_curves.*` |
| Online time to accuracy | Same constant-gradient command | `results/const_grad/seed_42/figures/time_vs_accuracy.*` |
| Heterogeneous error and contours | `python train.py --config configs/heterogeneous.yaml` | `results/heterogeneous/seed_42/figures/` |
| Held-out/far-source error slices | `python surrogate.py --config configs/surrogate.yaml` | `results/surrogate/seed_42/figures/far/` and `held_out_1/` |
| Source break-even | Same surrogate command | `results/surrogate/figures/break_even.*` |
| Transfer convergence | `python transfer.py --config configs/transfer.yaml` | `results/transfer/figures/transfer_convergence.*` |
| Memory versus batch | `python benchmarks/run.py --config configs/benchmark.yaml` | `results/benchmarks/figures/memory_vs_batch_size.*` and grouped subfolders |
| Grid scaling | Same benchmark command | `results/benchmarks/figures/grid_scaling.*` and grouped subfolders |
| Final-grid time versus accuracy | Same benchmark command | `results/benchmarks/figures/time_vs_accuracy.*` |
| Benchmark break-even | Same benchmark command | `results/benchmarks/figures/<experiment>_<precision>_<device>_b<batch>_c<budget>/break_even.*` |

Error planes are interpolated through the **actual source coordinate**, rather
than shifted to the nearest grid slice. Tables use mean ± sample std across seeds.
The aggregate plots group comparable experiment/device/precision settings;
individual run plots preserve their source/reference labels.

Regenerate figures from saved arrays and histories without retraining:

```powershell
& $pinneikPython benchmarks/plotting.py --fields results/const_grad/seed_42/field.npz --history results/const_grad/seed_42/history.json --output results/rebuilt/const_grad
& $pinneikPython benchmarks/run.py --config configs/benchmark.yaml --report-only
```

Use `configs/smoke.yaml` and `results/smoke/...` for the same small figure set.
The actual math-only figures available without PyTorch are regenerated with:

```powershell
python benchmarks/plotting.py --fields results/validation/field.npz --output results/rebuilt/mathematical_check
```

## Surrogate and transfer comparisons

Surrogates take `(x,y,z,xs,ys,zs)`. The default 4³ source lattice occupies
`[0.4,1.6]³` km. Collocation source indices sample that lattice each iteration;
held-out sources are never used in the objective. The auto far source maximizes
distance to its nearest lattice source over the physical domain (0.693 km for
the default box). Its extrapolation status and distance are recorded. Online
surrogate accuracy is the **worst relative L2 among all held-out sources**.

`transfer.yaml` tests a moved source and a velocity perturbation. It reuses
completed matching-seed constant-gradient checkpoints when available; otherwise
it pretrains them in `results/transfer/base/`. Random and transferred target runs
receive identical collocations and optimizer schedules. Target convergence times
exclude pretraining cost, which is saved separately. Speedups remain null when
either variant does not reach the target.

## Measurements and report interpretation

Every `metrics.csv/json` and benchmark row records effective hyperparameters,
seed, parameter count, source-code hash, UTC timestamp, device, GPU model, CUDA/PyTorch/Python/package versions,
source geometry, references, and actual failure/skip reasons.

* Training time includes sampling, optimization, online validation and periodic
  checkpoints; model setup and reference construction are separate. Final-grid
  inference, FSM, plotting and final serialization are outside training time.
* Time to relative L2 1e-2/1e-3 is the **first observed** crossing on a fixed
  validation grid. Evaluation resolution/interval are logged. Targets not reached
  remain null and are right-censored; successful counts accompany means/stds.
* Inference includes coordinate generation, device transfers and CPU output
  collection, in bounded chunks. CUDA is synchronized at timer boundaries.
  Warmup is excluded. Inference repeat times are logged in each seed run.
* FSM uses the CPU on the same host, irrespective of the PINN device. Its timing
  includes velocity sampling and solver allocation/initialization; compilation
  is warmed separately and excluded. Serial Gauss-Seidel is not GPU execution.
* CPU peak RAM is observed process RSS sampled at 10 ms and explicitly per
  training step. Absolute baseline and incremental peaks are recorded. GPU peak
  is `torch.cuda.max_memory_allocated`; baseline/incremental allocations are also
  recorded. Reserved allocator memory is not substituted for allocated memory.
* Fresh worker processes isolate training/memory runs by default. This prevents
  earlier optimizer/reference allocations from becoming another run's baseline.
* Collocation budgets are exact Adam draws (`iterations * batch_size`). L-BFGS
  is disabled for the default matrix so resampled budgets remain comparable;
  standalone full configs demonstrate its optional second stage.
* Cost break-even is the smallest integer `K` with
  `setup + training + K*inference < K*FSM`, computed as
  `floor((setup+training)/(FSM-inference))+1`. There is no finite crossing if
  inference costs at least FSM. A separate accuracy-qualified crossing requires
  both methods to meet the target on every measured held-out source, including
  the far source. Cost-only crossings do not imply equal accuracy.

`results/summary.md` is the latest generated paper/status report. Each experiment
also retains its own `summary.md`, `runs.csv/json`, and `aggregate.csv/json`.
Benchmark failures preserve completed earlier grid measurements and the failed
context in `worker_results.json`; they are never turned into zero-cost results.

## Configuration and checkpoints

YAML supports `extends: base.yaml`; paths in configs resolve relative to this
project directory. Output paths escaping it are rejected. Use `--dry-run` to
validate settings without importing PyTorch:

```powershell
& $pinneikPython train.py --config configs/const_grad.yaml --dry-run
& $pinneikPython surrogate.py --config configs/surrogate.yaml --dry-run
```

Enable mixed precision in YAML or save a separate CLI experiment:

```powershell
& $pinneikPython train.py --config configs/const_grad.yaml --precision mixed --device cuda --output-dir results/const_grad_mixed
```

Full-state resume restores model, optimizer, scheduler, scaler, adaptive weights,
collocation generator and Python/NumPy/PyTorch/CUDA RNG states. Resume requires
matching source/medium/architecture/precision/budgets so the trajectory remains
reproducible. `--initial` loads weights only for a changed target. Example resume:

```powershell
& $pinneikPython train.py --config configs/const_grad.yaml --seed 42 --resume results/const_grad/seed_42/checkpoint_iter_002000.pt --output-dir results/resumed_const_grad
```

`--seed` is a single-seed diagnostic override. Full experiment configs and the
benchmark/surrogate/transfer drivers require three seeds for their study reports.

## Project layout

```text
models/          analytic/gridded velocities and analytic traveltimes
networks.py      normalized MLP, neuron-wise adaptive atan, Fourier encoding
losses.py        factored residual, source/positivity loss, gradient balancing
train.py         Adam, schedules, fixed-set L-BFGS, checkpointing, inference
fsm3d.py         compiled factored first-order sweeping; optional scikit-fmm
surrogate.py     regular source lattice, disjoint holdouts, distant-source test
transfer.py      matched source/velocity transfer comparisons
common.py        YAML inheritance, seeds, runtime, bounded output paths
grids.py         xyz grids, chunking, interpolation
reference.py     analytic / h-over-2 FSM references
benchmarks/      fresh-process matrix, metrics, timers, plots, paper tables
configs/         small CPU workflows and full experiment presets
tests/           numerical equations, stencil refinement, networks, checkpoints
validate.py      dependency-aware verification and evidence reports
results/         measured artifacts and explicit validation status
```

## References

* Waheed, U. bin, Haghighat, E., Alkhalifah, T., Song, C., and Hao, Q. (2021).
  PINNeik: Eikonal solution using physics-informed neural networks.
  *Computers & Geosciences*, **155**, 104833.
* Jagtap, A. D., Kawaguchi, K., and Karniadakis, G. E. (2020). Locally adaptive
  activation functions with slope recovery for deep and physics-informed neural
  networks. *Proceedings of the Royal Society A*, **476**, 20200334.
* Wang, S., Teng, Y., and Perdikaris, P. (2020 preprint; 2021 publication).
  Understanding and mitigating gradient flow pathologies in physics-informed
  neural networks. *SIAM Journal on Scientific Computing*, **43**, A3055–A3081.
