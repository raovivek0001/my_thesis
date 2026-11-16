You are an expert in scientific machine learning and computational seismology. Build a complete, reproducible PyTorch project that extends the PINNeik method (Waheed et al., 2021, Computers & Geosciences 155, 104833) from 2D to 3D and measures its computational cost.
## 0. Constraints on existing work
- Create everything inside a NEW directory named `pinneik3d/`. Do not create, modify, rename or delete any file outside it.
- Do not modify any existing PINNeik code, data, notebooks or papers. If you want to reuse an existing 2D function, copy it into `pinneik3d/` instead of importing from or editing the original.
- Do not run any command that deletes or overwrites files outside `pinneik3d/`.
- Before writing any file, show me the list of files you intend to create and wait for my confirmation.

## 1. Mathematical formulation
Solve the isotropic eikonal equation |∇T|² = 1/v²(x), T(xs) = 0, x = (x, y, z), using the multiplicative factorization:
- T(x) = T0(x) · τ(x), with T0(x) = |x − xs| / v(xs)
- Factored residual: R = T0²|∇τ|² + τ²|∇T0|² + 2·T0·τ·(∇T0·∇τ) − 1/v²(x)
- ∇T0 = (x − xs) / (|x − xs| · v(xs)). Guard against division by zero near xs (add a small epsilon).
- Network N(x; θ) ≈ τ(x). Use autograd to get ∇τ (first derivatives only).
- Loss = mean(R²) + mean(H(−τ)·|τ|)² + (τ(xs) − 1)². Here H is the Heaviside function, so the second term penalizes negative τ.
- Final traveltime: T̂ = T0 · τ̂

## 2. Network and training
- Fully connected MLP, input 3, output 1, default 10 hidden layers × 20 neurons, linear output layer.
- Locally adaptive activation: atan(n · a_k · z) with a learnable slope a_k per neuron (n = scale factor, a_k initialised to 1/n), as in Jagtap et al. (2020). Make the activation configurable (tanh, atan, sin, adaptive atan).
- Adaptive loss weighting based on back-propagated gradient statistics (Wang et al., 2020), configurable on/off.
- Optional Fourier feature input encoding (random Fourier features, configurable scale and size) to address spectral bias.
- Mini-batch training: resample N_b random collocation points uniformly in the domain every iteration (the 3D grid is too large for full-batch). Stage 1 uses Adam with a learning-rate schedule. Stage 2 (optional) uses L-BFGS on a fixed collocation set that fits in memory.
- Velocity access: support (a) analytic velocity functions, and (b) gridded 3D velocity volumes via trilinear interpolation (torch.nn.functional.grid_sample).
- Support float32 and mixed precision. Set seeds for full reproducibility.

## 3. Velocity models (in a `models/` module)
1. Constant-gradient: v = v0 + g·z (defaults v0 = 2 km/s, gz = 0.5 /s, domain 2×2×2 km). Include an analytic solution: T = (1/|g|)·arccosh(1 + |g|²·|x − xs|² / (2·v(x)·v(xs))), with g the gradient vector.
2. Smooth heterogeneous model (e.g., constant gradient plus a few Gaussian anomalies).
3. Loader for a gridded 3D velocity volume (.npy / .bin), so SEG/EAGE salt or Overthrust can be dropped in later.

## 4. Baselines
- First-order 3D Fast Sweeping Method (8 sweep directions, factored version, Gauss-Seidel updates). Implement in Numba or C++ via pybind so timing is fair. Include a convergence tolerance.
- Optional: a higher-order FSM or fast marching baseline (use scikit-fmm if available).
- Compare against the analytic solution where one exists, otherwise against FSM on a 2× finer grid.

## 5. Surrogate and transfer learning
- Surrogate mode: network input becomes (x, y, z, xs, ys, zs). Train on a regular lattice of source positions (configurable, default 4×4×4 = 64 sources). Evaluate on held-out sources, including one deliberately far from all training sources.
- Transfer learning: initialise from previously trained weights for a changed source location or a perturbed velocity model, and compare convergence against random initialisation.

## 6. Benchmark and measurement suite (`benchmarks/`)
Record each of the following in a CSV and JSON:
- Wall-clock training time, and time to reach a fixed relative-L2 error (1e-2 and 1e-3).
- Peak GPU memory (torch.cuda.max_memory_allocated) and peak CPU RAM, as a function of batch size.
- Inference time for full-grid evaluation at 51³, 101³ and 201³.
- FSM runtime on the same grids and hardware.
- Errors: relative L2, max absolute error, and mean absolute error against the reference.
- Scaling study over grid sizes 51³, 101³, 201³ and collocation budgets.
- Break-even analysis: the number of source locations at which the surrogate (training time + per-source inference) beats FSM.
- Run every experiment with at least 3 seeds, and report mean ± std.
- Log GPU model, CUDA version, PyTorch version and all hyperparameters with every run.

## 7. Outputs
- Plots (matplotlib): absolute error slices (xy, xz, yz through the source), traveltime contour comparison (analytic / PINN / FSM), loss curves, time-vs-accuracy plot, memory-vs-batch-size plot, break-even plot.
- A final `results/summary.md` with tables ready to paste into a paper.

## 8. Project structure and engineering requirements
```
pinneik3d/
  models/        (velocity models + analytic solutions)
  networks.py    (MLP, adaptive activations, Fourier features)
  losses.py      (factored residual, adaptive weighting)
  train.py       (Adam + L-BFGS, mini-batching, checkpointing)
  fsm3d.py       (baseline)
  surrogate.py
  transfer.py
  benchmarks/
  configs/       (YAML, one per experiment)
  tests/
  README.md
  requirements.txt
```
- Drive everything from YAML configs and a CLI (`python train.py --config configs/const_grad.yaml`).
- Include unit tests: the analytic solution satisfies the eikonal equation numerically, the factored residual is ≈ 0 for the analytic τ, and the FSM matches the analytic solution within the expected first-order error.
- Add type hints and docstrings, and a README with exact commands to reproduce every figure.
- Start with a small smoke-test config (e.g., 31³ grid, a few hundred iterations) that runs on CPU in under 2 minutes, then provide the full-scale configs.

## 9. Process
First print the plan and the file tree. Then implement module by module, running the unit tests and the smoke test after each stage. Report any deviations from the spec, and anything you could not verify (for example GPU timings if no GPU is available).