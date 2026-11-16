"""Fixed-data L-BFGS refinement for the original TensorFlow/SciAnn 2D models.

Uses a float64 graph objective and gradient with one full batch, so both
have the same normalization and avoid Keras's rounded reporting metrics.
No reference traveltimes are added to a physics-only model's training loss.
"""

import time

import numpy as np


def stratified_source_indices(source_x, source_z, per_source=1000, seed=4321):
    """Keep equal collocation counts per original training source."""
    if per_source < 1:
        raise ValueError("per_source must be positive")
    sources = np.column_stack((np.asarray(source_x).ravel(), np.asarray(source_z).ravel()))
    unique, groups = np.unique(sources, axis=0, return_inverse=True)
    rng = np.random.RandomState(seed)
    selected = []
    for group in range(len(unique)):
        indices = np.flatnonzero(groups == group)
        selected.append(rng.choice(indices, min(per_source, len(indices)), replace=False))
    return np.concatenate(selected)


def lbfgs_refine(model, inputs, targets, iterations=600, target_weights=None,
                 log_every=50, ftol=1e-15, gtol=1e-10, verify_gradient=False):
    """Refine shared network weights with a stationary full-batch objective.

    Returns a SciModel sharing the original functionals and a measured history.
    Unlike SciAnn's default SciPy wrapper, the optimization tolerances are
    explicit; a small raw residual loss must not trigger premature termination.
    The best finite training objective is retained if the line search fails.
    """
    from scipy.optimize import minimize
    from sciann import SciModel
    from sciann.utils.optimizers import ScipyOptimizer
    from tensorflow.python.keras import backend as K

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if len(targets) != len(model.constraints):
        raise ValueError("Provide a target for every original model constraint")
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1, 1) for value in inputs]
    count = len(arrays[0])
    if any(len(value) != count or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Training inputs must have matching lengths and finite values")
    refined = SciModel(model.inputs, model.constraints, optimizer="scipy-l-BFGS-B")
    truth, weights = [], []
    factors = target_weights if target_weights is not None else [1.0]*len(targets)
    if len(factors) != len(targets):
        raise ValueError("Provide one fixed weight per target")
    for constraint, target, factor in zip(refined.constraints, targets, factors):
        ys, ws = SciModel._prepare_data(constraint.cond.outputs, [target],
                                       np.ones(count), count, 0.0)
        truth.extend(ys)
        weights.extend([value*factor for value in ws])
    optimizer = ScipyOptimizer(refined.model)
    parameters = [variable for layer in optimizer._layers for variable in layer.weights if variable.trainable]
    # train_on_batch reports metric accumulators, which can round a float64
    # objective to float32. Evaluate the unreduced graph loss and its matching
    # derivatives directly so tight L-BFGS tolerances remain meaningful.
    desired_tensors = [K.placeholder(shape=(None, 1), dtype="float64") for _ in truth]
    weight_tensors = [K.placeholder(shape=(None,), dtype="float64") for _ in weights]
    feed = list(refined.model.inputs)+desired_tensors+weight_tensors
    components = [K.mean(K.mean(K.square(K.cast(output, "float64")-K.cast(target, "float64")), axis=-1)
                         *K.cast(weight, "float64"))
                  for output, target, weight in zip(refined.model.outputs, desired_tensors, weight_tensors)]
    loss_tensor = sum(components)
    evaluate = K.function(feed, [loss_tensor]+K.gradients(loss_tensor, parameters))
    values = arrays+truth+weights
    state = {"epoch": 0}
    started = time.perf_counter()
    initial = optimizer._collect_weights()
    best = {"loss": float("inf"), "weights": initial.copy()}
    history = []
    calls = [0]

    def objective(vector):
        optimizer._update_weights(vector)
        result = evaluate(values)
        loss = float(result[0])
        gradient = np.concatenate([value.ravel() for value in result[1:]])
        optimizer._cost = loss
        calls[0] += 1
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Nonfinite L-BFGS objective or gradient")
        if loss < best["loss"]:
            best.update(loss=float(loss), weights=vector.copy())
        return float(loss), gradient

    initial_loss, initial_gradient = objective(initial)
    gradient_error = None
    if verify_gradient:
        direction = np.random.RandomState(19).normal(size=len(initial))
        direction /= np.linalg.norm(direction)
        # TF 2.2 CPU activation kernels lose finite-difference accuracy at
        # very small perturbations; 1e-3 keeps rounding below truncation error.
        delta = 1e-3
        plus, _ = objective(initial+delta*direction)
        minus, _ = objective(initial-delta*direction)
        numerical = (plus-minus)/(2*delta)
        analytic = float(np.dot(initial_gradient, direction))
        gradient_error = abs(numerical-analytic)/max(abs(numerical), abs(analytic), 1e-8)
        if gradient_error > 1e-4:
            raise AssertionError("Objective/gradient normalization mismatch: %.3e" % gradient_error)

    def callback(vector):
        state["epoch"] += 1
        row = {"iteration": state["epoch"], "loss": float(optimizer._cost),
               "seconds": time.perf_counter()-started}
        history.append(row)
        if state["epoch"] % log_every == 0:
            print("L-BFGS step=%d loss=%.6e elapsed=%.1fs" % (
                row["iteration"], row["loss"], row["seconds"]), flush=True)

    result = minimize(objective, initial, method="L-BFGS-B", jac=True, callback=callback,
                      options={"maxiter": iterations, "maxcor": 30, "maxls": 40,
                               "ftol": ftol, "gtol": gtol})
    optimizer._update_weights(best["weights"])
    report = {"initial_loss": initial_loss, "final_loss": best["loss"],
              "iterations": int(result.nit), "function_evaluations": calls[0],
              "seconds": time.perf_counter()-started,
              "success": bool(result.success), "termination": str(result.message),
              "sample_count": count, "ftol": ftol, "gtol": gtol, "history": history,
              "objective_precision": "float64 MSE reduction and matching graph gradients",
              "parameter_precision": str(parameters[0].dtype.name)}
    if gradient_error is not None:
        report["directional_gradient_relative_error"] = gradient_error
    print("Refinement: loss %.6e -> %.6e; %d steps in %.1fs (%s)" % (
        report["initial_loss"], report["final_loss"], report["iterations"],
        report["seconds"], report["termination"]), flush=True)
    return refined, report
