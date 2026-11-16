"""Regenerate and refine the original 2D results without overwriting baselines.

From the repository root: .venv/Scripts/python.exe codes/enhance_results.py
"""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from accuracy_training import lbfgs_refine, stratified_source_indices

CODES = Path(__file__).resolve().parent


def setup_notebook(name, cells):
    notebook = json.loads((CODES/name).read_text(encoding="utf-8"))
    scope = {}
    for index in cells:
        source = "".join(notebook["cells"][index]["source"])
        if "K.clear_session()" in source:
            # clear_session resets TF's graph seed. Restore it before creating
            # layers so a fresh rerun does not silently change initialization.
            seed = 1234 if name == "script3.ipynb" else 123
            source = source.replace("K.clear_session()", "K.clear_session()\nnp.random.seed(%d)\ntf.random.set_seed(%d)\nrandom.seed(%d)" % (seed, seed, seed))
        exec(compile(source, name+":cell%d" % index, "exec"), scope)
    return scope


def metrics(prediction, reference):
    error = np.asarray(prediction)-np.asarray(reference)
    return {"relative_l2": float(np.linalg.norm(error)/np.linalg.norm(reference)),
            "max_abs_error": float(np.max(np.abs(error))),
            "mean_abs_error": float(np.mean(np.abs(error)))}


def save_comparison(folder, data, baseline, enhanced, reference, fsm, x, z, sx, sz):
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    data.update(baseline=metrics(baseline, reference), enhanced=metrics(enhanced, reference),
                fsm=metrics(fsm, reference), source_x=sx, source_z=sz)
    data["units"] = {"coordinates": "km", "traveltime": "s"}
    data["relative_l2_reduction_percent"] = 100*(1-data["enhanced"]["relative_l2"]/data["baseline"]["relative_l2"])
    (folder/"metrics.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(str(folder/"traveltimes.npz"), x=x, z=z, source_x=sx, source_z=sz,
                        baseline=baseline, enhanced=enhanced, reference=reference, fsm=fsm)
    np.save(str(folder/"T_pred_baseline.npy"), baseline)
    np.save(str(folder/"T_pred_enhanced.npy"), enhanced)
    extent = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]
    limit = max(np.max(np.abs(baseline-reference)), np.max(np.abs(enhanced-reference)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, field, title in zip(axes, (baseline, enhanced), ("Regenerated baseline", "Refined PINN")):
        im = ax.imshow(np.abs(field-reference), extent=extent, cmap="jet", vmin=0, vmax=limit)
        ax.set(title=title, xlabel="Offset (km)", ylabel="Depth (km)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Absolute error (s)")
    for suffix in ("png", "pdf"):
        fig.savefig(str(folder/("error_comparison."+suffix)), bbox_inches="tight", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    levels = np.linspace(0, float(reference.max()), 8)[1:]
    for field, color, style in ((reference, "r", "solid"), (enhanced, "k", "dashed"), (fsm, "b", "dotted")):
        ax.contour(x, z, field, levels=levels, colors=color, linestyles=style)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=c, linestyle=s) for c, s in (("r", "solid"), ("k", "dashed"), ("b", "dotted"))],
              ["Analytic", "Refined PINN", "Fast sweeping"])
    ax.plot(sx, sz, "k*")
    ax.set(xlabel="Offset (km)", ylabel="Depth (km)")
    ax.invert_yaxis()
    for suffix in ("png", "pdf"):
        fig.savefig(str(folder/("contours."+suffix)), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("RESULT", folder.name, "baseline", data["baseline"], "enhanced", data["enhanced"], flush=True)
    return data


def run_script1(iterations, continue_refinement=False):
    scope = setup_notebook("script1.ipynb", [1, 2, 3, 4, 5, 6, 7, 9, 10, 11])
    model, tau = scope["model"], scope["tau"]
    checkpoint = CODES/"models/vofz_model-end.hdf5"
    model.load_weights(str(checkpoint))
    inputs = scope["X_star"]+[scope[k] for k in ("velmodel", "pz0", "px0", "T0")]
    baseline = tau.eval(model, inputs).reshape(scope["Z"].shape)*scope["T0"]
    source_ids = np.flatnonzero(scope["T0"].ravel() == 0)
    pde_ids = np.flatnonzero(scope["T0"].ravel() > 0)
    targets = [(source_ids, np.ones((len(source_ids), 1))),
               (pde_ids, np.zeros((len(pde_ids), 1))), "zeros"]
    print("Script 1 baseline:", metrics(baseline, scope["T_data"]), flush=True)
    folder = CODES/"results/enhanced_2d/script1"
    stages = []
    if continue_refinement:
        previous = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        stages = previous.get("refinement_stages", [previous["refinement"]])
        model.load_weights(str(folder/"model.hdf5"))
    refined, report = lbfgs_refine(model, inputs, targets, iterations=iterations)
    enhanced = tau.eval(refined, inputs).reshape(scope["Z"].shape)*scope["T0"]
    folder.mkdir(parents=True, exist_ok=True)
    refined.save_weights(str(folder/"model.hdf5"))
    return save_comparison(folder, {"case": "script1", "refinement": report,
        "refinement_stages": stages+[report],
        "baseline_checkpoint": str(checkpoint), "archived_relative_l2": 0.00038772136909505557,
        "training_labels": "source condition and physics only; source excluded from singular PDE"},
        baseline, enhanced, scope["T_data"], np.load(CODES/"inputs/vofz/traveltimes/Tcomp.npy"),
        scope["x"], scope["z"], scope["sx"], scope["sz"])


def run_script3(iterations, epochs, per_source, continue_refinement=False):
    scope = setup_notebook("script3.ipynb", [1, 2, 3, 4, 5, 6, 8, 9])
    from tensorflow.python.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
    from tensorflow.python.keras import backend as K
    from sciann.utils.callbacks import EarlyStoppingByLossVal, EarlyStoppingByLearningRate
    folder = CODES/"results/enhanced_2d/script3"
    folder.mkdir(parents=True, exist_ok=True)
    inputs = [scope[k] for k in ("X_star", "Z_star", "pz0_star", "px0_star", "T0_star", "SX_star", "SZ_star", "V_star")]
    baseline_checkpoint = folder/"baseline_model.hdf5"
    if baseline_checkpoint.exists() and epochs == 65:
        scope["model"].load_weights(str(baseline_checkpoint))
        previous_metrics = folder/"metrics.json"
        baseline_seconds = json.loads(previous_metrics.read_text(encoding="utf-8")).get("adam_seconds") if previous_metrics.exists() else None
        print("Reusing measured 65-epoch baseline checkpoint", flush=True)
    else:
        started = time.perf_counter()

        class Progress(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch+1) % 10 == 0:
                    print("Surrogate Adam epoch=%d loss=%.6e elapsed=%.1fs" % (
                        epoch+1, logs["loss"], time.perf_counter()-started), flush=True)

        # Preserve SciAnn's original default schedule when adding progress output.
        K.set_value(scope["model"].model.optimizer.lr, 0.0005)
        callbacks = [Progress(), ReduceLROnPlateau(monitor="loss", factor=0.5,
                     patience=max(10, epochs/10), verbose=1, min_delta=0., min_lr=0.),
                     EarlyStopping(monitor="loss", patience=epochs, min_delta=1e-9),
                     TerminateOnNaN(), EarlyStoppingByLossVal(1e-8), EarlyStoppingByLearningRate(1e-8)]
        scope["model"].train(inputs, scope["target_vals"], batch_size=2000, epochs=epochs,
                            learning_rate=0.0005, verbose=0, callbacks=callbacks)
        baseline_seconds = time.perf_counter()-started
        scope["model"].save_weights(str(baseline_checkpoint))
    # The notebook's original held-out source: np.random.seed(123) gives (x,z)=(1.7,4.2).
    sx, sz = 1.7, 4.2
    X, Z = scope["X"], scope["Z"]
    velocity = scope["v0"]+scope["horgrad"]*X+scope["vergrad"]*Z
    vs = scope["v0"]+scope["horgrad"]*sx+scope["vergrad"]*sz
    t0 = np.hypot(X-sx, Z-sz)/vs
    px = np.divide(X-sx, t0*vs**2, out=np.zeros_like(t0), where=t0 != 0)
    pz = np.divide(Z-sz, t0*vs**2, out=np.zeros_like(t0), where=t0 != 0)
    evaluation = [X, Z, pz, px, t0, np.full_like(X, sx), np.full_like(X, sz), velocity]
    gradient = np.hypot(scope["horgrad"], scope["vergrad"])
    reference = 2*np.arcsinh(gradient*np.hypot(X-sx, Z-sz)/(2*np.sqrt(velocity*vs)))/gradient
    baseline = scope["tau"].eval(scope["model"], evaluation).reshape(Z.shape)*t0
    print("Script 3 baseline:", metrics(baseline, reference), flush=True)
    indices = stratified_source_indices(scope["SX_star"], scope["SZ_star"], per_source)
    fixed_inputs = [value[indices] for value in inputs]
    pde_ids = np.flatnonzero(scope["T0_star"][indices].ravel() > 0)
    fixed_targets = [scope["tau_star"][indices], (pde_ids, np.zeros((len(pde_ids), 1))), "zeros"]
    stages = []
    if continue_refinement:
        previous = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        stages = previous.get("refinement_stages", [previous["refinement"]])
        scope["model"].load_weights(str(folder/"model.hdf5"))
    refined, report = lbfgs_refine(scope["model"], fixed_inputs, fixed_targets, iterations=iterations)
    enhanced = scope["tau"].eval(refined, evaluation).reshape(Z.shape)*t0
    refined.save_weights(str(folder/"model.hdf5"))
    return save_comparison(folder, {"case": "script3", "refinement": report,
        "refinement_stages": stages+[report],
        "adam_epochs": epochs, "adam_seconds": baseline_seconds, "collocations_per_source": per_source,
        "training_labels": "original analytic tau labels at the original 16 training sources; no held-out source labels"},
        baseline, enhanced, reference, np.load(CODES/"inputs/vofxz_surrogate/traveltimes/Tcomp.npy"),
        scope["x"], scope["z"], sx, sz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["script1", "script3", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument("--continue-refinement", action="store_true",
                        help="Continue from the previously saved refined weights, retaining original baseline data")
    args = parser.parse_args()
    os.chdir(str(CODES.parent))
    rows = []
    if args.case in ("script1", "all"):
        rows.append(run_script1(args.iterations, args.continue_refinement))
    if args.case in ("script3", "all"):
        os.chdir(str(CODES.parent))
        rows.append(run_script3(args.iterations, args.epochs, args.per_source, args.continue_refinement))
    root = CODES/"results/enhanced_2d"
    for row in rows:
        (root/(row["case"]+"_summary.json")).write_text(json.dumps(row, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
