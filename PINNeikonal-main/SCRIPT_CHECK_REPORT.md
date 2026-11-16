# Thesis notebook fixes and verification

Updated on 2026-11-02.

**The identified notebook source errors have been fixed in all five notebooks.** Local setup, syntax, and referenced-file checks pass. Full neural-network execution still requires the documented legacy environment, which could not be installed in this session.

## Changes applied

| File | Changes |
| --- | --- |
| `codes/script1.ipynb` | Local setup, dependency checks, corrected `linewidth`, accurate label for the source factor tau |
| `codes/script2.ipynb` | Same common fixes; GPU information uses TensorFlow instead of a shell command |
| `codes/script3.ipynb` | Same common fixes; evaluation inputs now follow the model's `pz0`, `px0` order |
| `codes/script4.ipynb` | Same common fixes; directly loads the supplied `vzcoarse.npy` and `vxcoarse.npy`, checks their shapes, and reports GPU information through TensorFlow |
| `codes/script5.ipynb` | Local setup, dependency checks, corrected `linewidth`, accurate label for the source factor tau |
| `codes/notebook_setup.py` | Shared local workspace preparation, input-file checks, output-directory creation, and dependency/version diagnostics |
| `requirements-legacy.txt` | Pinned Python 3.8 dependencies for the original TensorFlow 2.2.0 / SciAnn 0.5.4.0 implementation |
| `README.md` | PowerShell environment setup and notebook launch instructions |

All Google Drive mounts, hard-coded Colab working directories, inline package installations, and shell-only notebook commands were replaced. The notebooks select the `PINNeikonal (Python 3.8)` kernel described in the README.

The equations, network architectures, loss definitions, random seeds, and training settings match the archived originals. Existing input data, figure PDFs, and model-weight files were preserved. Notebook outputs and execution counts were cleared so historical results are not displayed as executions of the edited code.

## Verification completed

- All five edited notebooks parse as JSON.
- All 106 notebook code cells compile as ordinary Python; no notebook magic conversion is needed.
- The shared setup module compiles.
- The actual path-discovery and workspace-preparation cells executed successfully for all five notebooks from the repository root, `codes`, and a nested input-data directory: 15 successful setup runs.
- All input/model file references in the edited code exist, and their output directories exist.
- `script4` now loads the existing `(201, 201)` coarse velocity arrays without the erroneous `[::50, ::50]` downsampling. Its loader includes a matching-shape check.
- `script3`'s evaluation input order matches the order used for its model and training data.
- The dependency checker reports the current Python/dependency mismatch with setup instructions. Its compatible-version and incompatible-version branches were additionally checked using simulated package metadata; those checks do not establish that scientific packages were installed or imported.
- Comparison against the archived notebooks confirms that solver and training cells are unchanged apart from the documented corrections.
- The original-notebook ZIP backup passed its integrity check.

## Original files

The original notebooks, including their saved execution outputs, and the original check report are archived in:

`backups/notebooks-before-local-fixes-20261102-144913.zip`

## Remaining environment requirement

The available usable interpreter is Python 3.14.6. NumPy, pandas, Matplotlib, SciPy, TensorFlow, SciAnn, Keras, and h5py are absent. TensorFlow 2.2.0 cannot run under Python 3.14. The registered Python 3.12 interpreter could not be launched because access was denied.

Attempts to access PyPI and the Python 3.8 download server were blocked by the session's network restrictions (`WinError 10013`). Consequently, the pinned environment could not be installed and its package resolution remains unverified here.

Follow **Running the notebooks locally** in `README.md` to create the Python 3.8 environment, install `requirements-legacy.txt`, run `pip check`, register the notebook kernel, and run the notebooks in order.

Full SciAnn/TensorFlow imports, live model-weight loading, plotting, training convergence, runtime, and fresh PINN accuracy measurements remain unverified. Syntax and setup checks do not establish successful training.
