# PINNeikonal
Eikonal solution using physics-informed neural networks

The eikonal equation is utilized across a wide spectrum of science and engineering disciplines. In seismology, it regulates seismic wave traveltimes needed for applications like source localization, imaging, and inversion. Several numerical algorithms have been developed over the years to solve the eikonal equation. However, these methods require considerable modifications to incorporate additional physics, such as anisotropy, and may even breakdown for certain complex forms of the eikonal equation, requiring approximation methods. Moreover, they suffer from computational bottleneck when repeated computations are needed for perturbations in the velocity model and/or the source location, particularly in large 3D models. Here, we propose an algorithm to solve the eikonal equation based on the emerging paradigm of physics-informed neural networks (PINNs). By minimizing a loss function formed by imposing the eikonal equation, we train a neural network to output traveltimes that are consistent with the underlying partial differential equation. We observe sufficiently high traveltime accuracy for most applications of interest. We also demonstrate how the proposed algorithm harnesses machine learning techniques like transfer learning and surrogate modeling to speed up traveltime computations for updated velocity models and source locations. Furthermore, we use a locally adaptive activation function and adaptive weighting of the terms in the loss function to improve convergence rate and solution accuracy. We also show the flexibility of the method in incorporating medium anisotropy and free-surface topography compared to conventional methods that require significant algorithmic development. These properties of the proposed PINN eikonal solver are highly desirable in obtaining a flexible and efficient forward modeling engine for seismological applications.

## Libraries
The scripts work with Tensorflow v2.2.0 and SciAnn v0.5.4.0

## Running the notebooks locally

The five notebooks in `codes` now use local input files. They locate the `codes`
directory when launched from the repository root, `codes`, or a directory below
either location, and create their figure directories as needed.

Use **64-bit Python 3.8** for the original TensorFlow/SciAnn implementation.
TensorFlow 2.2.0 cannot be installed into Python 3.12 or 3.14. The pinned
dependencies are in `requirements-legacy.txt`; installing the latest SciAnn,
TensorFlow, or Keras independently may require changes to the solver APIs.

From the repository root, use PowerShell after installing Python 3.8:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_dependencies.ps1
```

The installer creates or reuses the project's `.venv`, installs the pinned
packages, checks their compatibility and imports, and registers the Jupyter
kernel. It needs internet access for package downloads. It stops if Python 3.8
is missing or an existing `.venv` uses another Python version.

To check prerequisites without installing anything, add `-CheckOnly`. If your
Python 3.8 installation is not detected, provide its executable explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_dependencies.ps1 -PythonExe "C:\Python38\python.exe"
```

Alternatively, run the individual setup commands:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "pip<25.1" "setuptools<81" wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-legacy.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pinneikonal --display-name "PINNeikonal (Python 3.8)"
.\.venv\Scripts\python.exe -m notebook codes
```

Select **PINNeikonal (Python 3.8)** and run the cells in order. The setup cells
check the input files and required packages; they do not install packages inside
a running kernel. CPU execution is supported. The GPU information cells report
devices through TensorFlow and do not require the `nvidia-smi` command.

| Notebook | Example |
| --- | --- |
| `script1.ipynb` | Smooth vertical velocity gradient; saves pretrained weights |
| `script2.ipynb` | Transfer learning for a smooth horizontal/vertical gradient |
| `script3.ipynb` | Multiple-source surrogate model |
| `script4.ipynb` | Anisotropy and surface topography; uses the supplied coarse velocity arrays |
| `script5.ipynb` | Marmousi velocity model |

The pretrained `codes/models/vofz_model-end.hdf5` file is supplied for scripts 2,
4, and 5. Training can take several minutes or longer depending on hardware and
the configured epoch count. The setup changes preserve the original equations,
network architectures, and training settings. Fresh training and accuracy
validation remain necessary before using new results in the thesis.


## Enhanced 2D results

The refined versions of scripts 1 and 3 are available in
`codes/script1_enhanced.ipynb` and `codes/script3_enhanced.ipynb`. They load
actually trained checkpoints and display the new predictions. Open them in the
same PINNeikonal Python 3.8 kernel. The original notebooks and input data are
preserved.

Measured comparisons, matching-color-scale plots, model weights, and generated
traveltime arrays are under `codes/results/enhanced_2d/`. See
[`codes/results/enhanced_2d/summary.md`](codes/results/enhanced_2d/summary.md).
`T_pred_enhanced.npy` contains the enhanced 2D traveltime array in seconds;
`traveltimes.npz` also contains the original baseline, analytic reference,
FSM, grid coordinates and source location.

The refinement restores random seeds after clearing the TensorFlow graph,
excludes the source from the singular PDE residual, and adds fixed-data L-BFGS
with explicit convergence tolerances. Script 1 uses physics and the source
condition. Script 3 retains the original analytic tau supervision at its 16
training sources; held-out source labels are not used for training.

To generate fresh refined results from the repository root:

```powershell
.\.venv\Scripts\python.exe codes\enhance_results.py --case script1 --iterations 800
.\.venv\Scripts\python.exe codes\enhance_results.py --case script3 --iterations 1000
.\.venv\Scripts\python.exe -m unittest discover -s codes\tests -v
```

The supplied weights reproduce the saved predictions. Fresh optimization can
give different numerical values, particularly with different hardware or an
initial checkpoint. The measured reports document the runs actually executed.

## Citation
If you find our work useful in your research, please cite:
```
@article{bin2021pinneik,
  title={PINNeik: Eikonal solution using physics-informed neural networks},
  author={bin Waheed, Umair and Haghighat, Ehsan and Alkhalifah, Tariq and Song, Chao and Hao, Qi},
  journal={Computers \& Geosciences},
  pages={104833},
  year={2021},
  publisher={Elsevier}
}
```

## Contact
If you have any questions, please feel free to email the author.
