# Enhanced original 2D PINNeik results

Measured on 2026-11-16 with Python 3.8.10, TensorFlow 2.2.0, SciAnn 0.5.4.0 and CPU execution.
Domain, velocity functions, source locations, grid spacing and the 10 x 20 network match the original cases.

| Case | Baseline relative L2 | Enhanced relative L2 | Relative L2 reduction | Baseline max error (s) | Enhanced max error (s) |
|---|---:|---:|---:|---:|---:|
| script1 | 0.000531845446 | 3.94478401e-05 | 92.58% | 0.00051823618 | 3.01320281e-05 |
| script3 | 0.000563703533 | 0.000238481943 | 57.69% | 0.00202135952 | 0.000523298351 |

Script 1 starts from the existing `vofz_model-end.hdf5` checkpoint. Its regenerated baseline matches the previously saved relative L2 exactly. The archived old notebook reports relative L2 0.000387721369; the refined result is also better than that archived value.

Script 3 uses a freshly rerun 65-epoch baseline with the original Adam settings and source lattice. Its exact previous in-memory neural field was not saved, so the comparison uses the retained baseline checkpoint from this run. Evaluation uses the original held-out source (x,z)=(1.7,4.2) km on the original 121 x 121 grid.

The original script 3 uses analytic tau labels at all 16 training sources in addition to the PDE. Refinement retains that supervision and does not add labels for the held-out source. Script 1 remains physics/source-only.

Script 1 was refined on all 10,201 grid points with the source masked out of the PDE, for 600 + 200 L-BFGS iterations. Script 3 used a fixed source-balanced set of 16,000 points (1,000 per source), for 600 + 400 iterations. Additional optimization cost is recorded per stage in metrics.json; these are accuracy improvements, not measured speedups.

The enhancement helper computes a float64 MSE reduction and matching graph gradients, uses explicit optimizer tolerances, and retains the lowest finite training objective. The enhanced workflow restores seeds after `clear_session`: verified graph seed 1234 becomes None on clearing and is restored to 1234 before layer creation.

Validation: two refinement unit tests passed, including a directional finite-difference gradient check and balanced sampling. Enhanced notebook execution is recorded in verification.json.

These comparisons are single trained runs; they do not establish a multi-seed generalization result.

Open `codes/script1_enhanced.ipynb` or `codes/script3_enhanced.ipynb` to load the supplied enhanced weights and view actual predictions. Each case folder includes `model.hdf5`, `T_pred_enhanced.npy`, `T_pred_baseline.npy`, `traveltimes.npz`, `metrics.json`, and PNG/PDF error comparisons and contours. Existing notebooks, weights and input data are unchanged.
