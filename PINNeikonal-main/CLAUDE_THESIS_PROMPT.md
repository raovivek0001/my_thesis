# Claude prompt: thesis report and presentation

Copy the prompt below into Claude, or attach this file together with the project files. Claude needs access to the listed source files and figures to produce a report grounded in the actual results.

---

Act as an academic thesis writer and scientific presentation designer. Create a thesis report and an editable PowerPoint presentation for my PINNeikonal project, using the actual code, plots, trained models, and measured results in the supplied repository.

Suggested title:

**Improving Two-Dimensional Eikonal Traveltime Prediction Using Physics-Informed Neural Networks**

## Primary scope

Focus on the original **2D PINNeik implementation** and the enhanced results for scripts 1 and 3.

Explain the eikonal equation, traveltime factorization, neural network, locally adaptive activations, training process, and additional L-BFGS refinement.

The repository also contains `pinneik3d/`, but its neural-network performance experiments were not completed. Mention it only as a proposed extension or future work. Do not present its mathematical/FSM checks as trained PINN results.

If a required source file is unavailable, identify the missing evidence and mark the affected content as pending. Do not invent results to complete a section.

## Source material

Read these files before writing:

- `README.md`
- The supplied PINNeik research paper: `PINNeik- Eikonal solution using physics-informed neural networks.pdf`
- `codes/script1.ipynb`
- `codes/script2.ipynb`
- `codes/script3.ipynb`
- `codes/script1_enhanced.ipynb`
- `codes/script3_enhanced.ipynb`
- `codes/accuracy_training.py`
- `codes/enhance_results.py`
- `codes/results/enhanced_2d/summary.md`
- `codes/results/enhanced_2d/verification.json`
- `codes/results/enhanced_2d/script1/metrics.json`
- `codes/results/enhanced_2d/script3/metrics.json`
- `codes/results/enhanced_2d/script1/traveltimes.npz`
- `codes/results/enhanced_2d/script3/traveltimes.npz`

Each case folder also contains `T_pred_baseline.npy`, `T_pred_enhanced.npy`, and trained weights in `model.hdf5`.

Use the following measured results as a cross-check. The saved metrics files are authoritative.

| Measurement | Script 1 | Script 3 |
|---|---:|---:|
| Baseline relative L2 error | 0.000531845446 | 0.000563703533 |
| Enhanced relative L2 error | 0.0000394478401 | 0.000238481943 |
| Relative L2 error reduction | 92.6% | 57.7% |
| Baseline maximum absolute error (s) | 0.000518236180 | 0.002021359525 |
| Enhanced maximum absolute error (s) | 0.0000301320281 | 0.000523298351 |

Distinguish archived old-code results from the regenerated baselines used in these comparisons. Script 1's archived relative L2 error was **0.000387721369**. Do not invent missing archived numerical results for script 3.

Script 1 uses a 101 × 101 evaluation grid. Script 3 uses a 121 × 121 evaluation grid and the original held-out source `(x, z) = (1.7, 4.2)` km. Confirm geometry and units from the source files.

## Figures to use

Select useful figures from:

- `codes/results/enhanced_2d/script1/error_comparison.png`
- `codes/results/enhanced_2d/script1/contours.png`
- `codes/results/enhanced_2d/script3/error_comparison.png`
- `codes/results/enhanced_2d/script3/contours.png`
- The corresponding PDF versions
- Velocity-model and source-location plots in `codes/figs/`
- Actual loss and refinement histories in the notebooks and metrics files

Prefer original PDF figures for the report when suitable. Preserve axis labels, units, legends, and color scales.

For baseline/enhanced error comparisons, use identical color limits. Explain what each figure demonstrates and reference it in the surrounding discussion.

You may generate additional charts from the saved numerical results, including:

- Relative L2 error comparisons
- Maximum absolute error comparisons
- Error-reduction percentages
- Refinement loss curves

Do not fabricate data or redraw figures with invented values. Keep original figures available alongside any reformatted versions.

## Deliverable 1: thesis report

Prepare a complete academic draft of approximately **35–50 pages**, excluding front matter and appendices. Prioritize substance over padding.

Include:

1. Title page with placeholders for student, supervisor, institution, department, degree, and submission date
2. Abstract and keywords
3. Table of contents, list of figures, list of tables, and notation
4. Introduction and motivation
5. Literature review covering eikonal solvers, fast sweeping/fast marching, PINNs, and PINNeik
6. Mathematical formulation
7. Original implementation and experimental setup
8. Identified accuracy and reproducibility issues
9. Enhancement methodology
10. Results and quantitative comparisons
11. Discussion and limitations
12. Conclusions and future work
13. Verified references
14. Appendices containing configuration details, reproduction commands, and validation evidence

Explain the verified enhancements:

- Restoring random seeds after clearing the TensorFlow session
- Excluding the source point from the singular PDE residual
- Additional fixed-data L-BFGS optimization
- Explicit convergence tolerances
- Float64 MSE reduction and matching graph gradients
- Improved collocation coverage for script 1
- Source-balanced refinement sampling for script 3

Use the exact **2D equations and loss definitions** found in the notebooks. Do not substitute the separate 3D implementation's loss formulation.

Explain the relationship between the neural-network output `tau`, the background factor `T0`, and the reconstructed traveltime `T = T0 * tau`. Derive the factored residual and explain the treatment of the source singularity. Include definitions of relative L2, mean absolute error, and maximum absolute error.

Clearly state that:

- Script 1 uses physics and the source condition.
- Script 3 already uses analytic tau labels at its 16 training sources in addition to the PDE.
- Held-out source labels were not used for training.
- Reported improvements come from individual measured runs, not a multi-seed statistical study.
- Additional refinement requires additional computation; accuracy improvement does not establish a speedup.
- GPU performance and completed 3D PINN accuracy are not demonstrated.
- A mathematical/FSM validation result is different from the accuracy of a trained neural network.

Discuss how changes in initialization and convergence can affect regenerated outputs. Explain the verified TensorFlow seed-reset issue without attributing every historical discrepancy to it.

Write in clear academic English. Support conclusions with equations, tables, figures, and measured evidence. Verify citations and avoid invented references, timings, experiments, or claims of novelty. Use the supplied PINNeik paper and verified literature to distinguish established techniques from this project's implementation and evaluation work.

## Deliverable 2: PowerPoint presentation

Create an **editable presentation of approximately 20 slides** for a **15–20-minute thesis presentation**.

Suggested sequence:

1. Title
2. Motivation and application
3. Research problem
4. Objectives
5. Relevant prior work
6. Eikonal equation
7. Traveltime factorization
8. PINN architecture and loss
9. Original experimental setup
10. Accuracy/reproducibility issues
11. Enhanced workflow
12. Script 1 setup
13. Script 1 error comparison
14. Script 1 quantitative results
15. Script 3 setup
16. Script 3 error and contour comparisons
17. Overall results
18. Limitations and validation
19. Conclusions and future work
20. Questions

Use a clean academic design, readable equations, large plot labels, and concise slide text. Make the numerical improvements prominent while explaining their experimental scope. Avoid dense paragraphs and screenshots of code as substitutes for explanations.

Add speaker notes explaining each slide, including likely examiner questions about:

- Why PINNs are useful for traveltime prediction
- How the source singularity is handled
- Why L-BFGS improves the fit
- Whether script 3 is entirely physics-only
- Fairness of baseline comparisons
- Computational cost and generalization
- What remains unverified

## Outputs and final checks

Save the deliverables inside a new `thesis_deliverables/` directory without overwriting source files:

- `Thesis_Report.docx`
- `Thesis_Report.pdf`
- `Thesis_Presentation.pptx`
- `Thesis_Presentation.pdf`, if supported
- A `figures/` folder containing the selected figures
- A short source map linking every figure and result table to its repository source

If a requested format cannot be generated, provide an editable alternative and explain the limitation. If working in a chat environment without repository write access, provide downloadable deliverables instead.

Before finishing, check that:

- Every numerical result matches its source.
- Captions and units are correct.
- Figures remain readable in both documents.
- Equations, notation, references, and figure numbering are consistent.
- No unverified 3D, GPU, speedup, or generalization claims appear.
- Missing personal or institutional information remains clearly marked as a placeholder.
- Both the report and slides explain the actual measured improvements and their limits.

Return the finished files and a short description of their contents, together with any specific evidence or formatting limitations that remain unresolved.
