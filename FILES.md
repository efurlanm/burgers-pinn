# Project File Structure

This document outlines the directory structure and the purpose of each key file in the project.

## Directories

-   **`sources/`**: Contains all source code. All scripts have been standardized to English.
-   **`results/`**: The designated directory for all output data and plots from program executions.
-   **`logs/`**: Contains log files with detailed output from script executions.
-   **`reviews/`**: Contains markdown files documenting the research process and experimental results.
    -   **`plateau/`**: The active directory for reviews related to the `ReduceLROnPlateau` experiment.
    -   **`precision/`**: The active directory for reviews related to the current focus on improving the precision of `nu` discovery.
    -   **`lbfgsb-scipy/`**: Archive of reviews from the initial SciPy-based implementation.
    -   **`lbfgsb-tfp/`**: Archive of reviews from the TensorFlow Probability implementation.
-   **`papers/`**: Contains reference papers and literature for the project.
-   **`aux#/`**: Auxiliary directory for drafts and temporary files. (Ignored in processing).
-   **`.gemini/`**: Contains configuration files for the Gemini CLI. (Ignored in processing).

## Files

### Primary Scripts

-   **`sources/main_plateau.py`**: **(New)** The primary script for the "plateau" experiment. It implements a **two-stage Parametric Physics-Informed Neural Network (PINN)** with `ReduceLROnPlateau` learning rate scheduling.
    -   **Stage 1**: Trains a generalizable surrogate model to learn the solution $u(x, y, t, \nu)$ across a range of $\nu$ values, incorporating `ReduceLROnPlateau` during Adam optimization.
    -   **Stage 2**: Solves the inverse problem by discovering a specific unknown $\nu$ using the pre-trained Stage 1 model.
    -   **Framework**: Uses TensorFlow 1.x for the model and a hybrid Adam/SciPy-L-BFGS-B optimizer.
    -   **Data Generation**: Includes an internal TensorFlow-based Finite Difference Method (FDM) solver to generate ground truth data for both stages.
    -   **Inputs**: None. All parameters are defined within the script.
    -   **Outputs**: A `.npz` data file (e.g., `results/parametric_inverse_results_nu_0.05_s1_epochs_5000_s2_epochs_5000_seed_1.npz`) containing predictions for various `nu` values from Stage 1, the discovered `nu` from Stage 2, and performance metrics. It also generates a log of the training process.

-   **`sources/main_prmtrc.py`**: **(Reference)** The primary script for the parametric generalization research. It implements a **two-stage Parametric Physics-Informed Neural Network (PINN)**.
    -   **Stage 1**: Trains a generalizable surrogate model to learn the solution $u(x, y, t, \nu)$ across a range of $\nu$ values.
    -   **Stage 2**: Solves the inverse problem by discovering a specific unknown $\nu$ using the pre-trained Stage 1 model.
    -   **Framework**: Uses TensorFlow 1.x for the model and a hybrid Adam/SciPy-L-BFGS-B optimizer.
    -   **Data Generation**: Includes an internal TensorFlow-based Finite Difference Method (FDM) solver to generate ground truth data for both stages.
    -   **Inputs**: None. All parameters are defined within the script.
    -   **Outputs**: A `.npz` data file (e.g., `results/parametric_inverse_results_nu_0.05_s1_epochs_100_s2_epochs_5000_seed_1.npz`) containing predictions for various `nu` values from Stage 1, the discovered `nu` from Stage 2, and performance metrics. It also generates a log of the training process.

-   **`sources/main_precision.py`**: **(Reference)** The original primary script for `nu` discovery. Kept for reference purposes.


-   **`sources/main_shape.py`**: **(New)** The primary script for the "shape" focus. It implements a Physics-Informed Neural Network (PINN) to visually fit the data, prioritizing data fidelity over parameter precision.
    -   **Framework**: Uses TensorFlow 1.x for the model and a hybrid Adam/SciPy-L-BFGS-B optimizer.
    -   **Data Generation**: Includes an internal TensorFlow-based Finite Difference Method (FDM) solver to generate ground truth data.
    -   **Inputs**: None. All parameters are defined within the script.
    -   **Outputs**: A `.npz` data file (e.g., `results/shape_results_data_100.0_pde_1.0_epochs_1000_pdepoints_60000_seed_1.npz`) and a log of the training process.

-   **`sources/plot_results.py`**: A utility script to visualize the output from `main_precision.py`. It generates a side-by-side comparison plot of the ground truth and the PINN-predicted solutions and calculates key error metrics.
    -   **Inputs**: A `.npz` file from the `results/` directory. Can be passed as a command-line argument.
    -   **Outputs**: A `.jpg` image file saved in the `results/` directory.

-   **`sources/plot_results_shape.py`**: A utility script to visualize the output from `main_shape.py`. It generates a side-by-side comparison plot of the ground truth and the PINN-predicted solutions and calculates key error metrics.
    -   **Inputs**: A `.npz` file from the `results/` directory. Can be passed as a command-line argument.
    -   **Outputs**: A `.jpg` image file saved in the `results/` directory.

-   **`sources/plot_main_figure.py`**: A utility script to generate the main comparison figure for the manuscript, combining results from both "precision" and "shape" experiments.
    -   **Inputs**: `.npz` files from the `results/` directory.
    -   **Outputs**: A `.jpg` image file saved in the `results/` directory.

### Supporting and Reference Scripts

-   **`sources/finite_difference.py`**: A standalone, NumPy-based FDM solver for the 2D Burgers' equation. It uses a different numerical scheme (upwind differences) than the internal solver in `main_precision.py` (central differences). It serves as a reference and is not directly used by the main training pipeline.
    -   **Inputs**: None.
    -   **Outputs**: `results/burgers2d_diff_results.jpg`.
-   **`results/burgers2d_precision_comparison.jpg`**: A comparison plot showing Ground Truth, PINN SciPy, and PINN Precision solutions for the 2D Burgers' equation, including discovered `nu` values and MSEs.
-   **`results/burgers2d_shape_comparison.jpg`**: A comparison plot showing Ground Truth and PINN Predicted solutions for the 2D Burgers' equation, focusing on visual fit, including discovered `nu` values and MSEs.
-   **`results/fig_main_comparison.jpg`**: The main comparison figure used in the manuscript, generated by `sources/plot_main_figure.py`.

-   **`sources/main_scipy.py`**: **(Reference)** An earlier version of the PINN implementation. Kept for reference purposes.

-   **`sources/main_scipy_nodgpinn.py`**: **(Snapshot)** A snapshot of `main_scipy.py` before porting DG-PINN and other improvements. Used for comparison and reference.

-   **`sources/main_tfp.py`**: **(Reference)** An earlier implementation using TensorFlow Probability. Kept for reference purposes.

-   **`sources/run_ensemble.sh`**: A shell script to run ensemble experiments for the "precision" part, iterating through different seeds and collecting results.

-   **`sources/run_shape_ensemble.sh`**: A shell script to run ensemble experiments for the "shape" part, iterating through different seeds and collecting results.

### Documentation

-   **`DRAFT.md`**: **(New)** A comprehensive summary of all work done, detailing the tuning roadmap, experimental findings, and key insights. This document serves as the primary source for updating the final manuscript.
-   **`FILES.md`**: This file.

### LaTeX

-   **`latex/manuscript.tex`**: The LaTeX source file for the academic paper.
-   **`latex/thebibliography.bib`**: The BibTeX bibliography for the manuscript.

### Reviews

-   **`reviews/precision/review-001.md`**: Initial setup and baseline experiment for the 'precision' focus.
-   **`reviews/precision/review-003.md`**: Analysis of the impact of loss weighting on `nu` discovery.
-   **`reviews/precision/review-004.md`**: Further experiments with different loss weighting schemes.
-   **`reviews/precision/review-005.md`**: Investigation into the effect of the number of training epochs.
-   **`reviews/precision/review-006.md`**: Deeper analysis of epoch variation on precision.
-   **`reviews/precision/review-007.md`**: Study on the influence of the number of PDE collocation points.
-   **`reviews/precision/review-009.md`**: Introduction of Data-Guided PINN (DG-PINN) with pre-training.
-   **`reviews/precision/review-011.md`**: Tuning of Adam epochs in the DG-PINN framework.
-   **`reviews/precision/review-012.md`**: Experimenting with the number of hidden layers in the neural network.
-   **`reviews/precision/review-013.md`**: Adjusting the number of neurons per hidden layer.
-   **`reviews/precision/review-014.md`**: Analysis of a learning rate schedule.
-   **`reviews/precision/review-015.md`**: Testing adaptive weights for the loss function.
-   **`reviews/precision/review-016.md`**: Increasing PDE points to 50,000 in the DG-PINN model.
-   **`reviews/precision/review-017.md`**: Increasing PDE points to 60,000.
-   **`reviews/precision/review-018.md`**: Analysis of the best result (0.044% error) with 60,000 PDE points.
-   **`reviews/precision/review-019.md`**: Experiment with 70,000 PDE points, showing a decrease in precision.
-   **`reviews/precision/review-020.md`**: Verification of the 0.044% relative error result.
-   **`reviews/precision/review-021.md`**: Experimental Design using Nested/Hierarchical Sampling.
-   **`reviews/precision/review-022.md`**: Improving Robustness with Ensemble Averaging.
-   **`reviews/precision/review-023.md`**: Generalizability Experiment with Lower `nu` (0.01).
-   **`reviews/precision/review-024.md`**: Generalizability Experiment with Higher `nu` (0.1).
-   **`reviews/precision/review-025.md`**: Generalizability with Lower `nu` (0.01) and Corrected Initial Guess.
-   **`reviews/precision/review-026.md`**: Automated Initial Guess Search for `nu`.
-   **`reviews/precision/review-027.md`**: Comprehensive Summary of Robustness and Generalizability Experiments.
-   **`reviews/precision/review-028.md`**: Harmonizing Methodologies: Porting Improvements to `main_scipy.py`.
-   **`reviews/precision/review-029.md`**: Attempt to Port Improvements to `main_scipy.py`.
-   **`reviews/plateau/review-001.md`**: Documentation of the implementation and verification of `ReduceLROnPlateau` in `main_plateau.py`.

<br><sub>Last edited: 2025-08-22 05:05:25</sub>
