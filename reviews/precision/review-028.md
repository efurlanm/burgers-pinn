# Review 28: Harmonizing Methodologies: Porting Improvements to `main_scipy.py`

## 1. Objective

To ensure a fair and robust comparison between the "precision" and "lbfgsb-scipy" methodologies, we will port key improvements, particularly the Data-Guided PINN (DG-PINN) approach, from `main_precision.py` to `main_scipy.py`. This harmonization will provide a consistent baseline for evaluating the specific contributions of each approach.

## 2. Identified Improvements to Port

Based on the successful developments in the "precision" branch, the following improvements will be integrated into `main_scipy.py`:

1.  **Data-Guided PINN (DG-PINN):** This involves adding a data-only pre-training phase to the optimization process. This has shown to significantly improve the model's ability to learn the underlying physics.
2.  **Automated Initial Guess Search for `nu`:** The mechanism to automatically find a good initial guess for the kinematic viscosity (`nu`) will be integrated. This enhances the model's robustness and reduces reliance on manual tuning.
3.  **Command-Line Argument for Random Seed:** The random seed will be configurable via a command-line argument, allowing for systematic robustness checks and ensemble runs.
4.  **Output Filename with Seed:** The output `.npz` filename will include the random seed, facilitating better tracking and organization of experimental results.
5.  **Detailed HPC Performance Metrics:** Comprehensive timing metrics for different phases of the training process will be added, consistent with the focus on High-Performance Computing analysis.

## 3. Phased Implementation Plan

To manage complexity, the integration will be performed in phases:

### Phase 1: Integrate Data-Guided PINN (DG-PINN)

-   Add the `train_data_only` method to the `PINN_Burgers2D` class in `main_scipy.py`.
-   Modify the `train` method in `main_scipy.py` to include a call to `train_data_only` before the main Adam optimization.

### Phase 2: Integrate Automated Initial Guess Search

-   Add the `find_best_initial_nu` method to the `PINN_Burgers2D` class in `main_scipy.py`.
-   Modify the `train` method to call `find_best_initial_nu` and use its output as the initial guess for `nu`.

### Phase 3: Argument Parsing, Output Filename, and HPC Metrics

-   Add `argparse` for `seed` and other relevant parameters.
-   Modify the output `.npz` filename to include the seed.
-   Add detailed timing metrics throughout the script.

## 4. Next Steps

We will now proceed with **Phase 1: Integrating Data-Guided PINN (DG-PINN)** into `main_scipy.py`. This involves adding the `train_data_only` method and modifying the `train` method.

<br><sub>Last edited: 2025-08-10 17:36:48</sub>
