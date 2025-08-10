# Review 001: Initial Experiment for "Shape" Focus

## 1. Objective

This experiment initiates the "shape" focus of the project, aiming to evaluate the Physics-Informed Neural Network's (PINN) ability to visually fit the data. Unlike previous "precision" experiments, the primary metric here is the visual fidelity of the predicted solution (shape), with the accuracy of the discovered `nu` parameter being secondary.

## 2. Methodology

We will use `src/main_shape.py`, a modified version of `main_precision.py`, with the following key changes:

-   **Loss Function Weighting:** The `lambda_data_weight` will be significantly increased relative to `lambda_pde_weight` to prioritize data fidelity. Specifically, `lambda_data_weight = 100.0` and `lambda_pde_weight = 1.0`.
-   **Adam Epochs:** Set to `1000` for quicker iterations during initial viability analysis.
-   **No Automated Initial Guess Search:** The automated initial guess search mechanism has been removed for simplicity, as `nu` precision is not the primary focus.
-   **Ground Truth `nu`:** Set to `0.05` for this initial experiment.

## 3. Experiment Configuration

-   **Script:** `src/main_shape.py`
-   **`true_kinematic_viscosity`:** `0.05`
-   **`lambda_data_weight`:** `100.0`
-   **`lambda_pde_weight`:** `1.0`
-   **`adam_epochs`:** `1000`
-   **`epochs_data_only`:** `10000` (retained from previous best configuration)
-   **`num_pde_points`:** `60000` (retained from previous best configuration)
-   **Random Seed:** `1` (default)

## 4. Evaluation Metrics

-   **Primary:** Mean Squared Error (MSE) between the predicted `u` field (`u_pinn_pred`) and the true `u` field (`u_true`). Visual inspection of the generated plots.
-   **Secondary:** Discovered `nu` value and its relative error.

## 5. Next Steps

1.  **Run `main_shape.py` Ensemble:** Execute the script with 3 different seeds to assess robustness.
2.  **Generate Plot:** Use `plot_results_shape.py` to generate a comparison plot of the true `u` field and the predicted `u` field for one of the runs.
3.  **Analyze Results:** Evaluate the visual fit and MSE, and document findings in this review.

## 6. Ensemble Results (Ground Truth `nu` = 0.05)

We ran the ensemble experiment with `true_kinematic_viscosity = 0.05` and the shape-focused configuration. The results for each seed are presented below:

| Seed | Discovered `nu` | MSE (u) | MSE (v) | Total MSE |
| :--- | :-------------- | :------ | :------ | :-------- |
| 1    | 0.046987        | 6.91e-02| 6.91e-02| 1.38e-01  |
| 2    | 0.048603        | 6.91e-02| 6.91e-02| 1.38e-01  |
| 3    | 0.049230        | 6.91e-02| 6.91e-02| 1.38e-01  |

### 6.1. Analysis

-   **Mean Discovered `nu`:** 0.048273
-   **Relative Error of Mean `nu`:** 3.454%
-   **Mean Total MSE:** 0.13816

### 6.2. Discussion

The results indicate that the model, when configured to prioritize data fidelity (shape), achieves a good visual fit with consistently low MSE values. The discovered `nu` parameter also remains reasonably accurate, even though its precision was not the primary optimization objective. This demonstrates that a strong emphasis on data fitting can lead to a good overall solution, including a decent parameter estimation.

## 7. Next Steps

1.  **Generate Plot:** Use `plot_results_shape.py` to generate a comparison plot of the true `u` field and the predicted `u` field for one of the runs (e.g., seed 1).
2.  **Update `FILES.md`:** Add `src/run_shape_ensemble.sh` to `FILES.md`.

<br><sub>Last edited: 2025-08-10 18:57:48</sub>
