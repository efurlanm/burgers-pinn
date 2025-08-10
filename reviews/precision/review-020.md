# Review 20: Verification of the 0.044% Relative Error Result

## 1. Objective

The primary objective of this step was to verify the reproducibility of the previously achieved best result—a 0.044% relative error in the discovery of the kinematic viscosity (`nu`).

## 2. Methodology

The experiment was rerun using the exact same configuration that produced the best result, as documented in Review 018 and confirmed in Review 019. The `src/main_precision.py` script was executed without any modifications to its parameters.

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 60,000
-   **Adam Epochs (Phase 2):** 2,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_verify_0.044_perc_error.txt`.

-   **Final Discovered `nu`:** 0.049978
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.049978 - 0.05| = 0.000022`
-   **Relative Error:** `(0.000022 / 0.05) * 100% = 0.044%`

The result was successfully reproduced.

### MSE Values

-   Prediction MSE (u): 6.874458e-02
-   Prediction MSE (v): 6.875561e-02
-   Total Prediction MSE (u+v): 1.375002e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.95 seconds
-   Model Initialization Duration: 3.60 seconds
-   Data-Only Pre-training Duration: 21.12 seconds
-   Adam Training Duration: 308.97 seconds (~5.15 minutes)
-   L-BFGS-B Training Duration: 98.32 seconds (~1.64 minutes)
-   **Total Execution Duration:** 437.49 seconds (~7.29 minutes)

## 5. Discussion

The verification was successful. The 0.044% relative error is a reproducible result with the current configuration. This provides confidence in the stability of the model and the chosen hyperparameters.

## 6. Next Steps

Now that the best result has been verified, the next steps will focus on evaluating the robustness and generalizability of the model, as outlined in the project's main objective.

-   **Robustness Check (Varying Random Seed):** Run the best configuration with different random seeds to ensure the result is not a statistical anomaly.
-   **Generalizability Check (Varying `nu` parameter):** Test the model's ability to discover different `nu` values.

<br><sub>Last edited: 2025-08-10 10:42:43</sub>
