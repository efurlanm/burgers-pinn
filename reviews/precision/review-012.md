# Review 012: Data-Guided PINN (DG-PINN) with Increased Adam Epochs (5000)

## 1. Experiment Objective

The objective of this experiment was to investigate if further increasing the number of Adam epochs from 2,000 to 5,000 could lead to even better precision for kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   `adam_epochs` was increased from 2,000 to 5,000.

All other parameters remained at the best configuration established in Review 008 (DG-PINN without adaptive weights).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 40,000
-   **Adam Epochs (Phase 2):** 5,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_adam_5000_epochs.txt`.

-   **Final Discovered `nu`:** 0.045967
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.045967 - 0.05| = 0.004033`
-   **Relative Error:** `(0.004033 / 0.05) * 100% = 8.066%`

### MSE Values

-   Prediction MSE (u): 6.884725e-02
-   Prediction MSE (v): 6.887596e-02
-   Total Prediction MSE (u+v): 1.377232e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.64 seconds
-   Model Initialization Duration: 3.63 seconds
-   Data-Only Pre-training Duration: 21.04 seconds
-   Adam Training Duration: 530.97 seconds (~8.85 minutes)
-   L-BFGS-B Training Duration: 28.00 seconds
-   **Total Execution Duration:** 588.38 seconds (~9.81 minutes)

## 5. Discussion and Comparison

Increasing the Adam epochs from 2,000 to 5,000 **did not improve** the precision of `nu` discovery. Instead, the relative error significantly *increased* from 0.542% to 8.066%. The MSE values for u and v predictions also slightly increased.

| Metric                    | DG-PINN with 2k Adam Epochs (Review 011) | DG-PINN with 5k Adam Epochs (Review 012) | Change |
| :------------------------ | :--------------------------------------- | :--------------------------------------- | :----- |
| Discovered `nu`           | 0.050271                                 | 0.045967                                 | Worse |
| Relative Error            | **0.542%**                               | **8.066%**                               | **~14.9x increase** |
| Total Prediction MSE (u+v)| 1.370478e-01                             | 1.377232e-01                             | Worse |
| Total Execution Time      | ~4.66 minutes                            | ~9.81 minutes                            | ~2.1x slower |

This result further reinforces that simply increasing the number of Adam epochs beyond a certain point (2,000 seems to be a sweet spot for this configuration) is not beneficial for improving `nu` precision and can even be detrimental. It leads to longer training times without accuracy gains, and potentially pushes the optimization into a less favorable basin for `nu` discovery.

## 6. Next Steps

Given the negative impact on precision, this approach of simply increasing Adam epochs beyond 2,000 will not be pursued further. The focus should revert to the best performing configuration (DG-PINN with 2,000 Adam epochs) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting `adam_epochs`:** Set `adam_epochs` back to 2,000 in `src/main_precision.py`.
-   **Investigate Neural Network Depth:** As previously discussed, explore increasing the number of hidden layers in the neural network. This is a relatively simple and quick change to implement and was identified as a promising direction.
-   **Update `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 00:30:36</sub>
