# Review 019: Data-Guided PINN (DG-PINN) with 70,000 PDE Collocation Points

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the number of PDE collocation points from 60,000 to 70,000 on the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   `num_pde_points` was increased from 60,000 to 70,000.

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs and fixed learning rate).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 70,000
-   **Adam Epochs (Phase 2):** 2,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_70000_pde_points.txt`.

-   **Final Discovered `nu`:** 0.050046
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.050046 - 0.05| = 0.000046`
-   **Relative Error:** `(0.000046 / 0.05) * 100% = 0.092%`

### MSE Values

-   Prediction MSE (u): 6.858969e-02
-   Prediction MSE (v): 6.851117e-02
-   Total Prediction MSE (u+v): 1.371008e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.56 seconds
-   Model Initialization Duration: 3.59 seconds
-   Data-Only Pre-training Duration: 17.12 seconds
-   Adam Training Duration: 358.69 seconds (~5.98 minutes)
-   L-BFGS-B Training Duration: 12.24 seconds (~0.20 minutes)
-   **Total Execution Duration:** 396.11 seconds (~6.60 minutes)

## 5. Discussion and Comparison

Increasing the number of PDE collocation points from 60,000 to 70,000 **did not improve** the precision of `nu` discovery. Instead, the relative error *increased* from 0.044% to 0.092%. The MSE values for u and v predictions remained similar.

| Metric                    | DG-PINN with 60k PDE Points (Review 018) | DG-PINN with 70k PDE Points (Review 019) | Change |
| :------------------------ | :--------------------------------------- | :--------------------------------------- | :----- |
| Discovered `nu`           | 0.049978                                 | 0.050046                                 | Worse |
| Relative Error            | **0.044%**                               | **0.092%**                               | **~109% increase** |
| Total Prediction MSE (u+v)| 1.375002e-01                             | 1.371008e-01                             | Similar |
| Total Execution Duration  | ~7.28 minutes                            | ~6.60 minutes                            | Slightly faster |

This result suggests that 60,000 PDE points might be the optimal number for our current configuration. While the total execution time slightly decreased, this came at the cost of precision.

## 6. Next Steps

Given the negative impact on precision, this approach of simply increasing PDE collocation points beyond 60,000 will not be pursued further. The focus should revert to the best performing configuration (DG-PINN with 60,000 PDE points) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting `num_pde_points`:** Set `num_pde_points` back to 60,000 in `src/main_precision.py`.
-   **Robustness Check (Varying Random Seed):** Run the best configuration (DG-PINN with 60,000 PDE points, 2,000 Adam epochs, 4 hidden layers, 60 neurons) multiple times with different random seeds to assess the robustness and reliability of the achieved precision. This is crucial for the academic paper.
-   **Generalizability Check (Varying `nu` parameter):** After the robustness check, demonstrate the method's ability to discover other `nu` values by generating ground truth data with different `nu` values and running the PINN.
-   **Explore other promising techniques from reviewed papers:** Revisit techniques like Residual-Based Adaptive Refinement (RAR) or more sophisticated adaptive weighting schemes, acknowledging their higher implementation complexity.
-   **Update `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.
<br><sub>Last edited: 2025-08-10 02:33:54</sub>
