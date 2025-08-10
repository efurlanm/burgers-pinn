# Review 017: Data-Guided PINN (DG-PINN) with 50,000 PDE Collocation Points

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the number of PDE collocation points from 40,000 to 50,000 on the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   `num_pde_points` was increased from 40,000 to 50,000.

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs and fixed learning rate).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 50,000
-   **Adam Epochs (Phase 2):** 2,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_50000_pde_points.txt`.

-   **Final Discovered `nu`:** 0.050243
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.050243 - 0.05| = 0.000243`
-   **Relative Error:** `(0.000243 / 0.05) * 100% = 0.486%`

### MSE Values

-   Prediction MSE (u): 6.858678e-02
-   Prediction MSE (v): 6.854301e-02
-   Total Prediction MSE (u+v): 1.371298e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.59 seconds
-   Model Initialization Duration: 3.60 seconds
-   Data-Only Pre-training Duration: 20.80 seconds
-   Adam Training Duration: 262.00 seconds (~4.37 minutes)
-   L-BFGS-B Training Duration: 23.18 seconds
-   **Total Execution Duration:** 314.20 seconds (~5.24 minutes)

## 5. Discussion and Comparison

Increasing the number of PDE collocation points from 40,000 to 50,000 **improved** the precision of `nu` discovery. The relative error decreased from 0.542% to **0.486%**.

| Metric                    | DG-PINN with 40k PDE Points (Review 011) | DG-PINN with 50k PDE Points (Review 017) | Change |
| :------------------------ | :--------------------------------------- | :--------------------------------------- | :----- |
| Discovered `nu`           | 0.050271                                 | 0.050243                                 | Better |
| Relative Error            | **0.542%**                               | **0.486%**                               | **~10.3% reduction** |
| Total Prediction MSE (u+v)| 1.370478e-01                             | 1.371298e-01                             | Similar |
| Total Execution Time      | ~4.66 minutes                            | ~5.24 minutes                            | Slightly slower |

This result brings the `nu` discovery precision very close to the values reported in Raissi et al. (2019) (0.469% for noise-free data in their inverse Burgers' problem). The slight increase in total execution time is a reasonable trade-off for the improved precision.

## 6. Next Steps

Given this promising result, the next steps should focus on further refining the precision and exploring other promising avenues.

Potential next steps include:
-   **Further increase `num_pde_points`:** Experiment with `num_pde_points = 60000` or `70000` to see if further improvements can be achieved, while monitoring the trade-off with execution time.
-   **Investigate Neural Network Depth and Width Simultaneously:** While individual increases in depth or width were detrimental, it might be beneficial to explore combinations, or a more systematic search for optimal architecture.
-   **Explore other promising techniques from reviewed papers:** Revisit techniques like Residual-Based Adaptive Refinement (RAR) or more sophisticated adaptive weighting schemes, acknowledging their higher implementation complexity.
-   **Update `project_checkpoint.md`:** Reflect the new best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 01:59:39</sub>
