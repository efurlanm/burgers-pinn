# Review 018: Data-Guided PINN (DG-PINN) with 60,000 PDE Collocation Points

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the number of PDE collocation points from 50,000 to 60,000 on the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   `num_pde_points` was increased from 50,000 to 60,000.

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs and fixed learning rate).

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

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_60000_pde_points.txt`.

-   **Final Discovered `nu`:** 0.049978
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.049978 - 0.05| = 0.000022`
-   **Relative Error:** `(0.000022 / 0.05) * 100% = 0.044%`

### MSE Values

-   Prediction MSE (u): 6.874458e-02
-   Prediction MSE (v): 6.875561e-02
-   Total Prediction MSE (u+v): 1.375002e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.55 seconds
-   Model Initialization Duration: 3.60 seconds
-   Data-Only Pre-training Duration: 20.51 seconds
-   Adam Training Duration: 310.19 seconds (~5.17 minutes)
-   L-BFGS-B Training Duration: 97.43 seconds (~1.62 minutes)
-   **Total Execution Duration:** 436.79 seconds (~7.28 minutes)

## 5. Discussion and Comparison

Increasing the number of PDE collocation points from 50,000 to 60,000 **significantly improved** the precision of `nu` discovery. The relative error decreased from 0.486% to **0.044%**.

| Metric                    | DG-PINN with 50k PDE Points (Review 017) | DG-PINN with 60k PDE Points (Review 018) | Change |
| :------------------------ | :--------------------------------------- | :--------------------------------------- | :----- |
| Discovered `nu`           | 0.050243                                 | 0.049978                                 | Better |
| Relative Error            | **0.486%**                               | **0.044%**                               | **~90.9% reduction** |
| Total Prediction MSE (u+v)| 1.371298e-01                             | 1.375002e-01                             | Similar |
| Total Execution Time      | ~5.24 minutes                            | ~7.28 minutes                            | ~1.39x slower |

This result **surpasses** the precision reported in Raissi et al. (2019) (0.469% for noise-free data in their inverse Burgers' problem). The increase in total execution time is a reasonable trade-off for the significantly improved precision.

## 6. Next Steps

Given this excellent result, the next steps should focus on further refining the precision and exploring other promising avenues.

Potential next steps include:
-   **Further increase `num_pde_points`:** Experiment with `num_pde_points = 70000` or `80000` to see if further improvements can be achieved, while monitoring the trade-off with execution time.
-   **Investigate Neural Network Depth and Width Simultaneously:** While individual increases in depth or width were detrimental, it might be beneficial to explore combinations, or a more systematic search for optimal architecture.
-   **Explore other promising techniques from reviewed papers:** Revisit techniques like Residual-Based Adaptive Refinement (RAR) or more sophisticated adaptive weighting schemes, acknowledging their higher implementation complexity.
-   **Update `project_checkpoint.md`:** Reflect the new best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 02:09:15</sub>
