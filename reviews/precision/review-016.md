# Review 011: Data-Guided PINN (DG-PINN) with Increased Adam Epochs (3000)

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the number of Adam epochs from 2,000 to 3,000 on the precision of kinematic viscosity (`nu`) discovery, as per the user's suggestion.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   `adam_epochs` was increased from 2,000 to 3,000.

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs and fixed learning rate).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 40,000
-   **Adam Epochs (Phase 2):** 3,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_adam_3000_epochs.txt`.

-   **Final Discovered `nu`:** 0.049137
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.049137 - 0.05| = 0.000863`
-   **Relative Error:** `(0.000863 / 0.05) * 100% = 1.726%`

### MSE Values

-   Prediction MSE (u): 6.874647e-02
-   Prediction MSE (v): 6.874058e-02
-   Total Prediction MSE (u+v): 1.374871e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.58 seconds
-   Model Initialization Duration: 3.61 seconds
-   Data-Only Pre-training Duration: 20.36 seconds
-   Adam Training Duration: 317.99 seconds (~5.30 minutes)
-   L-BFGS-B Training Duration: 23.88 seconds
-   **Total Execution Duration:** 370.47 seconds (~6.17 minutes)

## 5. Discussion and Comparison

Increasing the Adam epochs from 2,000 to 3,000 **did not improve** the precision of `nu` discovery. Instead, the relative error *increased* from 0.542% to 1.726%. The MSE values for u and v predictions also slightly increased.

| Metric                    | DG-PINN with 2k Adam Epochs (Review 011) | DG-PINN with 3k Adam Epochs (Review 016) | Change |
| :------------------------ | :--------------------------------------- | :--------------------------------------- | :----- |
| Discovered `nu`           | 0.050271                                 | 0.049137                                 | Worse |
| Relative Error            | **0.542%**                               | **1.726%**                               | **~3.18x increase** |
| Total Prediction MSE (u+v)| 1.370478e-01                             | 1.374871e-01                             | Worse |
| Total Execution Time      | ~4.66 minutes                            | ~6.17 minutes                            | ~1.32x slower |

This result further reinforces that increasing the number of Adam epochs beyond 2,000 (in our case, 2,000 seems to be the sweet spot) is not beneficial for improving `nu` precision and can even be detrimental. It leads to longer training times without accuracy gains, and potentially pushes the optimization into a less favorable basin for `nu` discovery.

## 6. Next Steps

Given the negative impact on precision, this approach of simply increasing Adam epochs beyond 2,000 will not be pursued further. The focus should revert to the best performing configuration (DG-PINN with 2,000 Adam epochs) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting `adam_epochs`:** Set `adam_epochs` back to 2,000 in `src/main_precision.py`.
-   **Investigate `num_pde_points`:** Explore a small increase in the number of PDE collocation points (e.g., to 50,000), as this was identified as a promising parameter for fine-tuning precision.
-   **Explore other promising techniques from reviewed papers:** Revisit techniques like Residual-Based Adaptive Refinement (RAR) or more sophisticated adaptive weighting schemes, acknowledging their higher implementation complexity.
-   **Update `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 01:52:01</sub>
