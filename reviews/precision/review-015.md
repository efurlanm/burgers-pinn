# Review 015: Data-Guided PINN (DG-PINN) with Exponential Decay Learning Rate Schedule

## 1. Experiment Objective

The objective of this experiment was to implement an exponential decay learning rate schedule for the Adam optimizer and determine if it could further improve the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was modified to use `tf.compat.v1.train.exponential_decay` for the Adam optimizer's learning rate. The `global_step` was incremented in each Adam epoch.

-   `initial_learning_rate`: 0.001
-   `decay_steps`: 1000
-   `decay_rate`: 0.9
-   `staircase`: True

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs and fixed learning rate).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 40,000
-   **Adam Epochs (Phase 2):** 2,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_lr_schedule.txt`.

-   **Final Discovered `nu`:** 0.050536
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.050536 - 0.05| = 0.000536`
-   **Relative Error:** `(0.000536 / 0.05) * 100% = 1.072%`

### MSE Values

-   Prediction MSE (u): 6.857076e-02
-   Prediction MSE (v): 6.849571e-02
-   Total Prediction MSE (u+v): 1.370665e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.50 seconds
-   Model Initialization Duration: 3.62 seconds
-   Data-Only Pre-training Duration: 23.45 seconds
-   Adam Training Duration: 1169.59 seconds (~19.5 minutes)
-   L-BFGS-B Training Duration: 21.95 seconds
-   **Total Execution Duration:** 1223.57 seconds (~20.4 minutes)

## 5. Discussion and Comparison

Implementing the exponential decay learning rate schedule **did not improve** the precision of `nu` discovery. Instead, the relative error *increased* from 0.542% to 1.072%. The MSE values for u and v predictions remained similar.

| Metric                    | DG-PINN with Fixed LR (Review 011) | DG-PINN with LR Schedule (Review 015) | Change |
| :------------------------ | :--------------------------------- | :------------------------------------ | :----- |
| Discovered `nu`           | 0.050271                           | 0.050536                              | Worse |
| Relative Error            | **0.542%**                         | **1.072%**                            | **~1.98x increase** |
| Total Prediction MSE (u+v)| 1.370478e-01                       | 1.370665e-01                          | Similar |
| Total Execution Time      | ~4.66 minutes                      | ~20.4 minutes                         | ~4.38x slower |

The total execution time increased significantly, primarily due to the much longer Adam training duration. While the L-BFGS-B iterations decreased (from 27 to 11), this did not lead to a better `nu` discovery.

This result suggests that, for our current configuration and problem, a fixed learning rate of 0.001 for Adam (as used in Review 011) is more effective than the exponential decay schedule implemented. It's possible that the decay rate or decay steps were not optimal, or that the fixed learning rate already allows the model to converge well enough before L-BFGS-B takes over.

## 6. Next Steps

Given the negative impact on precision and the increased training time, this specific learning rate schedule will not be pursued further. The focus should revert to the best performing configuration (DG-PINN with 2,000 Adam epochs and fixed learning rate) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting Learning Rate Schedule:** Revert the changes made to implement the learning rate schedule in `src/main_precision.py`.
-   **Explore other promising techniques from reviewed papers:** Revisit techniques like Residual-Based Adaptive Refinement (RAR) or more sophisticated adaptive weighting schemes, acknowledging their higher implementation complexity.
-   **Update `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 01:39:45</sub>
