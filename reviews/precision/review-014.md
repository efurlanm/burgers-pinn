# Review 014: Data-Guided PINN (DG-PINN) with 80 Neurons per Layer

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the neural network width from 60 to 80 neurons per hidden layer on the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   The `layers` configuration was changed from `[3, 60, 60, 60, 60, 2]` (4 hidden layers, 60 neurons each) to `[3, 80, 80, 80, 80, 2]` (4 hidden layers, 80 neurons each).

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 80, 80, 80, 80, 2]` (4 hidden layers, 80 neurons each)
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 40,000
-   **Adam Epochs (Phase 2):** 2,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_80_neurons.txt`.

-   **Final Discovered `nu`:** 0.053971
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.053971 - 0.05| = 0.003971`
-   **Relative Error:** `(0.003971 / 0.05) * 100% = 7.942%`

### MSE Values

-   Prediction MSE (u): 6.826865e-02
-   Prediction MSE (v): 6.835903e-02
-   Total Prediction MSE (u+v): 1.366277e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.58 seconds
-   Model Initialization Duration: 3.63 seconds
-   Data-Only Pre-training Duration: 20.83 seconds
-   Adam Training Duration: 286.74 seconds (~4.78 minutes)
-   L-BFGS-B Training Duration: 21.77 seconds
-   **Total Execution Duration:** 337.54 seconds (~5.63 minutes)

## 5. Discussion and Comparison

Increasing the neural network width from 60 to 80 neurons per layer **did not improve** the precision of `nu` discovery. Instead, the relative error significantly *increased* from 0.542% to 7.942%. The MSE values for u and v predictions remained similar.

| Metric                    | DG-PINN with 60 Neurons (Review 011) | DG-PINN with 80 Neurons (Review 014) | Change |
| :------------------------ | :----------------------------------- | :----------------------------------- | :----- |
| Discovered `nu`           | 0.050271                             | 0.053971                             | Worse |
| Relative Error            | **0.542%**                           | **7.942%**                           | **~14.6x increase** |
| Total Prediction MSE (u+v)| 1.370478e-01                         | 1.366277e-01                         | Similar |
| Total Execution Time      | ~4.66 minutes                        | ~5.63 minutes                        | Slightly slower |

This result suggests that simply increasing the width of the neural network beyond 60 neurons per layer, with the current configuration, is not beneficial for improving `nu` precision and can even be detrimental. It might be that the increased capacity leads to overfitting or makes the optimization landscape more complex for `nu` discovery.

## 6. Next Steps

Given the negative impact on precision, this approach of simply increasing neural network width beyond 60 neurons per layer will not be pursued further. The focus should revert to the best performing configuration (DG-PINN with 4 hidden layers and 60 neurons per layer) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting `layers` configuration:** Set `layers` back to `[3, 60, 60, 60, 60, 2]` in `src/main_precision.py`.
-   **Investigate Neural Network Depth and Width Simultaneously:** While individual increases in depth or width were detrimental, it might be beneficial to explore combinations, or a more systematic search for optimal architecture.
-   **Explore other promising techniques from reviewed papers:** Revisit techniques like Residual-Based Adaptive Refinement (RAR) or more sophisticated adaptive weighting schemes, acknowledging their higher implementation complexity.
-   **Update `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 00:48:25</sub>
