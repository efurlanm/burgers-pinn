# Review 013: Data-Guided PINN (DG-PINN) with 5 Hidden Layers

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the neural network depth from 4 to 5 hidden layers on the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   The `layers` configuration was changed from `[3, 60, 60, 60, 60, 2]` (4 hidden layers) to `[3, 60, 60, 60, 60, 60, 2]` (5 hidden layers).

All other parameters remained at the best configuration established in Review 011 (DG-PINN with 2,000 Adam epochs).

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 60, 2]` (5 hidden layers)
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 40,000
-   **Adam Epochs (Phase 2):** 2,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_5_hidden_layers.txt`.

-   **Final Discovered `nu`:** 0.055569
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.055569 - 0.05| = 0.005569`
-   **Relative Error:** `(0.005569 / 0.05) * 100% = 11.138%`

### MSE Values

-   Prediction MSE (u): 6.843318e-02
-   Prediction MSE (v): 6.854357e-02
-   Total Prediction MSE (u+v): 1.369767e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.56 seconds
-   Model Initialization Duration: 3.94 seconds
-   Data-Only Pre-training Duration: 23.36 seconds
-   Adam Training Duration: 267.35 seconds (~4.46 minutes)
-   L-BFGS-B Training Duration: 13.92 seconds
-   **Total Execution Duration:** 313.14 seconds (~5.22 minutes)

## 5. Discussion and Comparison

Increasing the neural network depth from 4 to 5 hidden layers **did not improve** the precision of `nu` discovery. Instead, the relative error significantly *increased* from 0.542% to 11.138%. The MSE values for u and v predictions remained similar.

| Metric                    | DG-PINN with 4 Hidden Layers (Review 011) | DG-PINN with 5 Hidden Layers (Review 013) | Change |
| :------------------------ | :---------------------------------------- | :---------------------------------------- | :----- |
| Discovered `nu`           | 0.050271                                  | 0.055569                                  | Worse |
| Relative Error            | **0.542%**                                | **11.138%**                               | **~20.5x increase** |
| Total Prediction MSE (u+v)| 1.370478e-01                              | 1.369767e-01                              | Similar |
| Total Execution Time      | ~4.66 minutes                             | ~5.22 minutes                             | Slightly slower |

This result suggests that simply increasing the depth of the neural network beyond 4 hidden layers, with the current configuration, is not beneficial for improving `nu` precision and can even be detrimental. It might be that the network becomes harder to train, or it overfits to the data/PDE constraints in a way that doesn't generalize well to `nu` discovery.

## 6. Next Steps

Given the negative impact on precision, this approach of simply increasing neural network depth beyond 4 hidden layers will not be pursued further. The focus should revert to the best performing configuration (DG-PINN with 4 hidden layers) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting `layers` configuration:** Set `layers` back to `[3, 60, 60, 60, 60, 2]` in `src/main_precision.py`.
-   **Investigate Neural Network Width:** Explore increasing the number of neurons per hidden layer while keeping the depth at 4. This is another common way to increase network capacity.
-   **Update `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 00:39:40</sub>
