# Review 011: Data-Guided PINN (DG-PINN) with Increased Adam Epochs (2000)

## 1. Experiment Objective

The objective of this experiment was to investigate the impact of increasing the number of Adam epochs from 1,000 to 2,000 on the precision of kinematic viscosity (`nu`) discovery.

## 2. Methodology

The `src/main_precision.py` script was configured with the following change:
-   `adam_epochs` was increased from 1,000 to 2,000.

All other parameters remained at the best configuration established in Review 008 (DG-PINN without adaptive weights).

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

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_adam_2000_epochs.txt`.

-   **Final Discovered `nu`:** 0.050271
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.050271 - 0.05| = 0.000271`
-   **Relative Error:** `(0.000271 / 0.05) * 100% = 0.542%`

### MSE Values

-   Prediction MSE (u): 6.854109e-02
-   Prediction MSE (v): 6.850673e-02
-   Total Prediction MSE (u+v): 1.370478e-01

### HPC Performance Metrics

-   Data Preparation Duration: 3.60 seconds
-   Model Initialization Duration: 3.64 seconds
-   Data-Only Pre-training Duration: 19.75 seconds
-   Adam Training Duration: 213.67 seconds (~3.56 minutes)
-   L-BFGS-B Training Duration: 37.72 seconds
-   **Total Execution Duration:** 279.54 seconds (~4.66 minutes)

## 5. Discussion and Comparison

Increasing the Adam epochs from 1,000 to 2,000 **significantly improved** the precision of `nu` discovery. The relative error decreased from 1.342% to **0.542%**.

| Metric                    | DG-PINN (Review 008) | DG-PINN with 2k Adam Epochs (Review 011) | Change |
| :------------------------ | :------------------- | :--------------------------------------- | :----- |
| Discovered `nu`           | 0.050671             | 0.050271                                 | Better |
| Relative Error            | **1.342%**           | **0.542%**                               | **~2.5x reduction** |
| Total Prediction MSE (u+v)| 1.369942e-01         | 1.370478e-01                             | Similar |
| Total Execution Time      | ~5.5 minutes         | ~4.66 minutes                            | ~1.2x faster |
| L-BFGS-B Iterations       | 188                  | 27                                       | Significant reduction |

The total execution time slightly decreased, which is a positive side effect. The substantial reduction in L-BFGS-B iterations (from 188 to 27) indicates that the longer Adam phase provided a much better starting point for the L-BFGS-B optimizer, leading to faster and more accurate convergence.

This result brings the `nu` discovery precision very close to the values reported in Raissi et al. (2019) (0.469% for noise-free data in their inverse Burgers' problem).

## 6. Next Steps

Given this promising result, the next steps should focus on further refining the precision and exploring other promising avenues.

Potential next steps include:
-   **Further increase Adam epochs:** Experiment with `adam_epochs = 5000` to see if further improvements can be achieved, while monitoring the trade-off with execution time.
-   **Investigate Neural Network Depth:** As previously discussed, explore increasing the number of hidden layers in the neural network. This is a relatively simple and quick change to implement and was identified as a promising direction.
-   **Update `project_checkpoint.md`:** Reflect the new best result and the revised next planned experiments.
-   **Update `manuscript.tex`:** Incorporate these findings into the academic paper.

<br><sub>Last edited: 2025-08-10 00:19:03</sub>
