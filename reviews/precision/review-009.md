# Review 009: Data-Guided PINN (DG-PINN) with Simple Adaptive Weighting

## 1. Experiment Objective

The objective of this experiment was to investigate if a simple adaptive weighting scheme for the loss functions during the Adam optimization phase (Phase 2) could further improve the precision of kinematic viscosity (`nu`) discovery. This was based on the general concept of adaptive weighting discussed in `zhou2024.txt`.

## 2. Methodology

The `src/main_precision.py` script was modified to implement a simple adaptive weighting strategy:
-   **Loss Function Return:** The `compute_loss` method was updated to return individual `loss_data` and `loss_pde` components, in addition to the total weighted loss.
-   **Lambda Variables:** `self.lambda_data` and `self.lambda_pde` were converted to `tf.Variable`s to allow dynamic updates during training.
-   **Adaptive Weighting Logic:** Inside the Adam training loop (Phase 2), every 100 epochs, the `lambda_data` and `lambda_pde` weights were updated. The new weights were calculated as inversely proportional to their respective loss magnitudes (with a small epsilon for stability) and then normalized to sum to 1.0.
    ```python
    new_lambda_data = 1.0 / (current_loss_data + epsilon)
    new_lambda_pde = 1.0 / (current_loss_pde + epsilon)
    total_new_lambda = new_lambda_data + new_lambda_pde
    new_lambda_data = new_lambda_data / total_new_lambda
    new_lambda_pde = new_lambda_pde / total_new_lambda
    ```

## 3. Experiment Configuration

-   **Neural Network Architecture (`layers`):** `[3, 60, 60, 60, 60, 2]`
-   **Activation Function:** `tf.tanh`
-   **Number of PDE Points (`num_pde_points`):** 40,000
-   **Adam Epochs (Phase 2):** 1,000
-   **Data-Only Pre-training Epochs (Phase 1):** 10,000
-   **Initial Loss Function Weighting:** `lambda_data_weight = 1.0`, `lambda_pde_weight = 1.0`
-   **Adaptive Weights Interval:** 100 epochs
-   **Random Seed:** 1
-   **Ground Truth `nu`:** 0.05

## 4. Results

The experiment was executed, and the output was logged to `logs/precision_run_dg_pinn_adaptive_weights.txt`.

-   **Final Discovered `nu`:** 0.043442
-   **Ground Truth `nu`:** 0.05

### Precision Analysis

-   **Absolute Error:** `|0.043442 - 0.05| = 0.006558`
-   **Relative Error:** `(0.006558 / 0.05) * 100% = 13.116%`

### HPC Performance Metrics

-   Data Preparation Duration: 3.60 seconds
-   Model Initialization Duration: 3.65 seconds
-   Data-Only Pre-training Duration: 18.73 seconds
-   Adam Training Duration: 113.82 seconds
-   L-BFGS-B Training Duration: 13.57 seconds
-   **Total Execution Duration:** 154.33 seconds (~2.57 minutes)

### Adaptive Weights Behavior

The log shows that the `New Lambda Data` consistently became very high (close to 1.0) while `New Lambda PDE` became very low (close to 0.0). This indicates that the data loss, being significantly smaller after the pre-training phase, dominated the weighting, effectively reducing the influence of the PDE loss.

## 5. Discussion and Comparison

The simple adaptive weighting scheme implemented in this experiment **did not improve** the precision of `nu` discovery. Instead, the relative error increased significantly from 1.342% (achieved with DG-PINN without adaptive weights) to 13.116%.

| Metric                    | DG-PINN (Review 008) | DG-PINN with Adaptive Weights (Review 009) | Change |
| :------------------------ | :------------------- | :----------------------------------------- | :----- |
| Discovered `nu`           | 0.050671             | 0.043442                                   | Worse |
| Relative Error            | **1.342%**           | **13.116%**                                | **~9.8x increase** |
| Total Execution Time      | ~5.5 minutes         | ~2.57 minutes                              | ~2.1x faster |

While the total execution time was further reduced, this came at a substantial cost to accuracy. The aggressive weighting towards the data loss, due to its much smaller magnitude after pre-training, likely suppressed the contribution of the PDE loss. The PDE loss is crucial for guiding the model to discover the unknown parameter `nu`.

This experiment suggests that simple adaptive weighting based solely on loss magnitudes is not suitable for PINNs, especially when there's a large disparity in the magnitudes of different loss components. More sophisticated adaptive weighting strategies, such as those that consider gradient magnitudes or the neural tangent kernel, as hinted at in `zhou2024.txt`, are likely required to effectively balance the loss terms without sacrificing accuracy.

## 6. Next Steps

Given the negative impact on precision, this specific adaptive weighting approach will not be pursued further for `nu` discovery. The focus should revert to the best performing configuration (DG-PINN without adaptive weights) and explore other promising avenues for improving precision.

Potential next steps include:
-   **Reverting to DG-PINN without adaptive weights:** Ensure `src/main_precision.py` is set back to the configuration that yielded 1.342% relative error.
-   **Analyzing MSE:** Implement MSE calculation for the predicted solution against the ground truth to provide another quantitative measure of accuracy, as previously planned.
-   **Exploring other optimization techniques:** Investigate other optimization techniques or hyperparameter tuning strategies that have shown promise in PINN literature for inverse problems.
-   **Updating `project_checkpoint.md`:** Reflect the current best result and the revised next planned experiments.
-   **Updating `manuscript.tex`:** Incorporate these findings into the academic paper, discussing the DG-PINN methodology and the results of the adaptive weighting experiment.

<br><sub>Last edited: 2025-08-09 23:27:53</sub>
