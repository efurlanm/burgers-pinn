# Review 007: Testing Swish Activation Function

## 1. Summary of Previous Findings

The investigation into the number of PDE collocation points revealed a clear optimal value. The best precision was achieved with 40,000 points, resulting in a relative error of 6.46% for `nu`.

**Best Configuration So Far**:
-   **Adam Epochs**: 1000
-   **Loss Weights**: `lambda_data = 1.0`, `lambda_pde = 1.0`
-   **PDE Points**: 40,000
-   **Activation Function**: `tanh`

## 2. New Hypothesis: Swish Activation

Based on the user's suggestion from the `cuomo2022` paper, the next hypothesis is that changing the activation function from `tanh` to `Swish` (`tf.nn.swish`) may improve gradient flow and lead to a more accurate result.

## 3. Experimental Plan

I will conduct a single, direct comparison using the best configuration identified above.

**Experiment 7.1**: 
-   **Objective**: Compare the performance and accuracy of the `Swish` activation function against `tanh`.
-   **Configuration**:
    -   **Adam Epochs**: 1000
    -   **Loss Weights**: `lambda_data = 1.0`, `lambda_pde = 1.0`
    -   **PDE Points**: 40,000
    -   **Activation Function**: `Swish`

This experiment will provide a clear indication of the impact of the activation function on the final precision.

## 4. Results of Swish Activation Experiment (Run 7.2)

-   **Configuration**: Adam Epochs = 1000, Lambda Data = 1.0, Lambda PDE = 1.0, PDE Points = 20000, Activation Function = Swish
-   **Accuracy**: Discovered `nu` = 0.059918 (19.84% error)
-   **Performance**: Total time = 234.15 s (~3.9 minutes)

**Conclusion**: While the Swish activation function led to a dramatic speedup in execution time (~17x faster than tanh), it resulted in a significant decrease in precision (19.84% error vs 8.05% for tanh). This suggests a trade-off where the faster convergence of Swish leads to a suboptimal solution for `nu`.

## 5. Summary of PDE Collocation Points Experiments

Our investigation into the number of PDE collocation points has yielded valuable insights:

| PDE Points | Discovered `nu` | Relative Error | Total Time (s) |
|:---:|:---:|:---:|:---:|
| 20,000 | 0.045974 | 8.05% | 4098 |
| **40,000** | **0.046770** | **6.46%** | **4108** |
| 80,000 | 0.059571 | 19.14% | (not timed with HPC) |
| 100,000 | 0.040729 | 18.54% | 2842 |

**Conclusion**: The optimal number of PDE collocation points for `nu` discovery is **40,000**, achieving a relative error of **6.46%**. This is a significant improvement over our initial baseline. Increasing the number of points beyond this (80,000 and 100,000) led to a decrease in precision, with the error returning to the ~20% range. The total execution time is largely dominated by the L-BFGS-B optimization phase, and while 100,000 points showed a faster total time, it came at the cost of accuracy.

## 6. Next Steps: Implementing Data-Guided PINN (DG-PINN)

Our current best configuration is:
-   Adam Epochs: 1000
-   Loss Weights: `lambda_data=1.0`, `lambda_pde=1.0`
-   PDE Points: 40,000
-   Activation Function: `tanh`

As discussed, the **Data-Guided PINN (DG-PINN)** approach from `zhou2024.pdf` is a promising strategy to further improve precision by addressing the loss imbalance problem through a two-phase training process.

**Experiment 8.1 (DG-PINN):**
-   **Objective**: Test if the DG-PINN two-phase training approach improves precision compared to our current best result.
-   **Methodology**: 
    -   I will use our best configuration as the base: 1000 Adam epochs (for the fine-tuning phase), 1/1 weights, 40k PDE points, `tanh` activation.
    -   I will modify the `main_precision.py` script to implement the two-phase training:
        -   **Phase 1 (Pre-training):** Train the network for a certain number of epochs (e.g., 10,000) using *only* the data loss (`loss_data`).
        -   **Phase 2 (Fine-tuning):** Load the weights from Phase 1 and then run the standard training process (Adam + L-BFGS) on the full composite loss.
-   **Evaluation**: Compare the final `nu` error, MSE, and total execution time against our best result (6.46% error).
<br><sub>Last edited: 2025-08-09 21:51:32</sub>
