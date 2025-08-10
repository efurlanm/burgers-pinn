# Review 006: Investigating the Impact of PDE Collocation Points

## 1. Summary of Previous Findings

Experiments in `review-004.md` and `review-005.md` have shown that neither the loss function weights nor the number of Adam epochs have been sufficient to significantly improve the precision of the discovered `nu` parameter. The error remains consistently high at ~20%.

The number of Adam epochs appears to be a sensitive parameter, with longer training runs not necessarily leading to better results. The best result so far was a ~19% error, achieved with 1000 Adam epochs and equal loss weights (1.0 for data, 1.0 for PDE).

## 2. New Hypothesis and Experimental Plan

**Hypothesis**: The number of collocation points used to enforce the PDE residual (`num_pde_points`) is a critical hyperparameter. The current value of 80,000 may not be optimal, and exploring different values could lead to a better-conditioned optimization problem and thus a more precise `nu`.

**Constraint**: The GPU memory is a limitation. The number of points cannot be increased excessively. The current value of 80,000 is close to the limit.

**Experimental Setup**:
-   **Adam Epochs**: Fixed at 1000.
-   **Loss Weights**: Fixed at `lambda_data = 1.0` and `lambda_pde = 1.0`.

**Experiments**:
-   **Baseline**: `num_pde_points = 80000` (from Run 5.1, error: 19.14%)
-   **Experiment 6.1**: `num_pde_points = 20000`
-   **Experiment 6.2**: `num_pde_points = 40000`
-   **Experiment 6.3**: `num_pde_points = 100000` (modest increase)

This series of experiments will help us understand the sensitivity of the model to the density of physics-based training points.

<br><sub>Last edited: 2025-08-09 15:17:57</sub>
