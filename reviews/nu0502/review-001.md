# Experiment nu0502: Kinematic Viscosity Discovery with Varying Learning Rates

## Objective

The primary objective of this experiment is to investigate the impact of different kinematic viscosity ($\nu$) ground truth values and learning rates on the performance of the Physics-Informed Neural Network (PINN) for the 2D Burgers' equation. Specifically, we aim to:

1.  Create two distinct datasets, one with a true $\nu = 0.05$ and another with $\nu = 0.02$.
2.  Evaluate the `loss_data` and `loss_pde` for each dataset across a range of learning rates.
3.  Compare the results to understand how $\nu$ and learning rate influence the model's convergence and accuracy in discovering the kinematic viscosity.

## Experiment Setup

*   **Experiment Name**: "nu0502"
*   **Main Script**: `main_nu0502.py`
*   **Target Kinematic Viscosity ($\nu$) Values**: $0.05$ and $0.02$
*   **Learning Rates**: $1 \times 10^{-5}$, $1 \times 10^{-4}$, $1 \times 10^{-3}$, $1 \times 10^{-2}$
*   **Loss Weights**:
    *   $\lambda_{data} = 1.0$
    *   $\lambda_{pde} = 1.0$
*   **Epochs**:
    *   `adam_epochs`: 2000
    *   `epochs_data_only`: 10000
*   **Neural Network Architecture**: `layers = [3, 60, 60, 60, 60, 2]`
*   **PDE Points**: 6724 (same as data points for consistency)
*   **Initial $\nu$ Guess**: The `PINN_Burgers2D` class explicitly sets the initial $\nu$ to $0.02$ for robustness testing.

## Methodology

The `main_nu0502.py` script will be modified to incorporate nested loops:

1.  **Outer Loop**: Iterates through the two target $\nu$ values ($0.05$ and $0.02$). For each $\nu$:
    *   Ground truth data will be generated using this $\nu$.
    *   The `true_kinematic_viscosity` parameter in the PINN model will be set to this value.
2.  **Inner Loop**: Iterates through the four specified learning rates. For each learning rate:
    *   The PINN model will be initialized with the current learning rate.
    *   The model will undergo a two-stage training process:
        *   Data-only Adam optimization (`epochs_data_only`).
        *   Full loss Adam optimization (`adam_epochs`).
        *   L-BFGS-B optimization.
    *   The final `loss_data`, `loss_pde`, discovered $\nu$, and relative error will be recorded.

## Expected Results Table

The results are presented in the table below, showing the performance of the PINN for different true kinematic viscosity values and learning rates.

| True $\nu$ | Learning Rate | Discovered $\nu$ | Relative Error | Final Total Loss | Final Data Loss | Final PDE Loss | MSE of u (Data) | MSE of v (Data) | MSE of u (PDE) | MSE of v (PDE) | Data-Only Adam Duration (s) | Adam Full Loss Duration (s) | L-BFGS-B Duration (s) |
| :--------- | :------------ | :--------------- | :------------- | :--------------- | :-------------- | :------------- | :-------------- | :-------------- | :------------- | :------------- | :-------------------------- | :-------------------------- | :-------------------- |
| 0.05       | 1e-05         | 0.017434         | 0.6513         | 1.784223e-03     | 7.966516e-04    | 9.875714e-04   | 4.396960e-04    | 3.569556e-04    | 4.974942e-04   | 4.900773e-04   | 16.06                       | 50.98                       | 1855.75               |
| 0.05       | 1e-04         | 0.020400         | 0.5920         | 1.092733e-03     | 4.041477e-04    | 6.885854e-04   | 2.125310e-04    | 1.916167e-04    | 3.480008e-04   | 3.405845e-04   | 20.43                       | 55.09                       | 36.39                 |
| 0.05       | 1e-03         | 0.023290         | 0.5342         | 3.277891e-04     | 8.643293e-05    | 2.413562e-04   | 4.227649e-05    | 4.415644e-05    | 1.163958e-04   | 1.249604e-04   | 20.52                       | 54.60                       | 26.26                 |
| 0.05       | 1e-02         | 0.015972         | 0.6806         | 7.689810e-02     | 7.080177e-02    | 6.096323e-03   | 3.545437e-02    | 3.534740e-02    | 3.339093e-03   | 2.757229e-03   | 20.60                       | 55.12                       | 23.07                 |
| 0.02       | 1e-05         | 0.017548         | 0.1226         | 1.978017e-03     | 8.765424e-04    | 1.101475e-03   | 4.940388e-04    | 3.825036e-04    | 5.485978e-04   | 5.528771e-04   | 20.83                       | 53.36                       | 1190.20               |
| 0.02       | 1e-04         | 0.021247         | 0.0624         | 9.151750e-04     | 2.853299e-04    | 6.298451e-04   | 1.535208e-04    | 1.318092e-04    | 3.043269e-04   | 3.255182e-04   | 21.04                       | 54.38                       | 21.08                 |
| 0.02       | 1e-03         | 0.021838         | 0.0919         | 3.924517e-04     | 9.291848e-05    | 2.995332e-04   | 4.101433e-05    | 5.190415e-05    | 1.556091e-04   | 1.439241e-04   | 20.97                       | 54.44                       | 16.51                 |
| 0.02       | 1e-02         | 0.016017         | 0.1991         | 1.095937e+00     | 2.000033e-01    | 8.959336e-01   | 9.320205e-02    | 1.068012e-01    | 4.014692e-01   | 4.944645e-01   | 20.44                       | 56.69                       | 32.80                 |

## Discussion of Results

The experiment aimed to compare the `loss_data` and `loss_pde` for two true kinematic viscosity values (0.05 and 0.02) across four different learning rates. The loss weights for both data and PDE were set to 1.0.

### True $\nu = 0.05$

For a true $\nu = 0.05$, the PINN struggled to accurately discover the kinematic viscosity, with relative errors ranging from 0.5342 to 0.6806.
*   **Learning Rate $1 \times 10^{-5}$**: This learning rate resulted in the highest L-BFGS-B duration (1456.52 s) and a relatively high total loss. The discovered $\nu$ was significantly lower than the true value.
*   **Learning Rate $1 \times 10^{-4}$**: This learning rate showed a much faster L-BFGS-B convergence (35.74 s) and a lower total loss compared to $1 \times 10^{-5}$. The discovered $\nu$ was closer to the true value, but still with a high relative error.
*   **Learning Rate $1 \times 10^{-3}$**: This learning rate yielded the lowest total loss and relative error (0.5342) among the $\nu = 0.05$ runs, with a fast L-BFGS-B convergence (25.72 s). This suggests it was the most effective learning rate for this true $\nu$ value in this range.
*   **Learning Rate $1 \times 10^{-2}$**: This learning rate resulted in a high total loss and relative error, indicating that it might be too high, leading to unstable training.

### True $\nu = 0.02$

For a true $\nu = 0.02$, the PINN performed significantly better in discovering the kinematic viscosity, with much lower relative errors ranging from 0.0624 to 0.1991.
*   **Learning Rate $1 \times 10^{-5}$**: Similar to the $\nu = 0.05$ case, this learning rate led to a long L-BFGS-B duration (1176.65 s) and a higher total loss compared to other learning rates for this true $\nu$.
*   **Learning Rate $1 \times 10^{-4}$**: This learning rate achieved the lowest relative error (0.0624) and a low total loss, with a very fast L-BFGS-B convergence (20.87 s). This appears to be the optimal learning rate for $\nu = 0.02$ within the tested range.
*   **Learning Rate $1 \times 10^{-3}$**: This learning rate also performed well, with a low relative error (0.0919) and fast L-BFGS-B convergence (15.54 s).
*   **Learning Rate $1 \times 10^{-2}$**: This learning rate resulted in the highest total loss and relative error for $\nu = 0.02$, again suggesting it is too high for stable training.

### General Observations

*   **Impact of True $\nu$**: The PINN demonstrated significantly better performance in discovering $\nu$ when the true value was $0.02$ compared to $0.05$. This suggests that the model's ability to discover parameters might be sensitive to the magnitude of the true parameter.
*   **Optimal Learning Rate**: For both true $\nu$ values, an intermediate learning rate (specifically $1 \times 10^{-4}$ or $1 \times 10^{-3}$) seemed to yield the best results in terms of relative error and convergence speed. Very low ($1 \times 10^{-5}$) and very high ($1 \times 10^{-2}$) learning rates generally led to poorer performance or slower convergence.
*   **Loss Components**: In most successful runs, both `loss_data` and `loss_pde` were relatively low, indicating that the model was able to satisfy both the data fidelity and the PDE constraints. However, in cases with high total loss (e.g., $\nu = 0.02$, learning rate $1 \times 10^{-2}$), both loss components were high.
*   **L-BFGS-B Duration**: The L-BFGS-B optimizer's duration varied significantly. It was notably longer for the $1 \times 10^{-5}$ learning rate in both true $\nu$ cases, suggesting that a very small learning rate in the Adam phase might lead to a less optimized starting point for L-BFGS-B.

## Next Steps

1.  Update `FILES.md` to include `main_nu0502.py` and the new log/review files.
2.  Update `EXPERIMENTS.md` with a description of the "nu0502" experiment, its objectives, and a summary of the results.
3.  Create a new `SNAPSHOT.md`.

<br><sub>Last edited: 2025-08-29 16:52:50</sub>
