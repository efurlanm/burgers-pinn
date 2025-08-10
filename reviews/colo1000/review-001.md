# Review 001: Colo1000 Experiment - High PDE Loss Weight and Ground Truth Collocation Points

## Objective

The primary objective of this experiment, named "colo1000", is to investigate the impact of a significantly increased PDE loss weight ($\lambda_{pde} = 1000.0$) and varying learning rates on the discovery of kinematic viscosity ($\nu$) in the 2D Burgers' equation using a Physics-Informed Neural Network (PINN). This experiment also utilizes the "Ground Truth 41x41 points" (totaling 6724 points) for both the data fidelity loss and the PDE residual loss, similar to the "coloc" experiment.

## Experiment Parameters

* **Learning Rates:** 10 logarithmically spaced values from $10^{-5}$ to $10^{-2}$ (obtained using `np.logspace(np.log10(1e-5), np.log10(1e-2), 10)`)
* **Grid Points (x, y):** 41x41
* **Time Steps:** 50
* **True Kinematic Viscosity $\nu$:** 0.05
* **Spatial Domain (x, y):** [0.0, 2.0] x [0.0, 2.0]
* **Time Domain (t):** [0.0, 0.05]
* **Neural Network Layers:** [3, 60, 60, 60, 60, 2] (3 input, 4 hidden layers with 60 neurons each, 2 output)
* **Adam Epochs (Full Loss):** 2000
* **Data-Only Pre-training Epochs:** 10000
* **Data Loss Weight ($\\lambda_{data}$):** 1.0
* **PDE Loss Weight ($\\lambda_{pde}$):** 1000.0
* **Number of PDE Collocation Points:** 6724 (same as data points)
* **PDE Collocation Points Source:** The `x_pde`, `y_pde`, `t_pde` tensors are directly assigned from the `x_data_tf`, `y_data_tf`, `t_data_tf` tensors, ensuring that the same 41x41 ground truth points (6724 points) are used for both data and PDE loss terms.
* **L-BFGS-B Options:** `maxiter`: 100000, `maxfun`: 100000, `maxcor`: 100, `maxls`: 50, `ftol`: 1e-20
* **Initial Kinematic Viscosity ($\\nu$) Value:** 0.02 (explicitly set for robustness test)

## Results

| Run | Learning Rate | Discovered $\nu$ | Relative Error | Loss Data | Loss PDE | Loss Total |
|:--- |:------------- |:---------------- |:-------------- |:--------- |:-------- |:---------- |
| 1   | 1.000000e-05  | 0.019213         | 0.6157         | 0.092308  | 0.000088 | 0.180680   |
| 2   | 2.154435e-05  | 0.019776         | 0.6045         | 0.073224  | 0.000138 | 0.211369   |
| 3   | 4.641589e-05  | 0.019983         | 0.6003         | 0.053810  | 0.000400 | 0.453843   |
| 4   | 1.000000e-04  | 0.019902         | 0.6020         | 0.045764  | 0.000222 | 0.267885   |
| 5   | 2.154435e-04  | 0.019873         | 0.6025         | 0.050273  | 0.000163 | 0.213037   |
| 6   | 4.641589e-04  | 0.020271         | 0.5946         | 0.030723  | 0.000135 | 0.166030   |
| 7   | 1.000000e-03  | 0.019496         | 0.6101         | 0.032528  | 0.000066 | 0.098262   |
| 8   | 2.154435e-03  | 0.018279         | 0.6344         | 0.041285  | 0.000020 | 0.061666   |
| 9   | 4.641589e-03  | 0.015588         | 0.6882         | 0.041767  | 0.000014 | 0.055726   |
| 10  | 1.000000e-02  | 0.016832         | 0.6634         | 0.087462  | 0.002540 | 2.627918   |

## Discussion of Results

This experiment investigated the impact of a high PDE loss weight ($\lambda_{pde} = 1000.0$) across a range of learning rates on the discovery of kinematic viscosity ($\nu$) in the 2D Burgers' equation using a Physics-Informed Neural Network (PINN). The results show that the discovered $\nu$ values consistently converge to a range between approximately 0.0156 and 0.0203, which is significantly lower than the true value of 0.05. This leads to relative errors ranging from 0.5946 to 0.6882.

Comparing these results to the "coloc" experiment (Review 007), where $\lambda_{pde} = 100.0$, the discovered $\nu$ values in "colo1000" are generally similar, and in some cases, the relative error is slightly higher. This suggests that increasing $\lambda_{pde}$ from 100 to 1000, while keeping other parameters constant, did not lead to a substantial improvement in the accuracy of $\nu$ discovery. In fact, for higher learning rates (e.g., $1.0 \times 10^{-2}$), the discovered $\nu$ deviates more significantly from the true value, and the total loss increases considerably, indicating potential instability or difficulty in convergence.

The `Loss Data` values generally decrease as the learning rate increases up to a certain point, and then start to increase again for very high learning rates. Conversely, the `Loss PDE` values tend to be very small across most learning rates, indicating that the PDE constraint is being strongly enforced, but this strong enforcement does not translate to accurate $\nu$ discovery. The `Loss Total` shows a similar trend to `Loss Data`, with the lowest total loss observed around a learning rate of $1.0 \times 10^{-3}$.

## Profiling and Analysis

The training process consistently followed the two-stage optimization:

1. **Data-Only Adam Pre-training:** This phase effectively reduced the data loss across all learning rates.
2. **Adam Optimization (Full Loss):** This phase optimized both data and PDE losses. The discovered $\nu$ generally remained close to the initial guess of 0.02 during this phase, with slight variations depending on the learning rate.
3. **L-BFGS-B Optimization:** This phase fine-tuned the model. The number of L-BFGS-B iterations varied, with some runs converging quickly (e.g., 3 iterations for LR $4.641589 \times 10^{-5}$) and others taking much longer (e.g., 139 iterations for LR $2.154435 \times 10^{-5}$). The convergence message "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH" was consistently observed, indicating successful convergence based on the specified tolerance.

The persistent challenge is the model's inability to accurately discover $\nu$, despite the high $\lambda_{pde}$ and the use of ground truth collocation points. The model appears to be trapped in a local minimum where the PDE residual is minimized, but the inferred $\nu$ is incorrect. This could be due to several factors:
*   **Over-constraining by PDE Loss:** A very high $\lambda_{pde}$ might force the network to satisfy the PDE at the expense of accurately learning the underlying physics, leading to a suboptimal $\nu$ value.
*   **Insensitivity to $\nu$:** The loss landscape might be relatively flat with respect to $\nu$ in the vicinity of the incorrect values, making it difficult for the optimizers to find the true $\nu$.
*   **Network Capacity:** While the network architecture is relatively deep, it might still lack the capacity or flexibility to simultaneously satisfy the data and PDE constraints while accurately inferring $\nu$.
*   **Initial Guess Dependence:** The consistent convergence to values around 0.02, close to the initial guess, suggests a strong dependence on the initialization of $\nu$.

The results indicate that simply increasing the PDE loss weight or varying the learning rate within this range is not sufficient to overcome the challenge of accurate $\nu$ discovery when the true value is 0.05 and the initial guess is 0.02. Further investigation into the balance of loss terms, network architecture, and initialization strategies is warranted.

## Tuning Roadmap

1.  **Re-evaluate Loss Weighting Strategy:** The current high $\lambda_{pde}$ does not yield better $\nu$ discovery.
    *   **Action:** Experiment with a more balanced approach to $\lambda_{data}$ and $\lambda_{pde}$, potentially exploring adaptive weighting schemes that dynamically adjust the weights during training based on the magnitudes of the individual losses.
    *   **Action:** Consider curriculum learning strategies where the PDE weight is gradually increased.
2.  **Network Capacity and Architecture:** The current network might not be optimal for this problem.
    *   **Action:** Systematically explore different numbers of hidden layers and neurons per layer.
    *   **Action:** Investigate alternative activation functions (e.g., Swish, GELU) that might offer better gradient flow.
3.  **Initial $\nu$ Guess Exploration:** The model's strong dependence on the initial $\nu$ guess is a concern.
    *   **Action:** Implement a more sophisticated automated initial guess search strategy that explores a wider range of $\nu$ values and uses a more robust metric than just initial loss.
    *   **Action:** Analyze the loss landscape around the true $\nu$ and the converged $\nu$ values to understand the optimization challenges.
4.  **Optimizer Parameters and Alternatives:**
    *   **Action:** Further fine-tune L-BFGS-B options, especially `ftol`, to ensure it is not stopping prematurely.
    *   **Action:** Consider alternative advanced optimization techniques beyond Adam + L-BFGS-B, such as those specifically designed for PINNs or inverse problems.
5.  **Collocation Point Sampling:** The current uniform sampling might not be ideal.
    *   **Action:** Investigate adaptive sampling strategies for collocation points, potentially focusing on regions with high PDE residuals or areas where the solution changes rapidly.
6.  **Learning Rate Schedule Refinement:** While varying learning rates were tested, a more structured schedule might be beneficial.
    *   **Action:** Implement and test different learning rate schedules (e.g., exponential decay, cosine annealing) for the Adam optimizer to see if it improves convergence and accuracy.

```

```
<br><sub>Last edited: 2025-08-27 08:33:44</sub>
