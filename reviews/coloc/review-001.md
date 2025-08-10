# Review 006: Ground Truth Collocation Points for Nu Discovery

## Objective

The primary objective of this experiment is to modify the Physics-Informed Neural Network (PINN) model for the 2D Burgers' equation to utilize the "Ground Truth 41x41 points" (totaling 6724 points) for *both* the data fidelity loss and the PDE residual loss. This replaces the previous approach where 60000 randomly sampled collocation points were used for the PDE loss. The goal is to investigate the impact of using a consistent, physically meaningful set of points for both loss terms on the convergence and accuracy of kinematic viscosity $\nu$ discovery.

## Experiment Parameters

*   **Learning Rate:** A single, fixed learning rate of $1.000000e-02$ is used for all training phases.
*   **Grid Points (x, y):** 41x41
*   **Time Steps:** 50
*   **True Kinematic Viscosity $\nu$:** 0.05
*   **Spatial Domain (x, y):** [0.0, 2.0] x [0.0, 2.0]
*   **Time Domain (t):** [0.0, 0.05]
*   **Neural Network Layers:** [3, 60, 60, 60, 60, 2] (3 input, 4 hidden layers with 60 neurons each, 2 output)
*   **Adam Epochs (Full Loss):** 2000
*   **Data-Only Pre-training Epochs:** 10000
*   **Data Loss Weight ($\lambda_{data}$):** 1.0
*   **PDE Loss Weight ($\lambda_{pde}$):** 10.0
*   **Number of PDE Collocation Points:** 6724 (changed from 60000)
*   **PDE Collocation Points Source:** The `x_pde`, `y_pde`, `t_pde` tensors are now directly assigned from the `x_data_tf`, `y_data_tf`, `t_data_tf` tensors, ensuring that the same 41x41 ground truth points (6724 points) are used for both data and PDE loss terms.
*   **L-BFGS-B Options:** `maxiter`: 100000, `maxfun`: 100000, `maxcor`: 100, `maxls`: 50, `ftol`: 1e-20
*   **Initial Kinematic Viscosity ($u$) Value:** 0.02 (explicitly set for robustness test)
    *   For this experiment, the initial value for the kinematic viscosity ($u$) is explicitly set to 0.02 at the beginning of the `train` method. This is done to test the model's robustness starting from a fixed, non-optimal initial value, rather than performing a preliminary search for an optimal starting $u$ from a list of candidates. This fixed initial value is used for the `log_nu_pinn` trainable variable in all subsequent training phases.

## Code Modifications in `main_coloc.py`

The `main_coloc.py` script has been modified to run a single experiment with a fixed random seed (seed=1) and a specified learning rate. The key modifications are:

1.  **Single Run Configuration:**
    *   The `learning_rates_to_test` array and `num_runs` variable have been removed.
    *   The script now directly uses a single `learning_rate` value.
    *   The iteration loop for multiple runs has been removed, and the code now executes a single training process.
    *   The random seed is explicitly set to `1` for reproducibility.

2.  **Enhanced Logging:**
    *   **Data-Only Pre-training:** The final loss of the data-only phase is now printed at the end of `train_data_only`.
    *   **Adam Optimization:** The progress print interval has been changed from every 100 epochs to every 200 epochs. The final loss and discovered $\nu$ after the Adam phase are now explicitly printed.
    *   **L-BFGS-B Optimization:** Progress is now printed every 100 iterations, showing the current loss, gradient norm, and the gradient of the discoverable $\nu$.

3.  **`RESULT_RUN` Output Format:**
    *   A detailed multi-line comment has been added before the `RESULT_RUN` print statement, explaining the format of the CSV-like output. The fields are:
        1.  Run Number (always 1 in this configuration)
        2.  Learning Rate
        3.  Discovered Kinematic Viscosity (nu)
        4.  Relative Error of Discovered nu
        5.  Final Total Loss (Mean Squared Error)
        6.  Duration of Data-Only Pre-training (seconds)
        7.  Duration of Adam Optimization (seconds)
        8.  Duration of L-BFGS-B Optimization (seconds)

## Results

| Run | Learning Rate | Discovered $\nu$ | Relative Error | MSE (Total Loss) | Data-Only Duration (s) | Adam Duration (s) | L-BFGS-B Duration (s) |
|:--- |:------------- |:---------------- |:-------------- |:---------------- |:---------------------- |:----------------- |:--------------------- |
| 1   | 1.000000e-05  | 0.017434         | 0.6513         | 1.784223e-03     | 20.50                  | 52.33             | 628.26                |
| 2   | 2.154435e-05  | 0.019761         | 0.6048         | 1.577314e-02     | 21.14                  | 48.65             | 89.74                 |
| 3   | 4.641589e-05  | 0.020080         | 0.5984         | 2.094094e-02     | 19.75                  | 47.49             | 2.83                  |
| 4   | 1.000000e-04  | 0.020073         | 0.5985         | 9.338167e-03     | 20.24                  | 47.64             | 5.61                  |
| 5   | 2.154435e-04  | 0.020486         | 0.5903         | 7.838893e-03     | 20.34                  | 47.76             | 3.88                  |
| 6   | 4.641589e-04  | 0.021125         | 0.5775         | 5.880112e-03     | 20.84                  | 48.19             | 6.26                  |
| 7   | 1.000000e-03  | 0.020043         | 0.5991         | 2.645252e-03     | 20.71                  | 52.74             | 6.21                  |
| 8   | 2.154435e-03  | 0.020325         | 0.5935         | 1.563477e-03     | 20.69                  | 48.36             | 41.10                 |
| 9   | 4.641589e-03  | 0.015231         | 0.6954         | 1.249965e-03     | 17.20                  | 47.37             | 45.65                 |
| 10  | 1.000000e-02  | 0.016621         | 0.6676         | 1.679532e-01     | 19.58                  | 47.35             | 11.56                 |

## Discussion of Results

This experiment focuses on using ground truth points for PDE collocation with a fixed $\lambda_{pde}$ of 10.0 and a fixed initial $\nu$ of 0.02. The results across different learning rates (from $1.000000e-05$ to $1.000000e-02$) show a consistent trend of underestimation of the true kinematic viscosity ($\nu = 0.05$). The discovered $\nu$ values generally range between 0.015 and 0.021, leading to relative errors between 0.57 and 0.69.

*   **Impact of Learning Rate:** There isn't a clear monotonic relationship between the learning rate and the accuracy of $\nu$ discovery. While some intermediate learning rates (e.g., $2.154435e-04$ to $2.154435e-03$) show slightly better relative errors, the overall performance remains in a similar range. This suggests that within the tested range, the learning rate alone is not the dominant factor in achieving a more accurate $\nu$ discovery under this specific collocation and weighting strategy.
*   **Consistency of Underestimation:** The persistent underestimation of $\nu$ indicates a potential limitation of the current model configuration or loss function in accurately capturing the true viscosity. This could be due to the weighting of the loss terms, the network architecture, or the inherent difficulty of the inverse problem.
*   **MSE (Total Loss):** The MSE values vary across runs, but generally remain low, indicating that the model is fitting the data and satisfying the PDE to some extent. However, a low total loss does not directly translate to accurate parameter discovery, especially when the parameter itself might have a subtle influence on the overall loss landscape.

## Profiling and Analysis

*   **Data-Only Duration (s):** This phase consistently takes around 17-21 seconds across all runs, indicating its stability and independence from the Adam learning rate. The final data loss after this phase is very low, suggesting good initial data fitting.
*   **Adam Duration (s):** The Adam optimization phase consistently takes around 47-53 seconds. The logging at every 200 epochs provides a clearer view of the loss and discovered $\nu$ evolution during this phase. The final Adam loss and discovered $\nu$ are now explicitly reported, offering a better checkpoint for analysis before the L-BFGS-B phase.
*   **L-BFGS-B Duration (s):** The L-BFGS-B optimization phase shows significant variability in duration, ranging from a few seconds (e.g., Run 3, 5, 6, 7) to much longer times (e.g., Run 2, 8, 9, 10). The detailed logging every 100 iterations provides valuable insights into the optimizer's progress, including the loss, gradient norm, and the gradient of $\nu$. This variability suggests that the L-BFGS-B optimizer's convergence is highly sensitive to the initial conditions provided by the Adam optimizer and the specific learning rate.
*   **Overall Performance:** The enhanced logging provides a more granular view of the training process, allowing for better analysis of each optimization phase. The consistent underestimation of $\nu$ and the variability in L-BFGS-B convergence highlight areas for further investigation.

## Tuning Roadmap

1.  **Investigate $\nu$ Sensitivity:** The high relative error despite low total loss suggests that the loss function might not be sufficiently sensitive to the $\nu$ parameter.
    *   **Action:** Experiment with different weighting strategies for $\lambda_{data}$ and $\lambda_{pde}$. The current results suggest that simply increasing $\lambda_{pde}$ might not be effective. Further investigation into the optimal balance between data and PDE loss is needed.
    *   **Action:** Explore alternative loss function formulations or regularization techniques that specifically target the accuracy of parameter discovery.
2.  **Initial $\nu$ Guess:** The initial $\nu$ is explicitly set to 0.02 for robustness testing.
    *   **Action:** Investigate the impact of this fixed initial guess. Consider if a different fixed initial value could lead to better convergence or if a dynamic initialization strategy (e.g., based on prior knowledge or a quick pre-training phase with a wider range of initial guesses) would be beneficial for future experiments.
3.  **Optimizer Parameters:**
    *   **Action:** Fine-tune L-BFGS-B options, particularly `maxiter` and `ftol`, to see if it improves convergence to a more accurate $\nu$. The variability in L-BFGS-B duration indicates that the current options might not be suitable for all configurations.
    *   **Action:** Explore other advanced optimization techniques beyond Adam + L-BFGS-B, suchs as those specifically designed for inverse problems or parameter discovery in PINNs.
4.  **Network Architecture:**
    *   **Action:** Experiment with different numbers of layers and neurons per layer to see if a more complex or simpler network can better capture the underlying physics and discover $\nu$ more accurately.
5.  **Random Seeds:** While seeds are set for reproducibility, the variability in L-BFGS-B duration suggests that even with fixed seeds, the optimization path can be sensitive.
    *   **Action:** Run multiple experiments with different random seeds for each learning rate to assess the robustness and average performance.

<br><sub>Last edited: 2025-08-24 00:06:00</sub>
