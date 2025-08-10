# Review 007: Ground Truth Collocation Points for Nu Discovery with $\lambda_{pde} = 100.0$

## Objective

The primary objective of this experiment is to modify the Physics-Informed Neural Network (PINN) model for the 2D Burgers' equation to utilize the "Ground Truth 41x41 points" (totaling 6724 points) for *both* the data fidelity loss and the PDE residual loss, with an increased PDE loss weight of $\lambda_{pde} = 100.0$. This builds upon the previous experiment ($\lambda_{pde} = 10.0$) to further investigate the impact of a stronger PDE constraint on the convergence and accuracy of kinematic viscosity $\nu$ discovery across a range of learning rates.

## Experiment Parameters

* **Learning Rates:** 10 logarithmically spaced values from $10^{-5}$ to $10^{-2}$ (obtained using `np.logspace(np.log10(1e-5), np.log10(1e-2), 10)`).
* **Grid Points (x, y):** 41x41
* **Time Steps:** 50
* **True Kinematic Viscosity $\nu$:** 0.05
* **Spatial Domain (x, y):** [0.0, 2.0] x [0.0, 2.0]
* **Time Domain (t):** [0.0, 0.05]
* **Neural Network Layers:** [3, 60, 60, 60, 60, 2] (3 input, 4 hidden layers with 60 neurons each, 2 output)
* **Adam Epochs (Full Loss):** 2000
* **Data-Only Pre-training Epochs:** 10000
* **Data Loss Weight ($\lambda_{data}$):** 1.0
* **PDE Loss Weight ($\lambda_{pde}$):** 100.0
* **Number of PDE Collocation Points:** 6724 (changed from 60000)
* **PDE Collocation Points Source:** The `x_pde`, `y_pde`, `t_pde` tensors are now directly assigned from the `x_data_tf`, `y_data_tf`, `t_data_tf` tensors, ensuring that the same 41x41 ground truth points (6724 points) are used for both data and PDE loss terms.
* **L-BFGS-B Options:** `maxiter`: 100000, `maxfun`: 100000, `maxcor`: 100, `maxls`: 50, `ftol`: 1e-20
* **Initial Kinematic Viscosity ($u$) Value:** 0.02 (explicitly set for robustness test)
  * For this experiment, the initial value for the kinematic viscosity ($u$) is explicitly set to 0.02 at the beginning of the `train` method. This is done to test the model's robustness starting from a fixed, non-optimal initial value, rather than performing a preliminary search for an optimal starting $u$ from a list of candidates. This fixed initial value is used for the `log_nu_pinn` trainable variable in all subsequent training phases.

## Code Modifications in `main_coloc.py`

For this experiment, the `main_coloc.py` script will be modified to run 10 experiments, each with a different learning rate, and with the PDE loss weight ($\lambda_{pde}$) set to 100.0. The modifications include:

1. **Reintroduction of Multiple Runs:**
   
   * The `learning_rates_to_test` array will be reintroduced with 10 logarithmically spaced values.
   * The `num_runs` variable will be set to `10`.
   * The `for i in range(num_runs):` loop will be reinstated to iterate through each learning rate and run the training process.
   * The random seed will be set dynamically based on the loop iteration (`seed_value = i + 1`).

2. **PDE Loss Weight Update:**
   
   * The `lambda_pde_weight` will be set to `100.0`.

3. **Reversion of Logging Changes:**
   
   * The enhanced logging for data-only phase final loss, Adam phase final loss/nu, and L-BFGS-B iteration prints will be reverted to their original, less verbose state.
   * The Adam print interval will be reverted to every 100 epochs.
   * The detailed multi-line comment for `RESULT_RUN` will be replaced with its original single-line comment.

## Results

| Run | Learning Rate | Discovered $\nu$ | Relative Error | MSE (Total Loss) | Data-Only Duration (s) | Adam Duration (s) | L-BFGS-B Duration (s) |
|:--- |:------------- |:---------------- |:-------------- |:---------------- |:---------------------- |:----------------- |:--------------------- |
| 1   | 1.000000e-05  | 0.019365         | 0.6127         | 8.681795e-02     | 20.12                  | 51.68             | 36.30                 |
| 2   | 2.154435e-05  | 0.020170         | 0.5966         | 8.264788e-02     | 17.66                  | 48.08             | 66.04                 |
| 3   | 4.641589e-05  | 0.020089         | 0.5982         | 5.399071e-02     | 20.19                  | 52.56             | 15.07                 |
| 4   | 1.000000e-04  | 0.020369         | 0.5926         | 4.768075e-02     | 20.22                  | 52.34             | 6.92                  |
| 5   | 2.154435e-04  | 0.020410         | 0.5918         | 4.841442e-02     | 19.75                  | 51.48             | 7.88                  |
| 6   | 4.641589e-04  | 0.020905         | 0.5819         | 3.646851e-02     | 20.23                  | 52.01             | 13.75                 |
| 7   | 1.000000e-03  | 0.019732         | 0.6054         | 1.736851e-02     | 20.32                  | 51.62             | 3.99                  |
| 8   | 2.154435e-03  | 0.019274         | 0.6145         | 2.628232e-02     | 20.23                  | 52.26             | 27.41                 |
| 9   | 4.641589e-03  | 0.014094         | 0.7181         | 9.417738e-03     | 20.22                  | 52.36             | 134.94                |
| 10  | 1.000000e-02  | 0.000876         | 0.9825         | 1.524186e-02     | 19.97                  | 52.55             | 6929.36               |

## Discussion of Results

Results for this experiment are pending. This section will be updated after all runs are completed and the data is collected.

## Profiling and Analysis

Profiling and analysis for this experiment are pending. This section will be updated after all runs are completed and the data is collected.

## Tuning Roadmap

1. **Analyze Impact of Increased $\lambda_{pde}$:** Compare the results from this experiment ($\lambda_{pde} = 100.0$) with the previous experiment ($\lambda_{pde} = 10.0$) to understand the effect of a stronger PDE constraint on $\nu$ discovery, convergence, and training times.
2. **Investigate $\nu$ Sensitivity:** Continue to assess if the loss function is sufficiently sensitive to the $\nu$ parameter, especially with the increased PDE weight.
   * **Action:** Further experiment with different weighting strategies for $\lambda_{data}$ and $\lambda_{pde}$ if the current results do not show significant improvement.
   * **Action:** Explore alternative loss function formulations or regularization techniques that specifically target the accuracy of parameter discovery.
3. **Initial $\nu$ Guess:** The initial $\nu$ is explicitly set to 0.02 for robustness testing.
   * **Action:** Re-evaluate the impact of this fixed initial guess. Consider if a different fixed initial value or a dynamic initialization strategy would be beneficial.
4. **Optimizer Parameters:**
   * **Action:** Fine-tune L-BFGS-B options, particularly `maxiter` and `ftol`, if convergence issues persist or if more accurate $\nu$ values are desired.
   * **Action:** Explore other advanced optimization techniques beyond Adam + L-BFGS-B.
5. **Network Architecture:**
   * **Action:** Experiment with different numbers of layers and neurons per layer to see if a more complex or simpler network can better capture the underlying physics and discover $\nu$ more accurately.
6. **Random Seeds:** Run multiple experiments with different random seeds for each learning rate to assess the robustness and average performance, especially given the potential for variability in L-BFGS-B convergence.
<br><sub>Last edited: 2025-08-24 09:15:56</sub>
