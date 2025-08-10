# Review 006: Ground Truth Collocation Points for Nu Discovery

## Objective

The primary objective of this experiment is to modify the Physics-Informed Neural Network (PINN) model for the 2D Burgers' equation to utilize the "Ground Truth 41x41 points" (totaling 6724 points) for *both* the data fidelity loss and the PDE residual loss. This replaces the previous approach where 60000 randomly sampled collocation points were used for the PDE loss. The goal is to investigate the impact of using a consistent, physically meaningful set of points for both loss terms on the convergence and accuracy of kinematic viscosity $\nu$ discovery.

## Experiment Parameters

* **Learning Rates:** 10 logarithmically spaced values from $10^{-5}$ to $10^{-2}$ (obtained using `np.logspace(np.log10(1e-5), np.log10(1e-2), 10)`).
* **Grid Points (x, y):** 41x41
* **Time Steps:** 50
* **True Kinematic Viscosity $\nu$:** 0.05
* **Spatial Domain (x, y):** [0.0, 2.0] x [0.0, 2.0]
* **Time Domain (t):** [0.0, 0.05]
* **Neural Network Layers:** [3, 60, 60, 60, 60, 2] (3 input, 4 hidden layers with 60 neurons each, 2 output)
* **Adam Epochs (Full Loss):** 2000 (temporarily set to 100 for initial testing)
* **Data-Only Pre-training Epochs:** 10000 (temporarily set to 500 for initial testing)
* **Data Loss Weight ($\lambda_{data}$):** 1.0
* **PDE Loss Weight ($\lambda_{pde}$):** 1.0
* **Number of PDE Collocation Points:** 6724 (changed from 60000)
* **PDE Collocation Points Source:** The `x_pde`, `y_pde`, `t_pde` tensors are now directly assigned from the `x_data_tf`, `y_data_tf`, `t_data_tf` tensors, ensuring that the same 41x41 ground truth points (6724 points) are used for both data and PDE loss terms.
* **L-BFGS-B Options:** `maxiter`: 100000, `maxfun`: 100000, `maxcor`: 100, `maxls`: 50, `ftol`: 1e-20
* **Initial Kinematic Viscosity ($u$) Value:** 0.02 (explicitly set for robustness test)
  * For this experiment, the initial value for the kinematic viscosity ($u$) is explicitly set to 0.02 at the beginning of the `train` method. This is done to test the model's robustness starting from a fixed, non-optimal initial value, rather than performing a preliminary search for an optimal starting $u$ from a list of candidates. This fixed initial value is used for the `log_nu_pinn` trainable variable in all subsequent training phases.

## Code Modifications in `main_coloc.py`

The following precise modifications were implemented in `main_coloc.py`:

1. **`num_pde_points` Update:**
   
   * **Old:** `num_pde_points = 60000`
   * **New:** `num_pde_points = 6724`

2. **PDE Collocation Points Assignment:**
   
   * **Old (inside the loop):**
     
     ```python
         x_pde = tf.constant(np.random.uniform(x_min, x_max, (num_pde_points, 1)), dtype=tf.float32)
         y_pde = tf.constant(np.random.uniform(y_min, y_max, (num_pde_points, 1)), dtype=tf.float32)
         t_pde = tf.constant(np.random.uniform(t_min, t_max, (num_pde_points, 1)), dtype=tf.float32)
         x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
         y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
         t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
         u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
         v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)
     ```
   
   * **New (inside the loop, after `tf_v1.reset_default_graph()`):**
     
     ```python
         x_data_tf = tf.constant(X_data_flat, dtype=tf.float32)
         y_data_tf = tf.constant(Y_data_flat, dtype=tf.float32)
         t_data_tf = tf.constant(T_data_flat, dtype=tf.float32)
         u_data_tf = tf.constant(U_data_flat, dtype=tf.float32)
         v_data_tf = tf.constant(V_data_flat, dtype=tf.float32)
         x_pde = x_data_tf
         y_pde = y_data_tf
         t_pde = t_data_tf
     ```
   
   * **Rationale:** This change ensures that the PDE collocation points are identical to the data points.

3. **Epochs and Runs for Testing:**
   
   * `adam_epochs = 2000`, `epochs_data_only = 10000`` .

## Results

| Run | Learning Rate | Discovered $\nu$ | Relative Error | MSE (Total Loss) | Data-Only Duration (s) | Adam Duration (s) | L-BFGS-B Duration (s) |
|:--- |:------------- |:---------------- |:-------------- |:---------------- |:---------------------- |:----------------- |:--------------------- |
| 1   | 1.000000e-05  | 0.017434         | 0.6513         | 1.784223e-03     | 20.50                  | 52.33             | 628.26                |
| 2   | 2.154435e-05  | 0.022522         | 0.5496         | 6.122849e-04     | 16.16                  | 308.35            | 1343.17               |
| 3   | 4.641589e-05  | 0.019970         | 0.6006         | 3.806449e-03     | 15.13                  | 308.44            | 10.96                 |
| 4   | 1.000000e-04  | 0.020846         | 0.5831         | 9.567895e-04     | 16.31                  | 308.32            | 253.45                |
| 5   | 2.154435e-04  | 0.021029         | 0.5794         | 1.187133e-03     | 16.74                  | 308.44            | 19.53                 |
| 6   | 4.641589e-04  | 0.019792         | 0.6042         | 9.937070e-04     | 20.76                  | 308.27            | 10.99                 |
| 7   | 1.000000e-03  | 0.022926         | 0.5415         | 2.815019e-04     | 17.66                  | 307.76            | 121.70                |
| 8   | 2.154435e-03  | 0.018309         | 0.6338         | 1.457183e-04     | 20.30                  | 308.55            | 88.98                 |
| 9   | 4.641589e-03  | 0.016604         | 0.6679         | 7.927192e-05     | 20.49                  | 308.37            | 242.02                |
| 10  | 1.000000e-02  | 0.011506         | 0.7699         | 1.296202e-04     | 14.58                  | 308.34            | 243.59                |

## Discussion of Results

The experiment aimed to investigate the impact of using ground truth points for PDE collocation on the discovery of kinematic viscosity $\nu$. The true $\nu$ 0.05.

For the first run, with a learning rate of $1.000000e-05$, the discovered $\nu$ is $0.017434$, resulting in a relative error of $0.6513$. The MSE (Total Loss) is $1.784223e-03$. Compared to the previous configuration (60000 random PDE points), the discovered $\nu$ for the first run is slightly higher (0.017434 vs 0.016964), and the relative error is slightly lower (0.6513 vs 0.6607). The total MSE is also slightly higher. This initial observation suggests that using ground truth points for PDE collocation might lead to a slightly different optimization path and potentially a different discovered $\nu$. However, a comprehensive comparison requires running all 10 experiments with the new configuration.

The overall trend of underestimation of $\nu$ persists, with the discovered values still significantly lower than the true value of 0.05. The relative errors remain high, indicating that the model continues to struggle with accurately identifying the kinematic viscosity.

## Profiling and Analysis

* **Data-Only Duration (s):** This phase consistently takes around 15-20 seconds across all runs, indicating that the initial data-only pre-training is not significantly affected by the learning rate variations in the Adam phase.
* **Adam Duration (s):** For the first run with the new configuration, the Adam optimization phase took 52.33 seconds, which is significantly less than the previous 305-308 seconds. This is due to the temporary reduction of `adam_epochs` to 100 for testing purposes. For the full experiment, this duration will revert to the expected range.
* **L-BFGS-B Duration (s):** For the first run, the L-BFGS-B optimization phase took 628.26 seconds. This is within the range of variability observed in previous experiments, but a full comparison will be possible after all runs are completed.
* **Overall Performance:** The initial test run with reduced epochs and `num_runs = 1` completed much faster, confirming the code modifications are functional. The full experiment will provide a more accurate picture of the performance with the new collocation point strategy.

## Tuning Roadmap

1. **Complete Full Experiment:** Run all 10 experiments with the new ground truth collocation point configuration to gather a complete set of results for a thorough analysis and comparison.
2. **Investigate $\nu$ Sensitivity:** The high relative error despite low total loss suggests that the loss function might not be sufficiently sensitive to the $\nu$ parameter.
   * **Action:** Experiment with different weighting strategies for $\\\lambda_{data}\\$ and $\\\lambda_{pde}\\$ . Increase $\\\lambda_{pde}\\$ to put more emphasis on satisfying the PDE, which directly involves $\nu$.
   * **Action:** Explore alternative loss function formulations or regularization techniques that specifically target the accuracy of parameter discovery.
3. **Initial $\nu$ Guess:** The initial $\nu$ is explicitly set to 0.02 for robustness testing.
   * **Action:** Investigate the impact of this fixed initial guess. Consider if a different fixed initial value could lead to better convergence or if a dynamic initialization strategy (e.g., based on prior knowledge or a quick pre-training phase with a wider range of initial guesses) would be beneficial for future experiments.
4. **Optimizer Parameters:**
   * **Action:** Fine-tune L-BFGS-B options, particularly `maxiter` and `ftol`, to see if it improves convergence to a more accurate $\nu$.
   * **Action:** Explore other advanced optimization techniques beyond Adam + L-BFGS-B, such as those specifically designed for inverse problems or parameter discovery in PINNs.
5. **Network Architecture:**
   * **Action:** Experiment with different numbers of layers and neurons per layer to see if a more complex or simpler network can better capture the underlying physics and discover $\nu$ more accurately.
6. **Random Seeds:** While seeds are set for reproducibility, the variability in L-BFGS-B duration suggests that even with fixed seeds, the optimization path can be sensitive.
   * **Action:** Run multiple experiments with different random seeds for each learning rate to assess the robustness and average performance.
<br><sub>Last edited: 2025-08-23 18:48:12</sub>
