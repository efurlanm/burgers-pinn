# Review 005: Controlled Learning Rate Experiment for Nu Discovery

## Objective

The primary objective of this experiment is to systematically investigate the impact of a predefined set of learning rates on the convergence and accuracy of the Physics-Informed Neural Network (PINN) model in discovering the kinematic viscosity ($\nu$) of the 2D Burgers' equation. Unlike previous experiments that used randomly sampled learning rates, this review focuses on a controlled set of 10 logarithmically spaced learning rates within a commonly accepted range.

## Experiment Parameters

* **Learning Rates:** 10 logarithmically spaced values from $10^{-5}$ to $10^{-2}$ (obtained using `np.logspace(np.log10(1e-5), np.log10(1e-2), 10)`).
* **Grid Points (x, y):** 41x41
* **Time Steps:** 50
* **True Kinematic Viscosity ($\nu$):** 0.05
* **Spatial Domain (x, y):** [0.0, 2.0] x [0.0, 2.0]
* **Time Domain (t):** [0.0, 0.05]
* **Neural Network Layers:** [3, 60, 60, 60, 60, 2] (3 input, 4 hidden layers with 60 neurons each, 2 output)
* **Adam Epochs (Full Loss):** 2000
* **Data-Only Pre-training Epochs:** 10000
* **Data Loss Weight ($\lambda_{data}$):** 1.0
* **PDE Loss Weight ($\lambda_{pde}$):** 1.0
* **Number of PDE Collocation Points:** 60000
* **L-BFGS-B Options:** `maxiter`: 100000, `maxfun`: 100000, `maxcor`: 100, `maxls`: 50, `ftol`: 1e-20
* **Initial Kinematic Viscosity ($u$) Value:** 0.02 (explicitly set for robustness test)
  * For this experiment, the initial value for the kinematic viscosity ($
    u$) is explicitly set to 0.02 at the beginning of the `train` method. This is done to test the model's robustness starting from a fixed, non-optimal initial value, rather than performing a preliminary search for an optimal starting $
    u$ from a list of candidates. This fixed initial value is used for the `log_nu_pinn` trainable variable in all subsequent training phases.

## Results

| Run | Learning Rate | Discovered $\nu$ | Relative Error | MSE (Total Loss) | Data-Only Duration (s) | Adam Duration (s) | L-BFGS-B Duration (s) |
|:--- |:------------- |:---------------- |:-------------- |:---------------- |:---------------------- |:----------------- |:--------------------- |
| 1   | 1.000000e-05  | 0.016964         | 0.6607         | 1.375368e-03     | 20.17                  | 305.61            | 914.33                |
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

The experiment aimed to investigate the impact of varying learning rates on the discovery of kinematic viscosity ($\nu$). The true ($\nu$) is 0.05.

From the results, it's evident that the discovered ($\nu$) values are consistently lower than the true value, ranging from approximately 0.0115 to 0.0229. This indicates a systematic underestimation of ($\nu$) by the PINN model across the tested learning rates.

The relative error is quite high, ranging from 0.5415 to 0.7699, further highlighting the discrepancy between the discovered and true ($\nu$).

There isn't a clear trend where a specific learning rate consistently leads to a significantly lower relative error or a ($\nu$) closer to the true value. The lowest relative error (0.5415) was observed at a learning rate of 1.000000e-03 (Run 7), while the highest (0.7699) was at 1.000000e-02 (Run 10).

The MSE (Total Loss) values are relatively small, ranging from 7.927192e-05 to 3.806449e-03. A low total loss, despite a high relative error in ($\nu$) discovery, suggests that the model might be fitting the data and PDE residuals well, but the parameter discovery aspect is still challenging. This could indicate issues with the sensitivity of the loss function to ($\nu$), or a local minimum that the optimizer is converging to.

## Profiling and Analysis

* **Data-Only Duration (s):** This phase consistently takes around 15-20 seconds across all runs, indicating that the initial data-only pre-training is not significantly affected by the learning rate variations in the Adam phase.
* **Adam Duration (s):** The Adam optimization phase consistently takes around 305-308 seconds for all runs. This is expected as the number of Adam epochs is fixed at 2000.
* **L-BFGS-B Duration (s):** The L-BFGS-B optimization phase shows significant variation, ranging from approximately 10.96 seconds (Run 3, 6) to 1343.17 seconds (Run 2). This variability suggests that the initial conditions provided by the Adam optimizer (which are influenced by the learning rate) have a substantial impact on the convergence speed of L-BFGS-B. A longer L-BFGS-B duration might indicate a more complex optimization landscape or a starting point further from the optimal solution.
* **Overall Performance:** The total training time is dominated by the L-BFGS-B phase, especially for runs with longer L-BFGS-B durations.

## Tuning Roadmap

1. **Investigate $\nu$ Sensitivity:** The high relative error despite low total loss suggests that the loss function might not be sufficiently sensitive to the $\nu$ parameter.
   * **Action:** Experiment with different weighting strategies for $\lambda_{data}$ and $\lambda_{pde}$. Increase $\lambda_{pde}$  to put more emphasis on satisfying the PDE, which directly involves $\nu$.
   * **Action:** Explore alternative loss function formulations or regularization techniques that specifically target the accuracy of parameter discovery.
2. **Initial $\nu$ Guess:** The initial $\nu$ is explicitly set to 0.02 for robustness testing.
   * **Action:** Investigate the impact of this fixed initial guess. Consider if a different fixed initial value could lead to better convergence or if a dynamic initialization strategy (e.g., based on prior knowledge or a quick pre-training phase with a wider range of initial guesses) would be beneficial for future experiments.
3. **Optimizer Parameters:**
   * **Action:** Fine-tune L-BFGS-B options, particularly `maxiter` and `ftol`, to see if it improves convergence to a more accurate $\nu$.
   * **Action:** Explore other advanced optimization techniques beyond Adam + L-BFGS-B, such as those specifically designed for inverse problems or parameter discovery in PINNs.
4. **Network Architecture:**
   * **Action:** Experiment with different numbers of layers and neurons per layer to see if a more complex or simpler network can better capture the underlying physics and discover $\nu$ more accurately.
5. **Data and PDE Point Distribution:**
   * **Action:** Investigate the impact of the number and distribution of data points and PDE collocation points. A denser or more strategically placed distribution might improve $\nu$ discovery.
6. **Random Seeds:** While seeds are set for reproducibility, the variability in L-BFGS-B duration suggests that even with fixed seeds, the optimization path can be sensitive.
   * **Action:** Run multiple experiments with different random seeds for each learning rate to assess the robustness and average performance.

<br><sub>Last edited: 2025-08-23 09:51:44</sub>
