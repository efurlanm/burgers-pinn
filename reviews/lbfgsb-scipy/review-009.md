# Review 009: Impact of Input Scaling and Continued L-BFGS-B Non-Convergence

## Precise and Detailed Analysis

The `burgers2d-03.py` script was executed again after implementing input scaling for `x`, `y`, and `t` and restoring the Adam epochs to `1000`. The execution completed without any `stderr` warnings.

Analysis of the updated `log.txt` reveals the following:

-   **Adam Optimizer Phase:**
    -   The initial `nu_pinn` guess was `0.100000`.
    -   Adam training ran for 900 epochs (as expected, confirming the fix for the previous issue).
    -   The loss decreased significantly from `145.428833` to `0.111334`. This is a much better initial loss and reduction compared to previous runs without input scaling, indicating that input scaling has a positive impact on the Adam pre-training.
    -   The discovered `nu` value after Adam training was `0.101258`, which is still close to the initial guess.

-   **L-BFGS-B Optimizer Phase:**
    -   L-BFGS-B training finished.
    -   **Convergence:** `L-BFGS-B converged: False`. Despite the improvements in Adam pre-training and input scaling, the L-BFGS-B optimizer still did not converge.
    -   **Number of Iterations:** `L-BFGS-B number of iterations: 12937`. This is still far from the `max_iterations` of `100000`, reinforcing that the optimizer is stopping due to the `tolerance` rather than hitting the iteration limit.
    -   **Function Evaluations:** `L-BFGS-B function evaluations: 38639`.
    -   **Final Discovered `nu`:** `0.035357`. This value is closer to the `True nu: 0.05` than in previous runs (e.g., `0.014110` or `0.020612`), which is a positive sign. However, it's still not accurate enough.

The implementation of input scaling significantly improved the Adam pre-training phase, leading to a much lower initial loss for L-BFGS-B and a `nu` value closer to the true value. However, the L-BFGS-B optimizer still fails to converge, indicating that the `tolerance` of `1e-5` might still be too strict, or there are other factors at play preventing it from reaching that level of precision.

**Clarification on `num_pde_points`:** It's important to distinguish `num_pde_points` from observational training data. `num_pde_points` refers to the number of *collocation points* where the PDE residual is enforced. These points do not require known solution values; they are locations where the neural network's output is constrained to satisfy the governing PDE. Increasing the density of these collocation points helps the network learn the underlying physics more accurately and consistently throughout the domain, providing more "physics supervision" rather than "data supervision." This is a standard practice for improving accuracy in PINNs, especially for higher-dimensional problems, and does not contradict the principle of using few observational data points.

## Next Steps

1.  **Relax L-BFGS-B `tolerance` further:**
    -   Given that L-BFGS-B is still not converging, the `tolerance` of `1e-5` might still be too aggressive. Try relaxing it to `1e-4` to see if it converges. This will help determine if the optimizer is simply unable to reach such a tight tolerance due to the problem's inherent complexity or numerical precision.

2.  **Consider the impact of `nu` initialization:**
    -   The `log_nu_pinn` is initialized with `tf.math.log(0.1)`. While Adam moves it slightly, L-BFGS-B then takes it to `0.035357`. It might be beneficial to experiment with different initial guesses for `nu` (e.g., closer to the true value if known for testing purposes) to see how sensitive the optimization is to this initial condition.

3.  **Review PINN Architecture and Data (Revisit with more focus):**
    -   Even with input scaling, the network might not be complex enough or the data distribution might not be optimal.
    -   **Neural Network Architecture:** Consider increasing the number of neurons per layer or adding more layers.
    -   **Data Density:** Increase `num_pde_points` to provide more constraints to the PDE residual.

4.  **Document Findings in `review-009.md`:** Create a new review file to document these observations and the plan for further investigation.

<br><sub>Last edited: 2025-08-03 16:07:40</sub>
