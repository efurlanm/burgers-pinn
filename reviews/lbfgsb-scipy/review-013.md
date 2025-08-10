# Review 013: Continued L-BFGS-B Non-Convergence with Increased Adam Epochs

## Precise and Detailed Analysis

The `burgers2d-03.py` script was executed again after increasing `epochs_adam` to `2000`. The execution completed without any `stderr` warnings.

Analysis of the updated `log.txt` reveals the following:

-   **Adam Optimizer Phase:**
    -   The initial `nu_pinn` guess was `0.100000`.
    -   Adam training ran for 1900 epochs.
    -   The loss decreased from `97.679146` to `0.073912`.
    -   The discovered `nu` value after Adam training was `0.102900`.

-   **L-BFGS-B Optimizer Phase:**
    -   L-BFGS-B training finished.
    -   **Convergence:** `L-BFGS-B converged: False`. Despite increasing Adam epochs, the L-BFGS-B optimizer still did not converge.
    -   **Number of Iterations:** `L-BFGS-B number of iterations: 34`. This is still extremely low, indicating that the optimizer stopped almost immediately.
    -   **Function Evaluations:** `L-BFGS-B function evaluations: 193`.
    -   **Final Discovered `nu`:** `0.103118`. This value is still far from the `True nu: 0.05` and very close to the `nu` value after Adam training, suggesting that L-BFGS-B performed minimal effective updates.

Increasing Adam epochs did not lead to L-BFGS-B convergence. The L-BFGS-B optimizer is still stopping almost immediately, suggesting that even with more Adam pre-training, the optimization landscape for the larger network (40 neurons, 40000 PDE points) remains challenging for L-BFGS-B to navigate to the specified tolerance (`1e-3`). The `nu` value is consistently stuck around the initial guess and Adam's output.

## Next Steps

1.  **Re-evaluate L-BFGS-B `tolerance`:** Since L-BFGS-B is still not converging even with a relaxed tolerance of `1e-3`, and it's stopping almost immediately, it's highly probable that the tolerance is still too strict for the current problem setup. I will try relaxing it further to `1e-2`.

2.  **Consider `max_iterations`:** Although it's not hitting `max_iterations`, setting it to a very high value (e.g., `500000`) ensures that if the tolerance is eventually met, it has enough iterations.

3.  **Explore alternative `nu` initialization:** Since `nu` is consistently stuck around 0.1, it might be beneficial to try initializing `log_nu_pinn` closer to the true value (0.05) for testing purposes, to see if it can converge from a better starting point. This would help diagnose if the issue is with the optimization landscape around 0.1 or a more general problem.

4.  **Document Findings in `review-013.md`:** Create a new review file to document these observations and the plan for further investigation.

---

## Insights from Internet Search on L-BFGS-B and PINN Convergence

The internet search on common issues with `tfp.optimizer.lbfgs_minimize` in PINNs highlights several challenges that align with our observations:

*   **Ill-Conditioned Loss Landscapes:** PINN loss functions, especially for PDEs, can be highly ill-conditioned, making it difficult for optimizers like L-BFGS-B to converge.
*   **Local Minima and Saddle Points:** L-BFGS-B is susceptible to getting trapped in local minima, leading to suboptimal solutions.
*   **Sensitivity to Initialization:** The optimizer's performance is highly dependent on the initial values of the network's weights and biases.
*   **Stalling of Optimization:** Even with Adam pre-training, L-BFGS-B can stall, meaning the loss stops decreasing significantly.
*   **Parameter Identification Challenges:** Inferring unknown physical parameters (like `nu`) is an inverse problem and can be particularly difficult if the solution is not very sensitive to that parameter.

Our current situation, where L-BFGS-B is stopping almost immediately (very few iterations) despite increased Adam epochs and a larger network, strongly suggests that:

1.  **The `tolerance` is still too strict:** The optimizer is likely hitting a very flat region in the loss landscape or a local minimum where the gradient is below the current tolerance, causing it to terminate prematurely.
2.  **Adam pre-training might still be insufficient:** Even with more epochs, Adam might not be guiding the network to a sufficiently "good" region for L-BFGS-B to effectively optimize.
3.  **The `nu` parameter's identification is challenging:** The loss function might be less sensitive to `nu` in the region where the optimization is occurring, making it hard for the optimizer to pinpoint the correct value.

Given these insights, the next logical step is to continue relaxing the L-BFGS-B `tolerance` and to consider the impact of `nu` initialization more directly.

<br><sub>Last edited: 2025-08-03 16:53:23</sub>
