# Review 012: Persistent L-BFGS-B Non-Convergence with Relaxed Tolerance and Increased Network Complexity

## Precise and Detailed Analysis

The `burgers2d-03.py` script was executed again after relaxing the L-BFGS-B `tolerance` to `1e-3`. The execution completed without any `stderr` warnings.

Analysis of the updated `log.txt` reveals the following:

-   **Adam Optimizer Phase:**
    -   The initial `nu_pinn` guess was `0.100000`.
    -   Adam training ran for 900 epochs.
    -   The loss decreased from `25.974913` to `0.076826`.
    -   The discovered `nu` value after Adam training was `0.099359`.

-   **L-BFGS-B Optimizer Phase:**
    -   L-BFGS-B training finished.
    -   **Convergence:** `L-BFGS-B converged: False`. Despite relaxing the `tolerance` further, the optimizer still did not converge.
    -   **Number of Iterations:** `L-BFGS-B number of iterations: 48`. This is still extremely low, indicating that the optimizer stopped almost immediately, likely due to hitting an internal stopping criterion related to the `tolerance` or a very flat region in the loss landscape.
    -   **Function Evaluations:** `L-BFGS-B function evaluations: 259`.
    -   **Final Discovered `nu`:** `0.099311`. This value is very close to the `nu` value after Adam training, suggesting that L-BFGS-B performed very few effective updates.

The persistent non-convergence of L-BFGS-B, even with a relaxed tolerance, and its immediate stopping, indicates that the optimization landscape for the larger network (40 neurons, 40000 PDE points) is very challenging for L-BFGS-B to navigate from the Adam-trained state. The Adam pre-training might not be sufficient to place the optimization in a region where L-BFGS-B can effectively converge.

## Next Steps

1.  **Increase Adam Epochs Significantly:**
    -   Since L-BFGS-B is not making significant progress, it's crucial to ensure the Adam optimizer provides a much better starting point. I will increase `epochs_adam` to `2000`. This will allow Adam more time to reduce the loss and potentially guide the network closer to a more favorable region for L-BFGS-B.

2.  **Document Findings in `review-012.md`:** Create a new review file to document these observations and the plan for further investigation.

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

<br><sub>Last edited: 2025-08-03 16:52:05</sub>
