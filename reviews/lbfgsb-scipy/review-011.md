# Review 011: Regression in L-BFGS-B Convergence with Increased Network Complexity

## Precise and Detailed Analysis

The `burgers2d-03.py` script was executed again after increasing the number of neurons per layer from 20 to 40 and `num_pde_points` from 20000 to 40000. The execution completed without any `stderr` warnings.

Analysis of the updated `log.txt` reveals the following:

-   **Adam Optimizer Phase:**
    -   The initial `nu_pinn` guess was `0.100000`.
    -   Adam training ran for 900 epochs.
    -   The loss decreased from `2.574193` to `0.069292`.
    -   The discovered `nu` value after Adam training was `0.099433`.

-   **L-BFGS-B Optimizer Phase:**
    -   L-BFGS-B training finished.
    -   **Convergence:** `L-BFGS-B converged: False`. This is a regression from the previous run where it converged with `tolerance=1e-4`.
    -   **Number of Iterations:** `L-BFGS-B number of iterations: 5`. This is extremely low, indicating that the optimizer stopped almost immediately.
    -   **Function Evaluations:** `L-BFGS-B function evaluations: 69`.
    -   **Final Discovered `nu`:** `0.099439`. This value is very close to the `nu` value after Adam training, suggesting that L-BFGS-B performed very few updates.

The non-convergence of L-BFGS-B and its immediate stopping after only 5 iterations is a significant setback. This suggests that increasing the network complexity (more neurons) and the number of PDE points has made the optimization landscape more challenging for L-BFGS-B to navigate from the Adam-trained state. The `tolerance=1e-4` that previously allowed convergence is now too strict for this more complex setup, causing the optimizer to terminate prematurely.

## Next Steps

1.  **Re-evaluate L-BFGS-B `tolerance` and `max_iterations`:**
    -   Given the very low number of iterations, the optimizer is likely hitting an internal stopping criterion related to the `tolerance` almost immediately.
    -   Try relaxing the `tolerance` further to `1e-3` to see if it can converge with the larger network.
    -   Consider increasing `max_iterations` to a very high value (e.g., `200000` or `500000`) to ensure that the optimizer has ample opportunity to run if it's not hitting the tolerance.

2.  **Increase Adam Epochs:**
    -   With a larger network, the Adam pre-training might need more epochs to reach a sufficiently good starting point for L-BFGS-B. Consider increasing `epochs_adam` to `2000` or `3000`.

3.  **Document Findings in `review-011.md`:** Create a new review file to document these observations and the plan for further investigation.

<br><sub>Last edited: 2025-08-03 16:16:04</sub>
