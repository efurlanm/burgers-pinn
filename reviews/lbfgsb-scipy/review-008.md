# Review 008: L-BFGS-B Convergence Achieved, but Nu Inaccurate and Adam Training Issue

## Precise and Detailed Analysis

The `burgers2d-03.py` script was executed again after relaxing the L-BFGS-B `tolerance` to `1e-5`. The execution completed without any `stderr` warnings.

Analysis of the updated `log.txt` reveals the following:

-   **Adam Optimizer Phase:**
    -   The initial `nu_pinn` guess was `0.100000`.
    -   Adam training appears to have run for only 0 epochs in the log, which is unexpected given the `epochs_adam=100` setting. This indicates that the Adam phase was effectively skipped or not properly logged, which is a concern as Adam is intended to provide a good initial guess for L-BFGS-B.

-   **L-BFGS-B Optimizer Phase:**
    -   L-BFGS-B training finished.
    -   **Convergence:** `L-BFGS-B converged: True`. This is a significant positive development, indicating that the optimizer successfully reached the specified tolerance.
    -   **Number of Iterations:** `L-BFGS-B number of iterations: 3571`.
    -   **Function Evaluations:** `L-BFGS-B function evaluations: 10505`.
    -   **Final Discovered `nu`:** `0.014110`. While the optimizer converged, this value is still significantly different from the `True nu: 0.05`.

The convergence of L-BFGS-B is a major step forward. However, the inaccurate `nu` value and the apparent skipping of the Adam training phase are critical issues. The Adam optimizer is crucial for providing a good initial set of weights and biases to the L-BFGS-B optimizer, especially for the `nu` parameter. Without proper Adam pre-training, L-BFGS-B might converge to a suboptimal local minimum.

## Next Steps

1.  **Restore Adam Epochs:**
    -   Revert the `epochs_adam` back to `1000` in `burgers2d-03.py` to ensure the Adam optimizer has sufficient time to train the network and provide a better initial guess for L-BFGS-B.

2.  **Re-run and Analyze:**
    -   Execute `burgers2d-03.py` again with the restored Adam epochs and the relaxed L-BFGS-B tolerance (`1e-5`).
    -   Analyze the `log.txt` output to check if Adam training now runs for the specified epochs and how this impacts the L-BFGS-B convergence and the discovered `nu` value.

3.  **Document Findings in `review-008.md`:** Create a new review file to document these observations and the plan for further investigation.

<br><sub>Last edited: 2025-08-03 14:02:41</sub>
