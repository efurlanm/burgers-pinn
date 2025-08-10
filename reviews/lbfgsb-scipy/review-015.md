# Review 015: SciPy L-BFGS-B Convergence but Inaccurate Nu and Adam Logging Issue

## Precise and Detailed Analysis

Analysis of `log_scipy_test.txt` reveals the following:

-   **Adam Optimizer Phase:**
    -   The initial `nu_pinn` guess was `0.100000`.
    -   Adam training ran for 0 epochs (only Adam Epoch 0 is printed, which is the initial state before any updates). This is because `epochs_adam` was set to `50`, but the print statement is `if epoch % 100 == 0:`. This means only epoch 0 is printed.
    -   The loss at epoch 0 was `1.735462`.
    -   The discovered `nu` value after Adam training was `0.100000`.

-   **L-BFGS-B Optimizer Phase (SciPy):**
    -   L-BFGS-B training finished.
    -   **Convergence:** `L-BFGS-B converged: True`. This is a positive sign, as the SciPy optimizer converged.
    -   **Message:** `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`. This indicates convergence based on a relative reduction in the function value.
    -   **Number of Iterations:** `L-BFGS-B number of iterations: 1`. This is extremely low.
    -   **Function Evaluations:** `L-BFGS-B function evaluations: 30`.
    -   **Final Discovered `nu`:** `0.100000`. This value is exactly the initial guess and the true `nu` is `0.05`.

The SciPy L-BFGS-B optimizer converged, which is good, but it did so in only 1 iteration and the `nu` value remained at its initial guess. This strongly suggests that the optimizer found a very flat region or a local minimum immediately and stopped, without actually optimizing the `nu` parameter or the network weights effectively. The very low number of Adam epochs (effectively 0 due to the print condition) means the L-BFGS-B started from a very unoptimized state.

## Next Steps

1.  **Correct Adam Epoch Logging:** Adjust the Adam logging condition to `if epoch % 10 == 0:` or similar, so we can actually see the progress of Adam training.
2.  **Increase Adam Epochs for SciPy Test:** Increase `epochs_adam` to a more reasonable number (e.g., `500` or `1000`) for the SciPy test to ensure Adam has a chance to pre-train the network effectively before L-BFGS-B takes over. This is crucial for L-BFGS-B to start from a better position and potentially find a more accurate `nu`.
3.  **Re-run SciPy Test:** Execute `burgers2d-03-scipy-test.py` with the updated Adam epochs and analyze the results.
4.  **Document Findings in `review-015.md`:** Create a new review file to document these observations and the plan for further investigation.

<br><sub>Last edited: 2025-08-03 21:49:27</sub>
