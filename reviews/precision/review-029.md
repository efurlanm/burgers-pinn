# Review 29: Attempt to Port Improvements to `main_scipy.py`

## 1. Objective

This review documents the attempt to port key improvements, specifically the Data-Guided PINN (DG-PINN) and the automated initial guess search, from `main_precision.py` to `main_scipy.py`. The goal was to harmonize the methodologies across both approaches to enable a fairer comparison and potentially improve the `lbfgsb-scipy` method's `nu` discovery precision.

## 2. Methodology

The following changes were integrated into `src/main_scipy.py`:

-   **Data-Guided PINN (DG-PINN):** The `train_data_only` method was added to the `PINN_Burgers2D` class, and the `train` method was modified to include a data-only pre-training phase (10,000 epochs).
-   **Automated Initial Guess Search:** The `find_best_initial_nu` method was added to the `PINN_Burgers2D` class, and the `train` method was modified to use its output as the initial guess for `nu`. The search evaluated candidates `[0.01, 0.05, 0.1, 0.5]` with 500 Adam epochs each.
-   **Command-Line Argument for Random Seed:** The `seed_value` was made configurable via `argparse`.
-   **Output Filename with Seed:** The output `.npz` filename was modified to include the random seed.
-   **HPC Performance Metrics:** Detailed timing metrics were added.

After these modifications, `main_scipy.py` was executed with `seed = 1` and `true_nu = 0.05`.

## 3. Results

The execution of the modified `main_scipy.py` (logged in `logs/main_scipy_test_with_improvements_seed_1.txt`) yielded the following key results:

-   **Ground Truth `nu`:** 0.05
-   **Best initial `nu` found by automated search:** 0.01 (This is incorrect, as the true nu is 0.05)
-   **Final Discovered `nu`:** 0.014012

### 3.1. HPC Performance Metrics

-   Data Preparation Duration: 0.01 seconds
-   Model Initialization Duration: 3.53 seconds
-   Total Execution Duration: 1146.74 seconds

## 4. Discussion

Despite integrating DG-PINN and the automated initial guess search, the `main_scipy.py` script still **failed to accurately discover the true `nu` value of 0.05**. The automated search incorrectly identified 0.01 as the best initial `nu`, and the subsequent training converged to 0.014012, which is a significant error.

This indicates that the improvements that proved beneficial for `main_precision.py` are not directly transferable or sufficient to resolve the `nu` discovery challenges in `main_scipy.py` for this specific problem. The underlying architecture or optimization landscape of the `lbfgsb-scipy` approach might have different sensitivities or require different tuning strategies.

## 5. Conclusion and Next Steps

The attempt to improve `main_scipy.py` by porting `main_precision.py`'s enhancements was unsuccessful in achieving accurate `nu` discovery for the `nu=0.05` case. Given the time constraints and the primary focus on the "precision" part of the project, further in-depth debugging and tuning of `main_scipy.py` will be deferred.

For the academic paper, we will:

-   **Revert `main_scipy.py`:** Restore `src/main_scipy.py` to its state before these improvements were ported, ensuring consistency with the `main_scipy_nodgpinn.py` snapshot.
-   **Acknowledge Limitations:** Clearly state in the paper that the `lbfgsb-scipy` approach, even with attempted improvements, struggled with `nu` discovery for this problem, contrasting it with the success of the "precision" approach.
-   **Future Work:** Suggest further investigation into the `lbfgsb-scipy` method's specific challenges as an area for future research.

<br><sub>Last edited: 2025-08-10 18:28:02</sub>
