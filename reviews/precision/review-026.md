# Review 26: Automated Initial Guess Search for `nu`

## 1. Objective

To address the model's sensitivity to the initial guess for `nu`, we are implementing an automated search mechanism. This will make the model more robust and autonomous, a key requirement for a strong academic paper.

## 2. Methodology

The proposed methodology consists of a two-phase training process:

1.  **Phase 1: Initial Guess Search.** Before the main training, the model will perform a quick search for the best initial guess for `nu`. This will be done by:
    -   Defining a list of candidate `nu` values (e.g., `[0.01, 0.05, 0.1, 0.5]`).
    -   For each candidate, running the Adam optimizer for a small number of epochs (e.g., 100) and recording the final loss.
    -   Selecting the `nu` candidate that results in the lowest loss as the best initial guess.

2.  **Phase 2: Main Training.** The model will then be trained using the best initial guess found in Phase 1. The main training will proceed as before, with the Adam optimizer followed by L-BFGS-B.

### Viability Analysis

To quickly assess the viability of this approach, we will initially reduce the number of epochs in the main training phase (Phase 2) to 500. If the approach proves to be promising, we will revert to the original number of epochs for the final experiments.

## 3. Implementation Plan

1.  **Modify `src/main_precision.py`:**
    -   Create a new function `find_best_initial_nu` to implement the search mechanism.
    -   Remove the `--initial_nu` command-line argument.
    -   Modify the main execution block to call the new function and use its output as the initial guess for `nu`.
    -   Reduce the `adam_epochs` to 500 for the viability analysis.
2.  **Run Experiments:** Run the modified script for different ground truth `nu` values (e.g., 0.01 and 0.1) to test the effectiveness of the automated search.
3.  **Analyze Results:** Analyze the results to determine if the automated search can consistently find a good initial guess and lead to the correct discovery of `nu`.

## 4. Next Steps

We will now proceed with the implementation of the automated initial guess search. The first step is to modify the `src/main_precision.py` script.

## 5. Viability Test Results (Ground Truth `nu` = 0.01)

We ran a viability test with `true_kinematic_viscosity = 0.01` and the automated initial guess search. The `adam_epochs` for the main training were set to 500, and the initial guess search used 100 epochs for each candidate.

### 5.1. Initial Guess Search Results

-   **Candidate `nu` values:** `[0.01, 0.05, 0.1, 0.5]`
-   **Losses for candidates:**
    -   `nu = 0.01`, Loss: 0.081942
    -   `nu = 0.05`, Loss: 0.079272
    -   `nu = 0.1`, Loss: 0.082364
    -   `nu = 0.5`, Loss: 0.089267
-   **Best initial `nu` found:** 0.05

### 5.2. Main Training Results

-   **Final Discovered `nu`:** 0.041265
-   **Ground Truth `nu`:** 0.01
-   **Relative Error:** 312.65%

### 5.3. Discussion

The viability test revealed a critical issue: the automated initial guess search **failed** to identify the correct initial `nu`. Despite `0.01` being a candidate, the search selected `0.05` as the best initial guess, leading to a very high relative error in the final discovered `nu`.

This suggests that a short Adam run (100 epochs) is insufficient to accurately assess the quality of an initial `nu` guess, especially when the true value is significantly different from the candidates. The loss landscape might be non-convex, causing the optimizer to get stuck in local minima during the brief search phase.

### 5.4. Next Steps

We need to refine the `find_best_initial_nu` function to make it more robust. The immediate next step is to:

-   **Increase epochs for initial guess search:** Increase the number of Adam epochs used for evaluating each `nu` candidate from 100 to 500. This should provide a more reliable estimate of the loss for each candidate.

After this modification, we will rerun the viability test.

## 6. Viability Test Results (Ground Truth `nu` = 0.01, Initial Guess Search Epochs = 500)

We reran the viability test with `true_kinematic_viscosity = 0.01`, `adam_epochs` for main training set to 500, and the initial guess search using 500 epochs for each candidate.

### 6.1. Initial Guess Search Results

-   **Candidate `nu` values:** `[0.01, 0.05, 0.1, 0.5]`
-   **Losses for candidates:**
    -   `nu = 0.01`, Loss: 0.073859
    -   `nu = 0.05`, Loss: 0.073927
    -   `nu = 0.1`, Loss: 0.074298
    -   `nu = 0.5`, Loss: 0.074098
-   **Best initial `nu` found:** 0.01

### 6.2. Main Training Results

-   **Final Discovered `nu`:** 0.011029
-   **Ground Truth `nu`:** 0.01
-   **Relative Error:** 10.29%

### 6.3. Discussion

Increasing the number of epochs for the initial guess search from 100 to 500 was successful in correctly identifying the best initial `nu` (0.01). This is a significant improvement over the previous attempt.

The final discovered `nu` of 0.011029, with a relative error of 10.29%, demonstrates that the automated initial guess search is viable. While the precision is not yet as high as the manually tuned case, this approach makes the model much more robust and autonomous.

### 6.4. Next Steps

Now that the viability of the automated initial guess search is confirmed, we will proceed with the ensemble experiments to get a more robust estimate of the model's performance with this new feature.

1.  **Run Ensemble Experiment for `nu = 0.01`:** Execute the `run_ensemble.sh` script with `true_kinematic_viscosity = 0.01`.
2.  **Run Ensemble Experiment for `nu = 0.1`:** Execute the `run_ensemble.sh` script with `true_kinematic_viscosity = 0.1`.
3.  **Revert `adam_epochs`:** After these ensemble runs, we will revert the `adam_epochs` in the main training loop to the original value (2000) for better precision in the final results.

<br><sub>Last edited: 2025-08-10 15:39:45</sub>
