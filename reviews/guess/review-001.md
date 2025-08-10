# Review: Initial Guess Experiment Setup

## Objective

The primary objective of this experiment, named "guess", is to investigate the impact of randomly generated initial guesses for the kinematic viscosity coefficient (nu) on the PINN's ability to discover the true nu value. This involves transitioning from a fixed set of nu_candidates to a dynamic set of random candidates.

Specifically, for each of 10 randomly generated `nu_candidates`, the full multi-stage PINN training process will be executed. The "Final Discovered nu" (after full training) will be compared against the `nu_candidate` and the `Ground Truth nu`. The Mean Squared Error (MSE) will also be recorded for each run.

## Current State Analysis

The existing `main_precision.py` script serves as the baseline for this experiment. It currently uses a predefined, fixed set of `nu_candidates` for its "Automated Initial Guess Search". The script also performs the PINN training and evaluation, which will be leveraged for the "guess" experiment.

## Proposed Steps

1. **Refactor `main_guess.py` (Major Change)**:
      * Modify the `train` method to iterate through the 10 random `nu_candidates`.
      * For *each* `nu_candidate`:
           * Re-initialize the PINN model's variables to ensure a fresh start.
           * Set the `log_nu_pinn` to `tf.math.log(nu_candidate)`.
           * Execute the full training pipeline (Data-Only Adam, Full Adam, L-BFGS-B).
           * Record the `nu_candidate`, the `final_discovered_nu` (after this full training), the relative error (against `true_nu`), and the total MSE. Store these in a list.
      * The `train` method will return this list of results for all candidates.
      * Adjust the `if __name__ == "__main__":` block to process and print these results in a comprehensive table, showing the comparison between `nu_candidate`, `final_discovered_nu`, relative error, and total MSE for each run.



## Experiment Results and Analysis (Seed 1)

### Experiment Parameters

* **Seed:** 1
* **Ground Truth nu:** 0.05
* **Number of Random nu Candidates:** 10 (range 0.01 to 0.5)
* **Adam Epochs:** 2000
* **Data-Only Epochs:** 10000
* **Lambda Data Weight:** 1.0
* **Lambda PDE Weight:** 1.0
* **PDE Points:** 20000
* **Neural Network Architecture (layers):** [3, 60, 60, 60, 60, 2]

### Final Results for Each Initial Guess Candidate (Seed 1)

| Candidate nu | Final Discovered nu | Relative Error (%) | Total MSE    |
|:------------ |:------------------- |:------------------ |:------------ |
| 0.151083     | 0.129235            | 158.47             | 1.316823e-01 |
| 0.334714     | 0.126401            | 152.80             | 1.335342e-01 |
| 0.069570     | 0.058011            | 16.02              | 1.364362e-01 |
| 0.214881     | 0.191152            | 282.30             | 1.277465e-01 |
| 0.384852     | 0.125341            | 150.68             | 1.336213e-01 |
| 0.021782     | 0.021796            | 56.41              | 1.377428e-01 |
| 0.171536     | 0.122459            | 144.92             | 1.341073e-01 |
| 0.390646     | 0.118917            | 137.83             | 1.342885e-01 |
| 0.059312     | 0.050681            | 1.36               | 1.356950e-01 |
| 0.282010     | 0.174015            | 248.03             | 1.311766e-01 |

### Overall HPC Performance Metrics (Seed 1)

* **Data Preparation Duration:** 4.17 seconds
* **Model Initialization Duration:** 0.04 seconds
* **Total Execution Duration:** 12654.71 seconds

<br><sub>Last edited: 2025-08-20 09:41:28</sub>
