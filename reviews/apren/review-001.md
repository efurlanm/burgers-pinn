# Review 001: Learning Rate Experiment Plan

This review outlines the plan for the "apren" experiment, which focuses on analyzing the impact of different learning rates on the convergence of the Physics-Informed Neural Network (PINN) for the 2D Burgers' equation parameter discovery problem.

## File Review

The primary file for this experiment is `main_apren.py`, which is a copy of `main_precision.py`. The code implements a PINN using TensorFlow to solve the inverse problem of discovering the viscosity parameter (`nu`) of the 2D Burgers' equation.

## Key Findings from Original Code (`main_precision.py`)

- The model uses the Adam optimizer with a fixed learning rate.
- The network architecture, number of training epochs, and data points are pre-defined.
- The script is set up for a single run, not for a systematic variation of hyperparameters like the learning rate.

## Internet Research

A brief search on "PINN hyperparameter tuning learning rate" confirms that the learning rate is a critical parameter. A learning rate that is too high can cause the optimizer to overshoot the minimum, leading to divergence. A learning rate that is too low can result in very slow convergence or getting stuck in a local minimum. Techniques like learning rate scheduling or using adaptive learning rates are common, but for this experiment, we will focus on testing a range of fixed learning rates to understand their direct impact.

## Proposed Next Steps

1.  Modify `main_apren.py` to accept the learning rate as a command-line argument.
2.  Create a loop that iterates 10 times. In each iteration:
    -   Generate a random learning rate within a reasonable range (e.g., between 1e-5 and 1e-2).
    -   Run the training process from `main_apren.py` with the generated learning rate.
    -   Capture the discovered `nu` value and calculate the relative error against the true `nu`.
3.  Store the results (learning rate, discovered `nu`, relative error) for each run.
4.  At the end of the script, print a summary table with the collected data.
5.  Redirect the output of the script to a log file in the `logs/` directory.
6.  Update `FILES.md` to reflect the new files and directories.

<br><sub>Last edited: 2025-08-20 11:57:13</sub>
