# PLATEAU Experiment (Base Case from CIACA 2025 paper)

This directory contains documentation and scripts related to the "PLATEAU" experiment, which is the base case described in the CIACA 2025 paper (`BASE/ciaca-2025-66.pdf`). This experiment focuses on the discovery of kinematic viscosity (`nu`) for a single, fixed `nu_true` value, using a self-contained Physics-Informed Neural Network (PINN) implementation.

## Overview

The `main_plateau.py` script implements a hybrid optimization approach (Adam + L-BFGS-B) and incorporates advanced techniques like adaptive weighting and curriculum learning, as detailed in the paper.

## Running the Experiment

The experiment can be run using the `run_plateau.sh` script. This script executes `scripts/main_plateau.py` with predefined or user-specified parameters.

### Usage

```bash
# To run with default parameters:
./scripts/run_plateau.sh

# To run with custom parameters:
# Parameters are: NU_INITIAL SEED NU_TRUE NOISE_LEVEL ADAM_EPOCHS_STAGE1 EPOCHS_INVERSE_ADAM_STAGE2 EPOCHS_INVERSE_ADAM_PRETRAIN
./scripts/run_plateau.sh 0.02 42 0.02 0.005 6000 6000 1500
```

### Script Parameters

The `run_plateau.sh` script accepts the following optional command-line arguments, which override the defaults:

1.  `NU_INITIAL` (float, default: `0.01`): Initial kinematic viscosity for the curriculum training range (`nu_min_train`).
2.  `SEED` (int, default: `1`): Random seed for reproducibility.
3.  `NU_TRUE` (float, default: `0.05`): True kinematic viscosity for data generation and the inverse problem.
4.  `NOISE_LEVEL` (float, default: `0.0`): Percentage of Gaussian noise to add to the data (e.g., `0.01` for 1%).
5.  `ADAM_EPOCHS_STAGE1` (int, default: `5000`): Number of Adam epochs for Stage 1 (parametric PINN training).
6.  `EPOCHS_INVERSE_ADAM_STAGE2` (int, default: `5000`): Number of Adam epochs for Stage 2 (inverse problem, `nu` discovery).
7.  `EPOCHS_INVERSE_ADAM_PRETRAIN` (int, default: `1000`): Number of Adam pre-training epochs for Stage 2 inverse problem.

### Output

-   **Results:** `.npz` files containing experiment data will be saved in `results/plateau/`.
-   **Logs:** Detailed console output will be redirected to `.log` files in `logs/plateau/`.

### Core Script

-   `scripts/main_plateau.py`: The Python script implementing the PINN model and training logic for this experiment. It is self-contained and includes its own `PINN_Burgers2D` class definition.
<br><sub>Last edited: 2025-12-09 13:50:20</sub>
