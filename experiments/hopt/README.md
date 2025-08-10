# HPO Experiment (Hyperparameter Optimization for Generalist Model)

This directory contains documentation and scripts related to the Hyperparameter Optimization (HPO) experiments for the generalist PINN model. This approach uses Latin Hypercube Sampling (LHS) or random sampling for `nu` values to train a model capable of generalizing across a range of kinematic viscosities.

## Overview

The HPO experiments involve:
-   `scripts/main_hopt_unified.py`: The primary script for running hyperparameter optimization trials using `hyperopt` with random sampling for `nu`.
-   `scripts/main_latin.py`: A specialized script for running HPO trials using `hyperopt` with Latin Hypercube Sampling (LHS) for `nu` values, ensuring a more uniform exploration of the parameter space.
-   `scripts/pinn_model.py`: The core Python module that defines the `PINN_Burgers2D` architecture, loss functions, and data generation logic, imported by both `main_hopt_unified.py` and `main_latin.py`.

## Running the Experiments

There are two main ways to run the HPO experiments:

### 1. Random Sampling HPO (`main_hopt_unified.py`)

This script can run in two modes: HPO (`--optimize`) or single-run.

#### Usage (HPO Mode)

To start a hyperparameter optimization run using `hyperopt` with random sampling for `nu`:

```bash
# To run with default HPO space and save trials to hyperopt_trials.pkl
./scripts/run_hopt.sh
```

**Note:** The `run_hopt.sh` script executes `main_hopt_unified.py` in a loop, running multiple sequential jobs. Each job uses the `--optimize` flag and saves its trials to a specific `.pkl` file.

#### HPO Search Space (Default in `main_hopt_unified.py`)

When `--optimize` is enabled, the following parameters are searched:
-   `seed`: [42, 97, 123]
-   `neurons`: [40, 50, 60]
-   `layers`: [4, 5]
-   `learning_rate`: log-uniform between `1e-4` and `1e-3`
-   `adam_epochs_stage1`: [5000, 8000]
-   `epochs_data_only_stage1`: [500, 1000]
-   `num_pde_points_stage1`: [10000, 20000]
-   `epochs_inverse_adam_pretrain`: [1000, 2000]
-   `num_datasets_gene`: [10, 15]
-   `noise_level`: uniform between `0.0` and `0.04`

#### Usage (Single Run Mode - using `main_hopt_unified.py` directly)

To run a single experiment with a fixed set of parameters (without HPO), modify the `default_params` dictionary in `scripts/main_hopt_unified.py` or specify them through command line (though `main_hopt_unified.py` does not currently parse individual parameters in non-optimize mode, it uses the `default_params` dict directly).

```bash
# Example of running a single experiment (requires modifying default_params in the script)
# python scripts/main_hopt_unified.py
```

### 2. Latin Hypercube Sampling HPO (`main_latin.py`)

This script is also designed to run in HPO mode, but specifically uses LHS for sampling `nu` values. It is typically run via `run_latin.sh`.

#### Usage (HPO Mode via `run_latin.sh`)

To run a series of Latin Hypercube Sampling experiments:

```bash
# This script executes main_latin.py multiple times with different seeds.
./scripts/run_latin.sh
```

#### HPO Search Space (Default in `main_latin.py`)

When `--optimize` is enabled, the following fixed parameters are used:
-   `seed`: [42] (or the seed passed to `run_latin.sh`)
-   `neurons`: [50]
-   `layers`: [4]
-   `learning_rate`: [0.000229]
-   `adam_epochs_stage1`: [6000]
-   `epochs_data_only_stage1`: [1500]
-   `num_pde_points_stage1`: [15000]
-   `epochs_inverse_adam_pretrain`: [2000]
-   `num_datasets_gene`: [19]
-   `noise_level`: [0.0399]

#### Usage (Single Run Mode - using `main_latin.py` directly)

The `main_latin.py` script *does* parse individual parameters for a single run when `--optimize` is false.

```bash
# Example of running a single LHS experiment:
python scripts/main_latin.py \
    --seed 1 \
    --run_id "my_lhs_run_1" \
    --results_dir "results/latin_custom" \
    --adam_epochs_stage1 7000
```

### Output

-   **Results:** `.npz` and `.pkl` (for `hyperopt` trials) files will be saved in `results/hopt/` for `main_hopt_unified.py` and `results/latin/` or user-specified directories for `main_latin.py`.
-   **Logs:** Detailed console output will be redirected to `.log` files in `logs/hopt_parallel/` or `logs/latin/` respectively.

### Core Scripts and Dependencies

-   `scripts/pinn_model.py`: Defines the fundamental PINN architecture and helper functions (e.g., `swish_activation`, `generate_ground_truth_data`). This module is imported by both `main_hopt_unified.py` and `main_latin.py`.
-   `scripts/main_hopt_unified.py`: Main script for random sampling HPO.
-   `scripts/main_latin.py`: Main script for Latin Hypercube Sampling experiments.
-   `scripts/run_hopt.sh`: Shell script to automate multiple `main_hopt_unified.py` runs.
-   `scripts/run_latin.sh`: Shell script to automate multiple `main_latin.py` runs with LHS.
-   `scripts/collect_results.py`: Utility to analyze HPO results.
-   `scripts/find_best_lhs_seed.py`: Utility to find the best seed from LHS runs.
-   `scripts/find_best_trial.py`: Utility to find the best trial from HPO runs.

<br><sub>Last edited: 2025-12-09 13:59:21</sub>
