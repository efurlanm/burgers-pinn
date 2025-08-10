# Burgers-PINN Project

This repository contains the source code and associated scripts for solving the 2D Burgers' equation using Physics-Informed Neural Networks (PINNs), focusing on hyperparameter optimization and various experimental setups. This README serves as an index to help navigate the project files.

MIRANDA, E. F.; SOUTO, R. P.; STEPHANY, S. Descoberta de parâmetro na equação 2D de burgers por rede neural informada pela física. Dec. 2025. **CIACA CIAWI 2025 Proceedings**. Lisboa: IADIS, Dec. 2025. pp. 87–94. Available at: <https://www.iadisportal.org/ciaca-ciawi-2025-proceedings> (in Portuguese)

Online version of the paper, including an appendix detailing the progress since submission (in English): <https://efurlanm.github.io/burgers-pinn/>

## Getting Started

For a first-time reader, we recommend starting with:

* `scripts/pinn_model.py`: To understand the PINN architecture.
* `scripts/main_hopt_unified.py`: As the primary script for running experiments.
* `configs/config.template.yaml`: To see the available hyperparameters for an experiment.

## Repository Structure

- `README.md`: This file.
- `scripts/`: Contains all Python (`.py`) and shell (`.sh`) scripts for running experiments, analysis, and other tasks.
- `configs/`: Contains configuration files (`.yaml`) for experiments.
- `environment/`: Contains Conda environment files (`.yml`) for reproducing the software environment.
- `reports/`: Contains analysis reports and documents related to the research.
- `docs/`: Online version of the paper.
- `presentation/`: Slides used in the presentation during the conference.

## How to Run an Experiment

1. **Set up the environment:**
   
   ```bash
   conda env create -f environment/environment.yml
   conda activate tf2
   ```

2.  **Generate Validation Data:** Before running any training, generate the hold-out validation dataset. This is required to calculate the generalization error reported in the results.

    ```bash
    python scripts/generate_validation.py
    ```

2. **Create an experiment configuration:** Copy the template `configs/config.template.yaml` to a new file, for example, `configs/my_experiment.yaml`.

3. **Modify the configuration:** Open `configs/my_experiment.yaml` and adjust the parameters for your experiment.

4. **Run the experiment:**
   
   ```bash
   python scripts/main_hopt_unified.py --config configs/my_experiment.yaml
   ```

### Reproducing the Best Result

To reproduce the best result presented in the CIACA 2025 slides (approx. 2.53% generalization error), use the specialized reproduction script. This experiment uses **Latin Hypercube Sampling (LHS)** with specific hyperparameters (Seed 2, 15k Epochs).

* **Command:**

```bash
bash scripts/run_lhs2_repro.sh
```
* **Configuration Reference:** `configs/lhs2_best.yaml`
* **Expected Result:** Validation error around **2.53%**.


## Experiments

This repository is structured to support two main experimental categories:

- **[PLATEAU Experiment](experiments/plateau/README.md)**: This section focuses on the reproducible "Base Case" experiment as described in the CIACA 2025 paper. It includes the original self-contained PINN implementation and scripts to run it.

- [**HPO Experiment \(Generalist Model\)**](experiments/hopt/README.md): This section details the Hyperparameter Optimization (HPO) efforts for developing a more generalized PINN model, utilizing techniques like Latin Hypercube Sampling. It includes scripts for HPO, analysis, and model validation.

## File Descriptions

### Core PINN Model and Training Scripts (`scripts/`)

* `scripts/pinn_model.py`: Defines the architecture and components of the Physics-Informed Neural Network (PINN).
* `scripts/main_hopt_unified.py`: The primary script for executing unified hyperparameter optimization experiments.
* `scripts/main_latin.py`: Script for running experiments utilizing Latin Hypercube Sampling (LHS) for parameter exploration.
* `scripts/main_plateau.py`: Main script implementing the plateau training strategy, likely originating from the base project.
* `scripts/main_validation.py`: Script dedicated to validating the trained PINN models against known solutions or test data.

### Analysis and Utility Scripts (`scripts/`)

* `scripts/analyze_ensemble_results.py`: Analyzes the aggregated results from ensemble model runs.
* `scripts/analyze_latin_results.py`: Processes and analyzes data generated from Latin Hypercube Sampling experiments.
* `scripts/collect_results.py`: Utility script to gather and consolidate results from various experiment runs.
* `scripts/compare_results.py`: Compares outcomes from different experimental setups or model configurations.
* `scripts/find_best_lhs_seed.py`: Identifies the optimal random seed used in Latin Hypercube Sampling experiments based on performance metrics.
* `scripts/find_best_trial.py`: Locates the best-performing trial within a set of hyperparameter optimization runs.
* `scripts/generate_timing_report.py`: Generates reports detailing the execution times and performance bottlenecks of experiments.
* `scripts/plot_results.py`: Visualizes the results of experiments, generating plots and figures.

### Configuration and Environment (`configs/` and `environment/`)

* `configs/config.template.yaml`: A template file for experiment configuration, allowing externalized parameter management.
* `environment/environment.yml`: Conda environment definition file for setting up the project's Python dependencies.
* `environment/base_environment.yml`: Conda environment definition file specific to the original (base) project setup.
* `configs/lhs2_best.yaml`: Configuration file documenting the exact hyperparameters for the best result.

### Experiment Management and Execution (`scripts/`)

* `scripts/run_ensemble.sh`: Shell script to orchestrate and execute ensemble training and evaluation runs.
* `scripts/run_hopt.sh`: Shell script to initiate and manage hyperparameter optimization processes.
* `scripts/run_latin.sh`: Shell script to run Latin Hypercube Sampling experiments.
* `scripts/run_lhs2_repro.sh`: Automates the reproduction of the best-performing LHS model (Seed 2, Extended Training).

## Acknowledgment

Authors thank LNCC (National Laboratory for Scientific Computing) for grant 205341 AMPEMI (call 2020-I), which allows access to the Santos Dumont supercomputer (node of the SINAPAD, the Brazilian HPC system). This study was financed in part by the Coordination for the Improvement of Higher Education Personnel (CAPES), Brazil, finance Code 001, and also by the CNPq Project 446053/2023-6. The authors also thank the Brazilian Ministry of Science, Technology and Innovation, and the Brazilian Space Agency.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0. See the `LICENSE` file for details.

## Notes

This project distinguishes between two specific uses of AI. The core scientific methodology relies on Physics-Informed Neural Networks (PINNs) to solve the 2D Burgers' equation. Distinctly, Generative AI (GenAI) was utilized to streamline administrative workflows, including literature scanning, file formatting, and file organization. By automating these repetitive tasks, the authors could dedicate significantly more time to rigorous human review, critical analysis, and final validation. Thus, GenAI served as an efficiency tool to support — not replace — intellectual work, ensuring that scientific standards remained uncompromised. Standard protocols were strictly followed to protect sensitive data.

<br><sub>Last edited: 2026-01-20</sub>
